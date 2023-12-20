"""
This could be moved to something more general, hopefully.
The idea is that we could train any GraphCast-like model, no matter the
resolution, variables, etc with these routines

These functions are taken from the GraphCast demo:
    https://github.com/google-deepmind/graphcast/blob/main/graphcast_demo.ipynb

Note that the functions:
    - run_forward.init
    - run_forward.apply
    - loss_fn.apply
    - grads_fn
are jitted with wrappers that give and takeaway things like the task and model configs, and the state and params
It looks like we don't have to do this with the emulator object being passed because
we are registering it as a pytree.
See the last few methods and lines of simple_emulatory.py, following this guidance:
    https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
"""

from functools import partial
import numpy as np
from jax import jit, value_and_grad, tree_util
from jax.random import PRNGKey
import optax
import haiku as hk

from graphcast.graphcast import GraphCast
from graphcast.casting import Bfloat16Cast
from graphcast.autoregressive import Predictor
from graphcast.xarray_tree import map_structure
from graphcast.normalization import InputsAndResiduals
from graphcast.xarray_jax import unwrap_data


def construct_wrapped_graphcast(emulator):
    """Constructs and wraps the GraphCast Predictor object"""

    predictor = GraphCast(emulator.model_config, emulator.task_config)

    # handle inputs/outputs float32 <-> BFloat16
    # ... and so that this happens after applying
    # normalization to inputs & targets
    predictor = Bfloat16Cast(predictor)
    predictor = InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=emulator.norm["stddiff"],
        mean_by_level=emulator.norm["mean"],
        stddev_by_level=emulator.norm["std"],
    )

    # Wraps everything so the one-step model can produce trajectories
    predictor = Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(emulator, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(emulator)
    return predictor(
        inputs,
        targets_template=targets_template,
        forcings=forcings
    )


@hk.transform_with_state
def loss_fn(emulator, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(emulator)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return map_structure(
        lambda x: unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))


def grads_fn(params, state, emulator, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, PRNGKey(0), emulator, i, t, f
        )
        return loss, (diagnostics, next_state)

    (loss, (diagnostics, next_state)), grads = value_and_grad(_aux, has_aux=True)(
        params,
        state,
        inputs,
        targets,
        forcings,
    )
    return loss, diagnostics, next_state, grads


def optimize(params, state, optimizer, emulator, input_batches, target_batches, forcing_batches):

    opt_state = optimizer.init(params)

    def optim_step(params, state, opt_state, emulator, inputs, targets, forcings):
        """Note that this has to be definied within optimize so that we do not
        pass optimizer as an argument. Otherwise we get some craazy jax errors"""

        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, PRNGKey(0), emulator, i, t, f
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = value_and_grad(_aux, has_aux=True)(
            params,
            state,
            inputs,
            targets,
            forcings,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, loss, diagnostics, opt_state, grads

    def with_params(fn):
        return partial(fn, params=params, state=state)

    optim_step_jitted = with_params( jit(
        optim_step
    ) )

    for i in input_batches["batch"].values:

        params, loss, diagnostics, opt_state, grads = optim_step_jitted(
            opt_state=opt_state,
            emulator=emulator,
            inputs=input_batches.sel(batch=[i]),
            targets=target_batches.sel(batch=[i]),
            forcings=forcing_batches.sel(batch=[i]),
        )
        mean_grad = np.mean(tree_util.tree_flatten(tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
        print(f"Step = {i}, loss = {loss}, mean(|grad|) = {mean_grad}")
        print("diagnostics: ")
        print(diagnostics)
        print()

    return params, loss, diagnostics, opt_state, grads
