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
from jax import jit, value_and_grad
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


def optim_step(params, state, emulator, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, PRNGKey(emulator.grad_rng_seed), emulator, i, t, f
        )
        return loss, (diagnostics, next_state)

    (loss, (diagnostics, next_state)), grads = value_and_grad(_aux, has_aux=True)(
        params,
        state,
        inputs,
        targets,
        forcings,
    )
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, loss, diagnostics, state, grads


def optimize(params, state, optimizer, emulator, input_batches, target_batches, forcing_batches):

    opt_state = optimizer.init(params)

    def with_params(fn):
        return partial(fn, params=params, state=state)

    optim_step_jitted = with_params( jit(
        optim_step
    ) )

    print(" ... done jitting optim_step")

    grads_jitted = with_params( jit(
        grads_fn
    ) )

    print(" ... done jitting grads_fn")

    loss, diagnostics, next_state, grads = grads_jitted(
        emulator=emulator,
        inputs=input_batches.sel(batch=[0]),
        targets=target_batches.sel(batch=[0]),
        forcings=forcing_batches.sel(batch=[0]),
    )
    return loss, grads



    #for i in input_batches["batch"].values:

    #    params, loss, diagnostics, opt_state, grads = optim_step_jitted(
    #        emulator=emulator,
    #        inputs=input_batches.sel(batch=[i]),
    #        targets=target_batches.sel(batch=[i]),
    #        forcings=forcing_batches.sel(batch=[i]),
    #    )
    #    print(f"Step = {i}, loss = {loss}")

    #return params, loss, diagnostics, opt_state, grads
