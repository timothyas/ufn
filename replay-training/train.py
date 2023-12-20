"""
This could be moved to something more general, hopefully.
The idea is that we could train any GraphCast-like model, no matter the
resolution, variables, etc with these routines

These functions are taken from the GraphCast demo:
    https://github.com/google-deepmind/graphcast/blob/main/graphcast_demo.ipynb
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
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
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



## Jax doesn't seem to like passing configs as args through the jit. Passing it
## in via partial (instead of capture by closure) forces jax to invalidate the
## jit cache if you change configs.
#def with_emulator(fn):
#    return partial(fn, emulator=emulator)
#
#
## Always pass params and state, so the usage below is simpler
#def with_params(fn):
#    return partial(fn, params=params, state=state)
#
## Our models aren't stateful, so the state is always empty, so just return the
## predictions. This is required by our rollout code, and generally simpler.
#def drop_state(fn):
#    return lambda **kw: fn(**kw)[0]
#
#
#
#def optimize(params, optimizer, emulator, input_batches, target_batches, forcing_batches):
#
#    opt_state = optimizer.init(params)
#
#    optim_step_jitted = drop_state( with_emulator( jit( with_emulator(
#        optim_step
#    ) ) ) )
#
#    for i, (inputs, targets, forcings) in enumerate(input_batches, target_batches, forcing_batches):
#
#        params, loss, diagnostics, opt_state, grads = optim_step_jitted(
#            params, opt_state, emulator, inputs, targets, forcings)
#        print(f"Step = {i}, loss = {loss}")
#
#    return params, loss, diagnostics, opt_state, grads
