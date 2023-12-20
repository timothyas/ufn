
from functools import partial
import xarray as xr
from jax import jit
from jax.random import PRNGKey
import optax

from ufs2arco.timer import Timer

from simple_emulator import ReplayEmulator1Degree
from train import optim_step, run_forward, grads_fn



def optimize(params, state, optimizer, emulator, input_batches, target_batches, forcing_batches):

    opt_state = optimizer.init(params)

    def with_em(fn):
        return partial(fn, emulator=emulator)

    def with_params(fn):
        return partial(fn, params=params, state=state)

    # Our models aren't stateful, so the state is always empty, so just return the
    # predictions. This is required by our rollout code, and generally simpler.
    #def drop_state(fn):
    #    return lambda **kw: fn(**kw)[0]
    optim_step_jitted = with_params( jit( #with_em(
        optim_step
    ) ) #)

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


if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Starting Training")

    localtime.start("Extracting Training Batches from Replay on GCS")

    gufs = ReplayEmulator1Degree()

    ds = xr.open_zarr(
        "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr",
        storage_options={"token": "anon"},
    )

    inputs, targets, forcings = gufs.get_training_batches(
        xds=ds,
        batch_indices=[0, 100, 200],
        n_steps=1,
    )
    localtime.stop()

    localtime.start("Loading Training Batches into Memory")

    inputs.load()
    targets.load()
    forcings.load()

    localtime.stop()


    localtime.start("Initializing Optimizer and Parameters")

#   We may not need this with the pytree stuff defined
#    # A note from the GraphCast demo:
#    # Jax doesn't seem to like passing configs as args through the jit. Passing it
#    # in via partial (instead of capture by closure) forces jax to invalidate the
#    # jit cache if you change configs.
#    def with_emulator(fn):
#        return partial(fn, emulator=gufs)

    init_jitted = jit( run_forward.init )
    params, state = init_jitted(
        rng=PRNGKey(gufs.init_rng_seed),
        emulator=gufs,
        inputs=inputs.sel(batch=[0]),
        targets_template=targets.sel(batch=[0]),
        forcings=forcings.sel(batch=[0]),
    )
    optimizer = optax.adam(learning_rate=1e-4)
    localtime.stop()

    localtime.start("Starting Optimization")

    #params, loss, diagnostics, opt_state, grads = optimize(
    loss, grads = optimize(
        params=params,
        state=state,
        optimizer=optimizer,
        emulator=gufs,
        input_batches=inputs,
        target_batches=targets,
        forcing_batches=forcings,
    )

    localtime.stop()

    walltime.stop("Total Walltime")
