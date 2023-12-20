
from functools import partial
import xarray as xr
from jax import jit
from jax.random import PRNGKey
import optax

from ufs2arco.timer import Timer

from simple_emulator import ReplayEmulator1Degree
from train import optimize, run_forward


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

    loss, grads = optimize(
        params=params,
        state=state,
        optimizer=optimizer,
        emulator=gufs,
        input_batches=inputs,
        target_batches=targets,
        forcing_batches=forcings,
    )
    print("loss: ", loss)

    localtime.stop()

    walltime.stop("Total Walltime")
