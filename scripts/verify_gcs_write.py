
import xarray as xr
from timer import Timer

if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Initializing job")

    localtime.start("opening and loading dataset")
    ds = xr.open_zarr("../zstores/local-replay-1deg-slurmtest/forecast/fv3.zarr")
    ds = ds[["tmp", "ugrd", "vgrd"]]
    ds.load()
    localtime.stop()

    localtime.start("writing to bucket")
    ds.to_zarr(
            f"gcs://{test_dir}/fv3.zarr",
            storage_options={
                "token": "/contrib/Tim.Smith/.gcs/replay-service-account.json",
                },
            )
    localtime.stop()

    walltime.stop()
