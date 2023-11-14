"""NOTE: This is just a first pass. Should take a more careful grid cell weighted average.
"""
import numpy as np
import xarray as xr
from timer import Timer

def preprocess_time(xds):

    # 1. get training data: 1993-1997 (inclustive)
    rds = xds.sel(time=slice(None, "1997"))
    assert rds.time[-1].dt.year == 1997
    assert rds.time[-1].dt.month == 12
    assert rds.time[-1].dt.day == 31
    assert rds.time[-1].dt.hour == 21

    # 2. Subsample to 6 hourly
    rds = rds.isel(time=slice(None, None, 2))
    assert rds.time[ 1].dt.hour == 0
    assert rds.time[-1].dt.hour == 18

    return rds


def calc_diffs_stddev_by_level(xds):

    with xr.set_options(keep_attrs=True):
        result = xds.diff("time")
        result = result.std(["grid_xt", "grid_yt", "time"])

    for key in result.data_vars:
        result[key].attrs["description"] = "std of 6 hourly difference over lat, lon, time (1993-1997, inclusive)"
    return result

def calc_stddev_by_level(xds):

    with xr.set_options(keep_attrs=True):
        result = xds.std(["grid_xt", "grid_yt", "time"])

    for key in result.data_vars:
        result[key].attrs["description"] = "std over lat, lon, time (1993-1997, inclusive)"
    return result


def calc_mean_by_level(xds):

    with xr.set_options(keep_attrs=True):
        result = xds.mean(["grid_xt", "grid_yt", "time"])

    for key in result.data_vars:
        result[key].attrs["description"] = "avg over lat, lon, time (1993-1997, inclusive)"
    return result


def main():

    rds = xr.open_zarr(
        "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr",
        storage_options={"token": "anon"},
    )

    rds = preprocess_time(rds)

    out_dir = "../zstores/ufs-hr1/1.00-degree/normalization"

    localtime = Timer()

    for basename, func in zip(
            ["mean_by_level.zarr", "stddev_by_level.zarr", "diffs_stddev_by_level.zarr"],
            [calc_mean_by_level, calc_stddev_by_level, calc_diffs_stddev_by_level]):

        fname = f"{out_dir}/{basename}"
        localtime.start(f"Writing {fname}")
        result = func(rds)
        result.to_zarr(fname, mode="w")
        localtime.stop(f"Done with {fname}")
        print()


if __name__ == "__main__":

    import subprocess

    jobscript = f"#!/bin/bash\n\n"+\
        f"#SBATCH -J calc_normalization\n"+\
        f"#SBATCH -o slurm/calc_normalization.%j.out\n"+\
        f"#SBATCH -e slurm/calc_normalization.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks=1\n"+\
        f"#SBATCH --cpus-per-task=30\n"+\
        f"#SBATCH --partition=compute\n"+\
        f"#SBATCH -t 120:00:00\n\n"+\
        f"source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"+\
        f"conda activate ufn\n"+\
        f"python -c 'from calc_normalization import main ; main()'"

    scriptname = "submit_normalization.sh"
    with open(scriptname, "w") as f:
        f.write(jobscript)

    subprocess.run(f"sbatch {scriptname}", shell=True)
