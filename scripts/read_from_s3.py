"""A basic script to pull datasets from s3"""

from os.path import join
from datetime import datetime
from shutil import rmtree
from collections.abc import Iterable

from ufs2arco import FV3Dataset

from timer import Timer

def replay_path(dates, forecast_hours, file_prefixes):
    upper = "simplecache::s3://noaa-ufs-gefsv13replay-pds/1deg"
    dates = [dates] if not isinstance(dates, Iterable) else dates

    files = []
    for date in dates:
        this_dir = f"{date.year:04d}/{date.month:02d}/{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}"
        for fp in file_prefixes:
            for fhr in forecast_hours:
                this_file = join(this_dir, f"{fp}{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}_fhr{fhr:02d}_control")
                files.append(this_file)
    return [join(upper, this_file) for this_file in files]

if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Initializing job")

    replay = FV3Dataset(path_in=replay_path, config_filename="config-replay.yaml")

    # Grab two cycles as an example
    cycles = [
        datetime(1994, 1, 1, 0),
        datetime(1994, 1, 1, 6)
    ]

    localtime.start(f"Pulling {cycles}")
    ds = replay.open_dataset(
        cycles,
        fsspec_kwargs={"s3":{"anon": True}},
        engine="h5netcdf"
    )
    replay.store_dataset(ds, mode="w")
    localtime.stop()
