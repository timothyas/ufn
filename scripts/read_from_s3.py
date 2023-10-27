"""A basic script to pull datasets from s3"""

from datetime import datetime
from shutil import rmtree

from UFS2ARCO import FV3Dataset
from UFS2ARCO.replay import cached_replay_path_1degree as replay_path

from timer import Timer

if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Initializing job")

    replay = FV3Dataset(path_in=replay_path, config_filename="config-replay.yaml")
    kw = {"mode": "w"}
    year = 1994
    month = 2
    for day in range(28,30):
        for hour in [0, 6, 12, 18]:

            try:
                date = datetime(year=year, month=month, day=day, hour=hour)
                valid_time = True

            except ValueError:

                # this should only happen when we ask for an invalid date, like Feb 30
                valid_time = False
                print(f" ... skipping year={year}, month={month}, day={day}, hour={hour}")

            if valid_time:
                localtime.start(f"Pulling {str(date)}")
                ds = replay.open_dataset(date, fsspec_kwargs={"s3":{"anon": True}}, engine="h5netcdf")
                replay.store_dataset(ds, **kw)
                kw = {"mode": "a", "append_dim":"time"}
                localtime.stop()

    walltime.stop("Total Walltime")
