
import datetime

from UFS2ARCO import FV3Dataset
from UFS2ARCO.replay import cached_replay_path_1degree as replay_path

from timer import Timer

if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start()

    replay = FV3Dataset(path_in=replay_path, config_filename="config-replay.yaml")
    kw = {"mode": "w"}
    day = 1
    for hour in [0, 6, 12, 18]:

        date = datetime.datetime(year=1994, month=1, day=day, hour=hour)
        localtime.start(f"Pulling {str(date)}")

        ds = replay.open_dataset(date, fsspec_kwargs={"s3":{"anon": True}}, engine="h5netcdf")
        replay.store_dataset(ds, **kw)

        kw = {"mode": "a", "append_dim":"time"}
        localtime.stop()

    walltime.stop("Total Walltime")
