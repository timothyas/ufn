from UFS2ARCO import FV3Dataset
from replay_mover import ReplayMover1Degree
from timer import Timer

if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Initializing job")

    # start with many jobs to just write a single slice of time steps for verification
    mover = ReplayMover1Degree(n_jobs=3000, config_filename="config-replay.yaml")

    localtime.start("Make and Store Container Dataset")
    mover.store_container()
    localtime.stop()

    localtime.start("Run job 1000 / 3000")
    job_id = 1000
    mover.run(job_id=job_id)
    localtime.stop()

    # Now do this the ol fashioned way (by ol' fashioned I mean a week ago)
    replay = FV3Dataset(path_in=mover.cached_path, config_filename="config-replay-local.yaml")

    kw = {"mode": "w"}
    for cycle in mover.my_cycles(job_id):

        date = mover.npdate2datetime(cycle)
        localtime.start(f"Reading {str(date)}")
        xds = replay.open_dataset(date, **mover.ods_kwargs)
        replay.store_dataset(xds, **kw)
        kw = {"mode": "a", "append_dim": "time"}
        localtime.stop()

    walltime.stop("Total Walltime")
