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
    mover.run(job_id=1000)
    localtime.stop()

    walltime.stop("Total Walltime")
