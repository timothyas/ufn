"""We can't use a dask cluster, because it cannot serialize the tasks of opening multiple
datasets with an io buffered reader object, orsomething.

So, this is easy enough to just submit separate slurm jobs that work on their own job ID.
"""

import os
import subprocess

from ufs2arco import FV3Dataset
from replay_mover import ReplayMoverQuarterDegree
from timer import Timer


def submit_slurm_mover(job_id, mover):

    the_code = \
        f"from replay_mover import ReplayMoverQuarterDegree\n"+\
        f"mover = ReplayMoverQuarterDegree(\n"+\
        f"    n_jobs={mover.n_jobs},\n"+\
        f"    n_cycles={mover.n_cycles},\n"+\
        f"    config_filename='{mover.config_filename}',\n"+\
        f"    storage_options={mover.storage_options},\n"+\
        f"    main_cache_path='{mover.main_cache_path}',\n"+\
        f")\n"+\
        f"mover.run({job_id})"

    txt = "#!/bin/bash\n\n" +\
        f"#SBATCH -J qrm{job_id:03d}\n"+\
        f"#SBATCH -o slurm/all_quarter_mover/{job_id:03d}.%j.out\n"+\
        f"#SBATCH -e slurm/all_quarter_mover/{job_id:03d}.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks=1\n"+\
        f"#SBATCH --cpus-per-task=30\n"+\
        f"#SBATCH --partition=compute\n"+\
        f"#SBATCH -t 120:00:00\n\n"+\
        f"source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"+\
        f"conda activate ufs2arco\n"+\
        f'python -c "{the_code}"'

    script_dir = "job-scripts"
    fname = f"{script_dir}/submit_qmover{job_id:03d}.sh"
    if not os.path.isdir(script_dir):
        os.makedirs(script_dir)

    with open(fname, "w") as f:
        f.write(txt)

    subprocess.run(f"sbatch {fname}", shell=True)


if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Initializing job")

    # start with many jobs to just write a single slice of time steps for verification
    # estimating ~4 GB per cycle, need 2x per job
    # (4 * 2 GB / cycle) * (4 cycles / job) * (15 jobs) = 480 GB
    # (4 * 2 GB / cycle) * (4 cycles / job) * (60 jobs) = 1440 GB
    # (4 * 2 GB / cycle) * (60 cycles / job) * (60 jobs) = 29 Tb
    # (4 * 2 GB / cycle) * (8 cycles / job) * (360 jobs) = 24 Tb
    mover = ReplayMoverQuarterDegree(
        n_jobs=360,
        n_cycles=8,
        config_filename="config-replay-0.25-degree.yaml",
        storage_options={"token": "/contrib/Tim.Smith/.gcs/replay-service-account.json"},
        main_cache_path="/lustre/Tim.Smith/tmp-replay/0.25-degree",
    )

    localtime.start("Make and Store Container Dataset")
    mover.store_container()
    localtime.stop()

    localtime.start("Run slurm jobs")
    for job_id in range(mover.n_jobs):
        submit_slurm_mover(job_id, mover)
    localtime.stop()

    walltime.stop()
