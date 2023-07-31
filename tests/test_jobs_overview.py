import pytest
import pandas as pd
from alert.jobs_overview import JobsOverview

def test_jobs_overview():
    n_jobs = 6
    num_cores = 16
    num_gpus = 4
    wall_secs = 36000
    secs_per_hour = 60 * 60
    df = pd.DataFrame({"jobid":["1234567"] * n_jobs,
                       "netid":["user1", "user2", "user1", "user1", "user2", "user1"],
                       "cluster":["della"] * n_jobs,
                       "cores":[num_cores] * n_jobs,
                       "state":["COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED", "CANCELLED"],
                       "partition":["cpu", "cpu", "cpu", "cpu", "serial", "gpu"],
                       "cpu-seconds":[wall_secs * num_cores] * n_jobs,
                       "gpu-seconds":[0, 0, 0, 0, 0, num_gpus * wall_secs],
                       "gpu-job":[0, 0, 0, 0, 0, 1],
                       "elapsedraw":[wall_secs, wall_secs, 0, wall_secs, wall_secs, wall_secs]})
    jobs = JobsOverview(df, 7, "", "", "")
    actual = jobs.gp[["netid", "jobs", "cpu", "gpu", "COM", "CLD", "cpu-hours", "gpu-hours", "partitions"]]
    expected = pd.DataFrame({"netid":["user1", "user2"],
                             "jobs":[3, 2],
                             "cpu":[2, 2],
                             "gpu":[1, 0],
                             "COM":[2, 2],
                             "CLD":[1, 0],
                             "cpu-hours":[3 * num_cores * wall_secs / secs_per_hour,
                                          2 * num_cores * wall_secs / secs_per_hour],
                             "gpu-hours":[num_gpus * wall_secs / 3600, 0],
                             "partitions":["cpu,gpu", "cpu,serial"]})
    expected["cpu-hours"] = expected["cpu-hours"].apply(round)
    expected["gpu-hours"] = expected["gpu-hours"].apply(round)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)
