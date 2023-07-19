import pytest
import pandas as pd
from alert.excessive_time_limits import ExcessiveTimeLimits

@pytest.fixture(autouse=True)
def setUp():
    print()
    print("setUp")
    print("tearDown")

def test_excessive_time_limits():
    n_jobs = 6
    cpus = 10
    wallclock_hrs = 10e3
    df = pd.DataFrame({"netid":["user3", "user1", "user1", "user2", "user1", "user2"],
                       "cluster":["della"] * n_jobs,
                       "state":["COMPLETED"] * n_jobs,
                       "partition":["cpu"] * n_jobs,
                       "elapsed-hours":[wallclock_hrs] * n_jobs,
                       "cpu-alloc-hours":[wallclock_hrs * cpus] * n_jobs,
                       "cpu-hours":[95e3, 5e3, 10e3, 19e3, 15e3, 19e3]})
    df["cpu-waste-hours"] = df["cpu-alloc-hours"] - df["cpu-hours"]
    limits = ExcessiveTimeLimits(df, 0, "", "", "Subject")
    actual = limits.gp[["NetID", "CPU-Hours-Unused", "median(%)", "rank", "jobs"]]
    expected = pd.DataFrame({"NetID":["user1", "user2"],
                             "CPU-Hours-Unused":[95e3+90e3+85e3, 81e3+81e3],
                             "median(%)":[10, 19],
                             "rank":[3, 2],
                             "jobs":[3, 2]})
    expected.index += 1
    pd.testing.assert_frame_equal(actual, expected)
