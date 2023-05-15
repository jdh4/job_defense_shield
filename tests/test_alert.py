import pytest
import pandas as pd
import sys
sys.path.append("../")
sys.path.append(".")
from base import Alert
from utils import seconds_to_slurm_time_format
from alert.datascience_mem_hours import DataScienceMemoryHours

@pytest.fixture(autouse=True)
def setUp():
    print()
    print("setUp")
    print("tearDown")

def test_answer():
    df = pd.DataFrame({"netid":["jdh4", "jdh4", "jdh4", "miz"],
                       "partition":["gpu", "mig", "gpu", "mig"],
                       "use":range(4)})
    A = Alert(df, 13, "", "", "Subject")
    assert 1 == 1

def test_seconds_to_slurm_time_format():
    assert seconds_to_slurm_time_format(100) == "00:01:40"
    assert seconds_to_slurm_time_format(1000) == "00:16:40"
    assert seconds_to_slurm_time_format(10000) == "02:46:40"
    assert seconds_to_slurm_time_format(100000) == "1-03:46:40"

def test_datascience_mem_hours():
    n_jobs = 5
    alloc = 1000 * 1024**3
    used = 500 * 1024**3
    cpus = 10
    run_time_hrs = 100
    GB_per_TB = 1000
    job1 = {
        "gpus": 0,
        "nodes": {
            "della-h12n1": {
                "cpus": cpus,
                "total_memory": alloc,
                "total_time": cpus * run_time_hrs * 60 * 60,
                "used_memory": used}},
        "total_time": run_time_hrs * 60 * 60}
    job2 = {
        "gpus": 0,
        "nodes": {
            "della-h12n2": {
                "cpus": cpus,
                "total_memory": 2 * alloc,
                "total_time": cpus * run_time_hrs * 60 * 60,
                "used_memory": used}},
        "total_time": run_time_hrs * 60 * 60}
    job3 = {
        "gpus": 0,
        "nodes": {
            "della-h12n3": {
                "cpus": cpus,
                "total_memory": 10 * alloc,
                "total_time": cpus * run_time_hrs * 60 * 60,
                "used_memory": used}},
        "total_time": run_time_hrs * 60 * 60}
    df = pd.DataFrame({"jobid":["1234567"] * n_jobs,
                       "netid":["user1", "user1", "user2", "user1", "user2"],
                       "admincomment":[job1, job2, job3, job2, job1],
                       "account":["cses"] * n_jobs,
                       "cluster":["della"] * n_jobs,
                       "state":["COMPLETED"] * n_jobs,
                       "partition":["datasci"] * n_jobs,
                       "elapsed-hours":[run_time_hrs] * n_jobs,
                       "cpu-hours":[cpus * run_time_hrs] * n_jobs})
    ds = DataScienceMemoryHours(df, 0, "", "", "Subject")
    assert ds.df["cpu-hours"].sum() == cpus * run_time_hrs * n_jobs
    expected = pd.DataFrame({"netid":["user2", "user1"], "mem-hrs-unused":[1000, 350]})
    expected.index += 1
    actual = ds.gp[["netid", "mem-hrs-unused"]]
    pd.testing.assert_frame_equal(actual, expected)
