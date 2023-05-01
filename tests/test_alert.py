import pytest
import pandas as pd
import sys
sys.path.append("../")
sys.path.append(".")
print(sys.path)
from base import Alert
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

def test_datascience_mem_hours():
    alloc = 1000 * 1024**3
    used = 500 * 1024**3
    unused = alloc - used
    job1 = {
        "gpus": 0,
        "nodes": {
            "tiger-h26c1n24": {
                "cpus": 16,
                "total_memory": alloc,
                "total_time": 68841.1,
                "used_memory": used}},
        "total_time": 6230}
    df = pd.DataFrame({"jobid":["123", "123"],
                       "netid":["jdh4", "jdh4"],
                       "account":["cses"] * 2,
                       "cluster":["della", "della"],
                       "state":["COMPLETED", "COMPLETED"],
                       "partition":["datasci", "datasci"],
                       "elapsed-hours":[4, 4],
                       "cpu-hours":[16, 16],
                       "admincomment":[job1, job1]})
    ds = DataScienceMemoryHours(df, 0, "", "", "Subject")
    assert ds.df["cpu-hours"].sum() == 32
    assert ds.gp[ds.gp.netid == "jdh4"]["mem-hrs-unused"].values[0] == float(500 * 4 + 500 * 4) / 1000
