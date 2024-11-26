import pandas as pd
from alert.mig import MultiInstanceGPU 

def test_mig():
    n_jobs = 5
    job1 = {
        "gpus": 1,
        "nodes": {
            "della-l05g2": {
                "cpus": 1,
                "gpu_total_memory": {
                    "0": 85899345920
                },
                "gpu_used_memory": {
                    "0": 78096302080
                },
                "gpu_utilization": {
                    "0": 80.6
                },
                "total_memory": 17179869184,
                "total_time": 42029.4,
                "used_memory": 4032090112
            }
        },
        "total_time": 42151.20111846924
    }
    job2 = {
        "gpus": 1,
        "nodes": {
            "della-l05g2": {
                "cpus": 1,
                "gpu_total_memory": {
                    "0": 85899345920
                },
                "gpu_used_memory": {
                    "0": 7809630208
                },
                "gpu_utilization": {
                    "0": 8.6
                },
                "total_memory": 17179869184,
                "total_time": 42029.4,
                "used_memory": 4032090112
            }
        },
        "total_time": 42151.20111846924
    }
    job3 = {
        "gpus": 1,
        "nodes": {
            "della-l05g2": {
                "cpus": 1,
                "gpu_total_memory": {
                    "0": 85899345920
                },
                "gpu_used_memory": {
                    "0":5368709120
                },
                "gpu_utilization": {
                    "0": 6.0
                },
                "total_memory": 17179869184,
                "total_time": 42029.4,
                "used_memory": 4032090112
            }
        },
        "total_time": 42151.20111846924
    }
    df = pd.DataFrame({"jobid":["1234567"] * n_jobs,
                       "netid":["user1", "user1", "user2", "user1", "user2"],
                       "admincomment":[job1, job2, job3, job2, job1],
                       "cluster":["della"] * n_jobs,
                       "cores":[1] * n_jobs,
                       "gpus":[1] * n_jobs,
                       "state":["COMPLETED"] * n_jobs,
                       "partition":["gpu"] * n_jobs,
                       "elapsed-hours":[10] * n_jobs})
    mig = MultiInstanceGPU(df,
                           0,
                           "",
                           "",
                           "Subject",
                           cluster="della",
                           partition="gpu",
                           excluded_users=["aturing"])
    actual = mig.df[["NetID", "GPU-Util", "GPU-Mem-Used", "CPU-Mem-Used", "Hours"]]
    expected = pd.DataFrame({"NetID":["user1", "user2", "user1"],
                             "GPU-Util":["9%", "6%", "9%"],
                             "GPU-Mem-Used":["7 GB", "5 GB", "7 GB"],
                             "CPU-Mem-Used":["4 GB", "4 GB", "4 GB"],
                             "Hours":[10, 10, 10]})
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)
