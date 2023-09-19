import pandas as pd
from alert.utilization_overview import UtilizationOverview

def test_jobs_overview():
    df = pd.DataFrame({"netid":["user1", "user2", "user3", "user1", "user5", "user4"],
                       "cluster":["della", "c2", "della", "della", "c3", "c3"],
                       "partition":["gpu", "q1", "mig", "gpu", "r1", "r1"],
                       "cpu-hours":[10, 20, 10, 50, 30, 10], 
                       "gpu-hours":[10, 20, 10, 50, 30, 10],
                       "elapsedraw":[1, 1, 1, 1, 1, 1]})
    util = UtilizationOverview(df, 7, "", "", "")
    actual = util.by_cluster
    expected = pd.DataFrame({"cluster":["della", "c2", "c3"],
                             "users":[2, 1, 2],
                             "cpu-hours":["70 (54%)", "20 (15%)", "40 (31%)"],
                             "gpu-hours":["70 (54%)", "20 (15%)", "40 (31%)"]})
    expected = pd.DataFrame({"cluster":["c2", "c3", "della"],
                             "users":[1, 2, 2],
                             "cpu-hours":["20 (15%)", "40 (31%)", "70 (54%)"],
                             "gpu-hours":["20 (15%)", "40 (31%)", "70 (54%)"]})
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)
