import pandas as pd
from alert.utilization_overview import UtilizationOverview

def test_jobs_overview():
    df = pd.DataFrame({"netid":["user1", "user2", "user3", "user1", "user5", "user4"],
                       "cluster":["c1", "c2", "c1", "c1", "c3", "c3"],
                       "partition":["p1", "q1", "p2", "p1", "r1", "r1"],
                       "cpu-hours":[10, 20, 10, 50, 30, 10], 
                       "gpu-hours":[10, 20, 10, 50, 30, 10],
                       "elapsedraw":[1, 1, 1, 1, 1, 1]})
    util = UtilizationOverview(df, 7, "", "", "")
    actual = util.by_cluster
    expected = pd.DataFrame({"cluster":["c1", "c2", "c3"],
                             "users":[2, 1, 2],
                             "cpu-hours":["70 (54%)", "20 (15%)", "40 (31%)"],
                             "gpu-hours":["70 (54%)", "20 (15%)", "40 (31%)"]})
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)
