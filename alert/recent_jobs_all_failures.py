import time
from datetiime import datetime
import numpy as np

def recent_jobs_all_failures(df):
    """All jobs failed on the last day that the user ran."""
    cols = ["user", "cluster", "state", "start", "end"]
    f = df[cols][df.start != "Unknown"].copy()
    # next line deals with RUNNING jobs
    f["end"] = f["end"].str.replace("Unknown", str(round(time.time())), regex=False)
    f["end"] = f["end"].astype("int64")
    def day_since(x):
        dt = datetime.fromtimestamp(x)
        return (datetime(dt.year, dt.month, dt.day) - datetime(1970, 1, 1)).days
    f["day-since-epoch"] = f["end"].apply(day_since)
    d = {"user":np.size, "state":lambda series: sum([s == "FAILED" for s in series])}
    cols = ["user", "cluster", "day-since-epoch"]
    renamings = {"user":"jobs", "state":"num-failed"}
    f = f.groupby(cols).agg(d).rename(columns=renamings).reset_index()
    f = f.groupby(["user", "cluster"]).apply(lambda d: d.iloc[d["day-since-epoch"].argmax()])
    f = f[(f["num-failed"] == f["jobs"]) & (f["num-failed"] > 3)]
    cols = ["user", "position", "dept", "cluster", "jobs", "num-failed"]
    return f[cols]
