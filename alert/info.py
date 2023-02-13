import time
from utils import JOBSTATES
from utils import SECONDS_PER_MINUTE
from utils import SECONDS_PER_HOUR
from utils import MINUTES_PER_HOUR
from utils import HOURS_PER_DAY
from efficiency import cpu_efficiency
from efficiency import gpu_efficiency

def jobs_with_the_most_cores(df):
  """Top 10 users by the highest number of CPU-cores in a job. Only one job per user is shown."""
  cols = ["jobid", "netid", "cluster", "cores", "nodes", "gpus", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment", "elapsedraw"]
  c = df[cols].groupby("netid").apply(lambda d: d.iloc[d["cores"].argmax()]).copy()
  c = c.sort_values("cores", ascending=False)[:10].drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  c.state = c.state.apply(lambda x: JOBSTATES[x])
  if c.empty: return c  # prevents next line from failing
  c.nodes = c.nodes.astype("int64")
  c.cores = c.cores.astype("int64")
  c["eff(%)"] = c.apply(lambda row: cpu_efficiency(row["admincomment"], row["elapsedraw"], row["jobid"], row["cluster"], single=True) if row["admincomment"] != {} else "", axis="columns")
  cols = cols[:8] + ["hours", "eff(%)"]
  return c[cols]

def jobs_with_the_most_gpus(df):
  """Top 10 users by the highest number of GPUs in a job. Only one job per user is shown."""
  cols = ["jobid", "netid", "cluster", "gpus", "nodes", "cores", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment", "elapsedraw"]
  df = df[(df.partition != "cryoem") & (df.netid != "cryoem")]
  g = df[cols].groupby("netid").apply(lambda d: d.iloc[d["gpus"].argmax()]).copy()
  g = g.sort_values("gpus", ascending=False)[:10].drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  g.state = g.state.apply(lambda x: JOBSTATES[x])
  if g.empty: return g  # prevents next line from failing
  g.nodes = g.nodes.astype("int64")
  g.cores = g.cores.astype("int64")
  g["eff(%)"] = g.apply(lambda row: gpu_efficiency(row["admincomment"], row["elapsedraw"], row["jobid"], row["cluster"], single=True) if row["admincomment"] != {} else "", axis="columns")
  cols = cols[:8] + ["hours", "eff(%)"]
  return g[cols]

def longest_queue_times(raw):
  q = raw[raw.state == "PENDING"].copy()
  q = q[~q.jobid.str.contains("_")]
  q["s-days"] = round((time.time() - q["submit"])   / SECONDS_PER_HOUR / HOURS_PER_DAY)
  q["e-days"] = round((time.time() - q["eligible"]) / SECONDS_PER_HOUR / HOURS_PER_DAY)
  q["s-days"] = q["s-days"].astype("int64")
  q["e-days"] = q["e-days"].astype("int64")
  q.nodes = q.nodes.astype("int64")
  q.cores = q.cores.astype("int64")
  cols = ["jobid", "netid", "cluster", "nodes", "cores", "qos", "partition", "s-days", "e-days"]
  q = q[cols].groupby("netid").apply(lambda d: d.iloc[d["s-days"].argmax()]).sort_values("s-days", ascending=False)
  return q[q["s-days"] >= 4][:10]
