import argparse
import os
import time
import math
import subprocess
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd

# jobstats
import json
import gzip
import base64

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# external
import dossier


# conversion factors
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24

# slurm job states
states = {
  'BF'  :'BOOT_FAIL',
  'CLD' :'CANCELLED',
  'COM' :'COMPLETED',
  'DL'  :'DEADLINE',
  'F'   :'FAILED',
  'NF'  :'NODE_FAIL',
  'OOM' :'OUT_OF_MEMORY',
  'PD'  :'PENDING',
  'PR'  :'PREEMPTED',
  'R'   :'RUNNING',
  'RQ'  :'REQUEUED',
  'RS'  :'RESIZING',
  'RV'  :'REVOKED',
  'S'   :'SUSPENDED',
  'TO'  :'TIMEOUT'
  }
JOBSTATES = dict(zip(states.values(), states.keys()))

def send_email(s, addressee, sender="halverson@princeton.edu"):
  msg = MIMEMultipart('alternative')
  msg['Subject'] = "Slurm job alerts"
  msg['From'] = sender
  msg['To'] = addressee
  text = "None"
  html = f'<html><head></head><body><font face="Courier New, Courier, monospace"><pre>{s}</pre></font></body></html>'
  part1 = MIMEText(text, 'plain'); msg.attach(part1) 
  part2 = MIMEText(html, 'html');  msg.attach(part2)
  s = smtplib.SMTP('localhost')
  s.sendmail(sender, addressee, msg.as_string())
  s.quit()
  return None

def raw_dataframe_from_sacct(flags, start_date, fields, renamings=[], numeric_fields=[], use_cache=False):
  fname = f"cache_sacct_{start_date.strftime('%Y%m%d')}.csv"
  if use_cache and os.path.exists(fname):
    print("\nUsing cache file.\n", flush=True)
    rw = pd.read_csv(fname, low_memory=False)
  else:
    cmd = f"sacct {flags} -S {start_date.strftime('%Y-%m-%d')}T00:00:00 -E now -o {fields}"
    if use_cache: print("\nCalling sacct (which may require several seconds) ... ", end="", flush=True)
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=120, text=True, check=True)
    if use_cache: print("done.", flush=True)
    lines = output.stdout.split('\n')
    if lines != [] and lines[-1] == "": lines = lines[:-1]
    cols = fields.split(",")
    rw = pd.DataFrame([line.split("|")[:len(cols)] for line in lines])
    rw.columns = cols
    rw.rename(columns=renamings, inplace=True)
    rw[numeric_fields] = rw[numeric_fields].apply(pd.to_numeric)
    if use_cache: rw.to_csv(fname, index=False)
  return rw

def gpus_per_job(tres):
  # billing=8,cpu=4,mem=16G,node=1
  # billing=112,cpu=112,gres/gpu=16,mem=33600M,node=4
  if "gres/gpu=" in tres:
    for part in tres.split(","):
      if "gres/gpu=" in part:
        gpus = int(part.split("=")[-1])
        assert gpus > 0
    return gpus
  else:
    return 0

def is_gpu_job(tres):
  return 1 if "gres/gpu=" in tres and not "gres/gpu=0" in tres else 0

def get_stats_dict(x):
  if not x or pd.isna(x) or x == "JS1:Short" or x == "JS1:None":
    return {}
  else:
    return json.loads(gzip.decompress(base64.b64decode(x[4:])))

def add_new_and_derived_fields(df):
  # new and derived fields
  df["gpus"] = df.alloctres.apply(gpus_per_job)
  df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')
  df["gpu-job"] = df.alloctres.apply(is_gpu_job)
  df["cpu-only-seconds"] = df.apply(lambda row: 0 if row["gpus"] else row["cpu-seconds"], axis="columns")
  df["elapsed-hours"] = df.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR))
  df["start-date"] = df.start.apply(lambda x: x if x == "Unknown" else datetime.fromtimestamp(int(x)).strftime("%a %-m/%-d"))
  df["cpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * SECONDS_PER_MINUTE - row["elapsedraw"]) * row["cores"] / SECONDS_PER_HOUR), axis="columns")
  df["gpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * SECONDS_PER_MINUTE - row["elapsedraw"]) * row["gpus"]  / SECONDS_PER_HOUR), axis="columns")
  df["cpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * SECONDS_PER_MINUTE * row["cores"] / SECONDS_PER_HOUR), axis="columns")
  df["gpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * SECONDS_PER_MINUTE * row["gpus"]  / SECONDS_PER_HOUR), axis="columns")
  df["cpu-hours"] = df["cpu-seconds"] / SECONDS_PER_HOUR
  df["gpu-hours"] = df["gpu-seconds"] / SECONDS_PER_HOUR
  df["admincomment"] = df["admincomment"].apply(get_stats_dict)
  return df

def cpu_efficiency(d, elapsedraw, single=False):
  total = 0
  total_used = 0
  for node in d['nodes']:
    try:
      used  = d['nodes'][node]['total_time']
      cores = d['nodes'][node]['cpus']
    except:
      print("total_time not found")
      return (0, 1)  # dummy values that avoid division by zero later
    else:
      alloc = elapsedraw * cores  # equal to cputimeraw
      total += alloc
      total_used += used
  return round(100 * total_used / total, 1) if single else (total_used, total)

def gpu_efficiency(d, elapsedraw, single=False):
  total = 0
  total_used = 0
  for node in d['nodes']:
    try:
      gpus = list(d['nodes'][node]['gpu_utilization'].keys())
    except:
      print("gpu_utilization not found")
      return (0, 1)  # dummy values that avoid division by zero later
    else:
      for gpu in gpus:
        util = d['nodes'][node]['gpu_utilization'][gpu]
        total      += elapsedraw
        total_used += elapsedraw * (float(util) / 100)
  return round(100 * total_used / total, 1) if single else (total_used, total)

def xpu_efficiencies_of_heaviest_users(df, cluster, partitions, xpu):
  # ignore ondemand and salloc?
  # compute proportion using all much data as possible
  pr = df[(df.cluster == cluster) & (df.partition.isin(partitions)) & pd.notna(df[f"{xpu}-seconds"])].copy()
  pr = pr.groupby("netid").agg({f"{xpu}-seconds":np.sum}).reset_index(drop=False)
  pr["proportion(%)"] = pr[f"{xpu}-seconds"].apply(lambda x: round(100 * x / pr[f"{xpu}-seconds"].sum()))
  pr = pr.rename(columns={f"{xpu}-seconds":f"{xpu}-seconds-all"}).sort_values(by=f"{xpu}-seconds-all", ascending=False)
  # 2nd dataframe based on admincomment
  ce = df[(df.cluster == cluster) & \
          (df["elapsedraw"] >= 0.5 * SECONDS_PER_HOUR) & \
          (df.partition.isin(partitions))].copy()
  ce = ce[ce.admincomment != {}]  # need to get running jobs
  ce = ce.merge(pr, how="left", on="netid")
  if xpu == "cpu":
    ce[f"{xpu}-tuples"] = ce.apply(lambda row: cpu_efficiency(row["admincomment"], row["elapsedraw"]), axis="columns")
  else:
    ce[f"{xpu}-tuples"] = ce.apply(lambda row: gpu_efficiency(row["admincomment"], row["elapsedraw"]), axis="columns")
  ce[f"{xpu}-seconds-used"]  = ce[f"{xpu}-tuples"].apply(lambda x: x[0])
  ce[f"{xpu}-seconds-total"] = ce[f"{xpu}-tuples"].apply(lambda x: x[1])
  ce["interactive"] = ce["jobname"].apply(lambda x: 1 if x.startswith("sys/dashboard") or x.startswith("interactive") else 0)
  d = {"netid":np.size, f"{xpu}-seconds-used":np.sum, f"{xpu}-seconds-total":np.sum, \
       "proportion(%)":"first", f"{xpu}-seconds-all":"first", "cores":np.mean, "interactive":np.sum}
  ce = ce.groupby("netid").agg(d).rename(columns={"netid":"jobs"})
  ce = ce.sort_values(by=f"{xpu}-seconds-total", ascending=False).reset_index(drop=False)
  ce = ce.head(15)
  ce["eff(%)"] = 100.0 * ce[f"{xpu}-seconds-used"] / ce[f"{xpu}-seconds-total"]
  ce[f"{xpu}-hours"] = ce.apply(lambda row: round(row[f"{xpu}-seconds-total"] / SECONDS_PER_HOUR), axis="columns")
  ce["coverage"] = ce.apply(lambda row: round(row[f"{xpu}-seconds-total"] / row[f"{xpu}-seconds-all"], 2), axis="columns")
  ce["eff(%)"] = ce["eff(%)"].apply(lambda x: round(x))
  ce["cores"] = ce["cores"].apply(lambda x: round(x, 1))
  ce = ce[["netid", f"{xpu}-hours", "eff(%)", "proportion(%)", "jobs", "interactive", "cores", "coverage"]][ce["eff(%)"] < 70]
  ce.index += 1
  return ce

def gpu_jobs_zero_util(df, cluster, partitions):
  zu = df[(df.cluster == cluster) & \
          (df["elapsedraw"] > 1 * SECONDS_PER_HOUR) & \
          (df.partition.isin(partitions))].copy()
  zu = zu[zu.admincomment != {}]  # need to get running jobs
  def gpus_with_zero_util(d):
    ct = 0
    for node in d['nodes']:
      try:
        gpus = list(d['nodes'][node]['gpu_utilization'].keys())
      except:
        print("gpu_utilization not found")
        return 0
      else:
        for gpu in gpus:
          util = d['nodes'][node]['gpu_utilization'][gpu]
          if int(util) == 0: ct += 1
    return ct
  zu["gpus-unused"] = zu.apply(lambda row: gpus_with_zero_util(row["admincomment"]), axis="columns")
  zu = zu[zu["gpus-unused"] > 0].rename(columns={"elapsed-hours":"hours"}).sort_values(by="netid")
  zu.state = zu.state.apply(lambda x: JOBSTATES[x])
  return zu[["netid", "gpus", "gpus-unused", "jobid", "cluster", "state", "hours", "start-date"]]

def unused_allocated_hours_of_completed(df, cluster, partitions, xpu):
  wh = df[(df.cluster == cluster) & \
          (df.state == "COMPLETED") & \
          (df["elapsed-hours"] >= 2) & \
          (df.partition.isin(partitions))].copy()
  wh["ratio"] = 100 * wh[f"{xpu}-hours"] / wh[f"{xpu}-alloc-hours"]
  d = {f"{xpu}-waste-hours":np.sum, f"{xpu}-alloc-hours":np.sum, f"{xpu}-hours":np.sum, "netid":np.size, \
       "partition":lambda series: ",".join(sorted(set(series))), "ratio":"median"}
  wh = wh.groupby("netid").agg(d).rename(columns={"netid":"jobs", "ratio":"median(%)"})
  wh = wh.sort_values(by=f"{xpu}-hours", ascending=False).reset_index(drop=False)
  # warning pandas.DataFrame.rank is a built-in so use df["rank"]
  wh["rank"] = wh.index + 1
  wh = wh.sort_values(by=f"{xpu}-waste-hours", ascending=False).reset_index(drop=False)
  wh = wh[:5]
  wh.index += 1
  wh[f"{xpu}-hours"] = wh[f"{xpu}-hours"].apply(round).astype("int64")
  wh["mean(%)"] = 100 * wh[f"{xpu}-hours"] / wh[f"{xpu}-alloc-hours"]
  wh["mean(%)"] = wh["mean(%)"].apply(round).astype("int64")
  wh["median(%)"] = wh["median(%)"].apply(round).astype("int64")
  # apply filter
  wh = wh[(wh["mean(%)"] < 20) & (wh["median(%)"] < 20) & (wh["rank"] < 10)]
  wh = wh[["netid", f"{xpu}-waste-hours", f"{xpu}-hours", f"{xpu}-alloc-hours", "mean(%)", "median(%)", "rank", "jobs", "partition"]]
  return wh.rename(columns={f"{xpu}-waste-hours":"unused", f"{xpu}-hours":"used", f"{xpu}-alloc-hours":"total"})

def multinode_cpu_fragmentation(df):
  cols = ["jobid", "netid", "cluster", "nodes", "cores", "state", "partition", "elapsed-hours", "start-date", "start"]
  cond = (df["elapsed-hours"] >= 2) & (df.nodes > 1) & (df.cores / df.nodes < 14) & (df.gpus == 0)
  m = df[cond][cols].copy()
  m = m.sort_values(["netid", "start"], ascending=[True, False]).drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  m.state = m.state.apply(lambda x: JOBSTATES[x])
  return m

def multinode_gpu_fragmentation(df):
  cols = ["jobid", "netid", "cluster", "nodes", "gpus", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment", "elapsedraw"]
  cond1 = (df["elapsed-hours"] >= 2) & (df.nodes > 1) & (df.gpus > 0) & (df.nodes == df.gpus)
  cond2 = (df["elapsed-hours"] >= 2) & (df.nodes > 1) & (df.gpus > 0) & (df.cluster.isin(["tiger", "traverse"])) & (df.gpus < 4 * df.nodes)
  m = df[cond1 | cond2][cols].copy()
  m.state = m.state.apply(lambda x: JOBSTATES[x])
  m = m.sort_values(["netid", "start"], ascending=[True, False]).drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  m["eff(%)"] = m.apply(lambda row: gpu_efficiency(row["admincomment"], row["elapsedraw"], single=True) if row["admincomment"] != {} else "", axis="columns")
  cols = cols[:7] + ["hours", "eff(%)"]
  return m[cols]

def recent_jobs_all_failures(df):
  """All jobs failed on the last day that the user ran."""
  cols = ["netid", "cluster", "state", "start", "end"]
  f = df[cols][df.start != "Unknown"].copy()
  # next line deals with RUNNING jobs
  f["end"] = f["end"].str.replace("Unknown", str(round(time.time())), regex=False)
  f["end"] = f["end"].astype("int64")
  def day_since(x):
    from datetime import datetime
    dt = datetime.fromtimestamp(x)
    return (datetime(dt.year, dt.month, dt.day) - datetime(1970, 1, 1)).days
  f["day-since-epoch"] = f["end"].apply(day_since)
  d = {"netid":np.size, "state":lambda series: sum([s == "FAILED" for s in series])}
  f = f.groupby(["netid", "cluster", "day-since-epoch"]).agg(d).rename(columns={"netid":"jobs", "state":"num-failed"}).reset_index()
  f = f.groupby(["netid", "cluster"]).apply(lambda d: d.iloc[d["day-since-epoch"].argmax()])
  f = f[(f["num-failed"] == f["jobs"]) & (f["num-failed"] > 3)]
  f["dossier"]  = f.netid.apply(lambda x: dossier.ldap_plus([x])[1])
  f["position"] = f.dossier.apply(lambda x: x[3])
  f["dept"]     = f.dossier.apply(lambda x: x[1])
  return f[["netid", "position", "dept", "cluster", "jobs", "num-failed"]]

def jobs_with_the_most_cores(df):
  """Top 10 users with the highest number of CPU-cores in a job. Only one job per user is shown."""
  cols = ["jobid", "netid", "cluster", "cores", "nodes", "gpus", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment", "elapsedraw"]
  c = df[cols].groupby("netid").apply(lambda d: d.iloc[d["cores"].argmax()]).copy()
  c = c.sort_values("cores", ascending=False)[:10].drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  c.state = c.state.apply(lambda x: JOBSTATES[x])
  c["eff(%)"] = c.apply(lambda row: cpu_efficiency(row["admincomment"], row["elapsedraw"], single=True) if row["admincomment"] != {} else "", axis="columns")
  cols = cols[:8] + ["hours", "eff(%)"]
  return c[cols]

def jobs_with_the_most_gpus(df):
  """Top 10 users with the highest number of GPUs in a job. Only one job per user is shown."""
  cols = ["jobid", "netid", "cluster", "gpus", "nodes", "cores", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment", "elapsedraw"]
  df = df[df.partition != "cryoem"]
  g = df[cols].groupby("netid").apply(lambda d: d.iloc[d["gpus"].argmax()]).copy()
  g = g.sort_values("gpus", ascending=False)[:10].drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  g.state = g.state.apply(lambda x: JOBSTATES[x])
  g["eff(%)"] = g.apply(lambda row: gpu_efficiency(row["admincomment"], row["elapsedraw"], single=True) if row["admincomment"] != {} else "", axis="columns")
  cols = cols[:8] + ["hours", "eff(%)"]
  return g[cols]

def longest_queue_times(raw):
  q = raw[raw.state == "PENDING"].copy()
  q = q[~q.jobid.str.contains("_")]
  q["s-days"] = round((time.time() - q["submit"])   / SECONDS_PER_HOUR / HOURS_PER_DAY)
  q["e-days"] = round((time.time() - q["eligible"]) / SECONDS_PER_HOUR / HOURS_PER_DAY)
  q["s-days"] = q["s-days"].astype("int64")
  q["e-days"] = q["e-days"].astype("int64")
  cols = ["jobid", "netid", "cluster", "nodes", "cores", "qos", "partition", "s-days", "e-days"]
  q = q[cols].groupby("netid").apply(lambda d: d.iloc[d["s-days"].argmax()]).sort_values("s-days", ascending=False)[:10]
  return q

def add_dividers(df, title="", pre="\n\n\n"):
  rows = df.split("\n")
  width = max([len(row) for row in rows])
  padding = " " * max(1, math.ceil((width - len(title)) / 2))
  divider = padding + title + padding
  if bool(title): 
    rows.insert(0, divider)
    rows.insert(1, "-" * len(divider))
    rows.insert(3, "-" * len(divider))
  else:
    rows.insert(0, "-" * len(divider))
    rows.insert(2, "-" * len(divider))
  return pre + "\n".join(rows)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Slurm job alerts')
  parser.add_argument('-d', '--days', type=int, default=14, metavar='N',
                      help='Create report over N previous days from now (default: 14)')
  parser.add_argument('--email', action='store_true', default=False,
                      help='Send output via email')
  args = parser.parse_args()

  # pandas display settings
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", 1000)

  # convert slurm timestamps to seconds
  os.environ["SLURM_TIME_FORMAT"] = "%s"

  flags = "-L -a -X -P -n"
  start_date = datetime.now() - timedelta(days=args.days)
  fields = "jobid,user,cluster,account,partition,cputimeraw,elapsedraw,timelimitraw,nnodes,ncpus,alloctres,submit,eligible,start,end,qos,state,admincomment,jobname"
  renamings = {"user":"netid", "cputimeraw":"cpu-seconds", "nnodes":"nodes", "ncpus":"cores", "timelimitraw":"limit-minutes"}
  numeric_fields = ["cpu-seconds", "elapsedraw", "limit-minutes", "nodes", "cores", "submit", "eligible"]
  raw = raw_dataframe_from_sacct(flags, start_date, fields, renamings, numeric_fields, use_cache=not args.email)

  raw = raw[~raw.cluster.isin(["tukey", "perseus"])]
  raw.cluster   =   raw.cluster.str.replace("tiger2", "tiger")
  raw.partition = raw.partition.str.replace("datascience", "datasci")
  raw.partition = raw.partition.str.replace("physics", "phys")
  raw.state = raw.state.apply(lambda x: "CANCELLED" if "CANCEL" in x else x)

  # df excludes pending jobs
  df = raw.copy()
  df = df[pd.notnull(df.alloctres) & (df.alloctres != "")]
  df.start = df.start.astype("int64")
 
  #print(df.admincomment.head(25))
  #for i, x in enumerate(df.admincomment):
  #  print(i, x)
  #  print(get_stats_dict(x))
  #df = df.head(250)
  df = add_new_and_derived_fields(df)

  if not args.email:
    df.info()
    print(df.describe())
    print("\nTotal NaNs:", df.isnull().sum().sum(), "\n")

  # header
  fmt = "%a %b %-d"
  s = f"{start_date.strftime(fmt)} - {datetime.now().strftime(fmt)}\n\n"
  s += f"Total users: {raw.netid.unique().size}\n"
  s += f"Total jobs:  {raw.shape[0]}\n\n"

  ####################################
  ### used allocated cpu/gpu hours ###
  ####################################
  cls = (("della", "Della (CPU)", ("cpu", "datascience", "physics"), "cpu"), \
         ("della", "Della (GPU)", ("gpu",), "gpu"), \
         ("stellar", "Stellar (AMD)", ("bigmem", "cimes"), "cpu"), \
         ("stellar", "Stellar (Intel)", ("all", "pppl", "pu", "serial"), "cpu"), \
         ("tiger", "TigerCPU", ("cpu", "ext", "serial"), "cpu"), \
         ("tiger", "TigerGPU", ("gpu",), "gpu"), \
         ("traverse", "Traverse (GPU)", ("all",), "gpu"))
  s += "           Unused allocated CPU/GPU-Hours (of COMPLETED 2+ hour jobs)"
  for cluster, name, partitions, xpu in cls:
    un = unused_allocated_hours_of_completed(df, cluster, partitions, xpu)
    if not un.empty:
      df_str = un.to_string(index=True, justify="center")
      s += add_dividers(df_str, title=name, pre="\n\n")

  ####### consider jobs in the last N days only #######
  #thres_days = 4
  #df = df[df.start >= time.time() - thres_days * HOURS_PER_DAY * SECONDS_PER_HOUR]
  #s += f"\n\n\n            --- everything below is for the last {thres_days} days ---"

  ######################
  ### cpu efficiency ###
  ######################
  s += "\n\n\n           CPU/GPU Efficiencies of top 15 users (30+ minute jobs, ignoring running)"
  for cluster, name, partitions, xpu in cls:
    un = xpu_efficiencies_of_heaviest_users(df, cluster, partitions, xpu)
    if not un.empty:
      df_str = un.to_string(index=True, justify="center")
      s += add_dividers(df_str, title=name, pre="\n\n")

  ####### consider jobs in the last N days only #######
  thres_days = 3
  df = df[df.start >= time.time() - thres_days * HOURS_PER_DAY * SECONDS_PER_HOUR]
  s += f"\n\n\n            --- everything below is for the last {thres_days} days ---"

  ##############################
  ### 0 utilization on a GPU ###
  ##############################
  s += "\n\n\nZero utilization on a GPU (>1 hour jobs)"
  for cluster, name, partitions in [("tiger", "TigerGPU", ("gpu",)), \
                                    ("della", "Della (GPU)", ("gpu",)), \
                                    ("traverse", "Traverse (GPU)", ("all",))]:
    zu = gpu_jobs_zero_util(df, cluster, partitions)
    if not zu.empty:
      df_str = zu.to_string(index=False, justify="center")
      s += add_dividers(df_str, title=name, pre="\n\n")

  #####################
  ### fragmentation ###
  #####################
  fg = multinode_cpu_fragmentation(df)
  if not fg.empty:
    df_str = fg.to_string(index=False, justify="center")
    s += add_dividers(df_str, title="Multinode CPU jobs with < 14 cores per node (all jobs, 2+ hours)", pre="\n\n\n")
    
  fg = multinode_gpu_fragmentation(df)
  if not fg.empty:
    df_str = fg.to_string(index=False, justify="center")
    s += add_dividers(df_str, title="Multinode GPU jobs with fragmentation (all jobs, 2+ hours)")

  ##############################
  ### last jobs are failures ###
  ##############################
  fl = recent_jobs_all_failures(df)
  if not fl.empty:
    df_str = fl.to_string(index=False, justify="center")
    s += add_dividers(df_str, title="All jobs failed on last day (4+ jobs)")

  ##############################
  ### large cpu and gpu jobs ###
  ##############################
  df_str = jobs_with_the_most_cores(df).to_string(index=False, justify="center")
  s += add_dividers(df_str, title="Jobs with the most CPU-cores (1 job per user)")
  df_str = jobs_with_the_most_gpus(df).to_string(index=False, justify="center")
  s += add_dividers(df_str, title="Jobs with the most GPUs (1 job per user, ignoring cryoem)")

  ###########################
  ### longest queue times ###
  ###########################
  df_str = longest_queue_times(raw).to_string(index=False, justify="center")
  s += add_dividers(df_str, title="Longest queue times of PENDING jobs (1 job per user, ignoring job arrays)")

  send_email(s, "halverson@princeton.edu") if args.email else print(s)
