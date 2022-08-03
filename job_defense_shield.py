#!/home/jdh4/bin/jds-env/bin/python -uB

import argparse
import os
import sys
import time
import math
import glob
import subprocess
import textwrap
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# jobstats
import json
import gzip
import base64

from efficiency import get_stats_dict
from efficiency import cpu_efficiency
from efficiency import gpu_efficiency
from efficiency import cpu_memory_usage
from efficiency import num_gpus_with_zero_util

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# external
import dossier  # wget https://raw.githubusercontent.com/jdh4/tigergpu_visualization/master/dossier.py


# conversion factors
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
MINUTES_PER_HOUR = 60
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

def send_email(s, addressee, subject="Slurm job alerts", sender="halverson@princeton.edu"):
  msg = MIMEMultipart('alternative')
  msg['Subject'] = subject
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

def add_new_and_derived_fields(df):
  df["gpus"] = df.alloctres.apply(gpus_per_job)
  df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')
  df["gpu-job"] = df.alloctres.apply(is_gpu_job)
  df["cpu-only-seconds"] = df.apply(lambda row: 0 if row["gpus"] else row["cpu-seconds"], axis="columns")
  df["elapsed-hours"] = df.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR))
  df["start-date"] = df.start.apply(lambda x: x if x == "Unknown" else datetime.fromtimestamp(int(x)).strftime("%a %-m/%d"))
  df["cpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * SECONDS_PER_MINUTE - row["elapsedraw"]) * row["cores"] / SECONDS_PER_HOUR), axis="columns")
  df["gpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * SECONDS_PER_MINUTE - row["elapsedraw"]) * row["gpus"]  / SECONDS_PER_HOUR), axis="columns")
  df["cpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * SECONDS_PER_MINUTE * row["cores"] / SECONDS_PER_HOUR), axis="columns")
  df["gpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * SECONDS_PER_MINUTE * row["gpus"]  / SECONDS_PER_HOUR), axis="columns")
  df["cpu-hours"] = df["cpu-seconds"] / SECONDS_PER_HOUR
  df["gpu-hours"] = df["gpu-seconds"] / SECONDS_PER_HOUR
  df["admincomment"] = df["admincomment"].apply(get_stats_dict)
  return df

def get_first_name(netid):
  cmd = f"ldapsearch -x uid={netid} displayname"
  output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=5, text=True, check=True)
  lines = output.stdout.split('\n')
  for line in lines:
    if line.startswith("displayname:"):
      full_name = line.replace("displayname:", "").strip()
      if full_name.replace(".", "").replace(",", "").replace(" ", "").replace("-", "").isalpha():
        return f"Hi {full_name.split()[0]}"
  return "Hello"

def large_memory_needed(usage, account, safety=1.0):
  cascade_max_mem = safety * 190
  physics_max_mem = safety * 380
  if (usage >= cascade_max_mem and account != "physics") or (usage >= physics_max_mem and account == "physics"):
    return "Yes"
  else:
    return "No"

def datascience_node_violators(df):
  ds = df[(df.cluster == "della") & 
          (df.partition == "datasci") & 
          (df["elapsed-hours"] >= 1)].copy()
  #ds["admincomment"] = ds.apply(lambda row: get_stats_for_running_job(row["jobid"], row["cluster"]) if row["admincomment"] == {} else row["admincomment"], axis='columns')
  ds = ds[ds.admincomment != {}]
  ds = ds[ds.state != "OUT_OF_MEMORY"]
  ds["memory-tuple"] = ds.apply(lambda row: cpu_memory_usage(row["admincomment"], row["jobid"], row["cluster"]), axis="columns")
  ds["memory-used"]  = ds["memory-tuple"].apply(lambda x: x[0])
  ds["memory-alloc"] = ds["memory-tuple"].apply(lambda x: x[1])
  ds["Large-Memory-Needed?"] = ds.apply(lambda row: large_memory_needed(row["memory-used"], row["account"]), axis="columns")
  ds["within-safety"] = ds.apply(lambda row: True if large_memory_needed(row["memory-used"], row["account"], safety=0.8) == "Yes" else False, axis="columns")
  #ds["cpu-hours"] = ds["cpu-hours"].apply(round).astype("int64")
  #ds.state = ds.state.apply(lambda x: JOBSTATES[x])
  #ds = ds[ds.partition != "datasci"]
  #ds = ds[ds.partition != "gpu"]
  #print(ds[(ds["memory-alloc"] > 190) & (ds["memory-alloc"] < 380)][["netid", "jobid", "account", "memory-alloc"]])
  #print(ds[(ds["memory-alloc"] < 190) & (ds["cores"] > 40)][["netid", "jobid", "account", "memory-alloc"]])
  #print(ds[(ds["memory-alloc"] < 190)][["netid", "jobid", "account", "memory-alloc"]])
  #import sys; sys.exit()

  ### EMAIL
  if args.email:
    for netid in np.sort(ds.netid.unique()):
      usr = ds[ds.netid == netid].copy()
      total_jobs = usr.shape[0]
      bad_jobs           = usr[usr["Large-Memory-Needed?"] == "No"].shape[0]
      jobs_within_safety = usr[usr["within-safety"]].shape[0]
      is_physics = "physics" in usr.account.unique().tolist()
      #max_cores = 40 if is_physics else 32
      #too_many_cores = usr[usr.cores > max_cores].shape[0]
      if bad_jobs > 0:
        # TDO: create datascience if needed
        vfile = f"{args.files}/datascience/{netid}.email.csv"
        last_write_date = datetime(1970, 1, 1)
        if os.path.exists(vfile):
          last_write_date = datetime.fromtimestamp(os.path.getmtime(vfile))
        s = f"{get_first_name(netid)},\n\n"
        usr["memory-used"]  = usr["memory-used"].apply(lambda x: f"{x} GB")
        usr["memory-alloc"] = usr["memory-alloc"].apply(lambda x: f"{x} GB")
        max_mem = "380 GB" if is_physics else "190 GB"

        if (datetime.now().timestamp() - last_write_date.timestamp() >= 7 * HOURS_PER_DAY * SECONDS_PER_HOUR):
          cols = ["jobid", "netid", "memory-used", "memory-alloc", "Large-Memory-Needed?", "elapsed-hours"]
          renamings = {"elapsed-hours":"Hours", "jobid":"JobID", "netid":"NetID", "memory-used":"Memory-Used", \
                       "cpu-hours":"CPU-hours", "memory-alloc":"Memory-Allocated"}
          usr = usr[cols].rename(columns=renamings)
          s += "Below are jobs that ran on the large-memory (datascience) nodes on Della in the "
          s += "\npast 7 days:"
          s += "\n\n"
          s += "\n".join([3 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
          s += "\n"
          if bad_jobs == total_jobs and jobs_within_safety == 0:
            s += textwrap.dedent(f"""
            The large-memory nodes should only be used for jobs that require {max_mem} or more.
            It appears that none of the jobs above needed one of these nodes. For future jobs,
            please lower the value of the --mem-per-cpu or --mem Slurm directive so that the
            overall memory requirement is less than {max_mem}. You should use the smallest value
            possible but include an extra 20% for safety. For more information on Slurm and
            CPU memory:

               https://researchcomputing.princeton.edu/support/knowledge-base/memory

            Users that continually run jobs on the large-memory nodes without justification
            risk losing access to these nodes since it prevents others from getting their
            work done.
            """)
          elif bad_jobs == total_jobs and jobs_within_safety != 0:
            s += textwrap.dedent(f"""
            The large-memory nodes should only be used for jobs that require {max_mem} or more.
            It appears that none of the jobs above needed one of these nodes. However, some job(s)
            were within 20% of the threshold value. If possible please lower the value of the
            --mem-per-cpu or --mem Slurm directive so that the overall memory requirement of
            the job is less than {max_mem}. You should use the smallest value possible but include
            an extra 20% for safety. For more information on Slurm and CPU memory:

               https://researchcomputing.princeton.edu/support/knowledge-base/memory

            Users that continually run jobs on the large-memory nodes without justification
            risk losing access to these nodes since it prevents others from getting their
            work done.
            """)
          else:
            s += textwrap.dedent(f"""
            The large-memory nodes should only be used for jobs that require {max_mem} or more.
            It appears that some of the jobs above did not need these nodes. Whenever possible
            please lower the value of the --mem-per-cpu or --mem Slurm directive so that the
            overall memory requirement of the job is less than {max_mem}. You should use the
            smallest value possible but include an extra 20% for safety.

            We understand that for some jobs it can be very difficult or impossible to estimate
            the memory requirements. For those jobs please disregard this email.

            For more information on Slurm and CPU memory:

               https://researchcomputing.princeton.edu/support/knowledge-base/memory
            """)

          #if too_many_cores:
          #  s += "\n"
          #  text = (
          #         f"You have job(s) that requested more than {max_cores} CPU-cores. This may be the reason why "
          #         f"your job(s) ran on the large-memory nodes. If so then please use at most {max_cores} CPU-cores "
          #          "per job to avoid running jobs on the large-memory nodes."
          #         )
          #  s += "\n".join(textwrap.wrap(text, width=80))
          #  s += "\n"

          # if a physics user does not specify -p physics and they request more than 194G then it will go to datascience
          if is_physics and usr[usr["memory-alloc"] < 380].shape[0]:
            s += textwrap.dedent(f"""
            Your Slurm account is physics. This means that you have acccess to nodes with 380 GB
            of memory that are not available to other users. To access these nodes add the
            following line to your Slurm scripts: 
            
               #SBATCH --partition=physics
            """)
        
          s += textwrap.dedent(f"""
          Add the following lines to your Slurm scripts to receive an email report with
          memory usage information after each job finishes:

             #SBATCH --mail-type=begin
             #SBATCH --mail-type=end
             #SBATCH --mail-user={netid}@princeton.edu
          
          Replying to this email will open a support ticket with CSES. Let us know if we
          can be of help.
          """)

          # send email and append violation file
          date_today = datetime.now().strftime("%Y-%m-%d")
          cal = USFederalHolidayCalendar()
          us_holiday = date_today in cal.holidays()
          pu_holidays  = ["2022-07-05", "2022-11-25", "2022-12-23", "2022-12-26", "2022-12-30", "2023-01-02",
                          "2023-06-16", "2023-11-24", "2023-12-26", "2023-01-02", "2023-06-19"]
          pu_holiday = date_today in pu_holidays
          if args.email and not us_holiday and not pu_holiday:
            send_email(s,   f"{netid}@princeton.edu", subject="Jobs on the Della large-memory nodes", sender="cses@princeton.edu")
            send_email(s,  "halverson@princeton.edu", subject="Jobs on the Della large-memory nodes", sender="cses@princeton.edu")
            usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M")
            if os.path.exists(vfile):
              curr = pd.read_csv(vfile)
              curr = pd.concat([curr, usr]).drop_duplicates()
              curr.to_csv(vfile, index=False, header=True)
            else:
              usr.to_csv(vfile, index=False, header=True)
          else:
            print(s)
  ## EMAIL END

  cols = ["netid", "jobid", "state", "memory-used", "memory-alloc", "elapsed-hours", "cores", "account", "Large-Memory-Needed?"]
  return ds[cols].sort_values(by="memory-used", ascending=False)

def xpu_efficiencies_of_heaviest_users(df, cluster, partitions, xpu):
  # compute proportion using as much data as possible
  pr = df[(df.cluster == cluster) & (df.partition.isin(partitions)) & pd.notna(df[f"{xpu}-seconds"])].copy()
  pr = pr.groupby("netid").agg({f"{xpu}-seconds":np.sum}).reset_index(drop=False)
  pr["proportion(%)"] = pr[f"{xpu}-seconds"].apply(lambda x: round(100 * x / pr[f"{xpu}-seconds"].sum()))
  pr = pr.rename(columns={f"{xpu}-seconds":f"{xpu}-seconds-all"}).sort_values(by=f"{xpu}-seconds-all", ascending=False)

  # 2nd dataframe based on admincomment
  ce = df[(df.cluster == cluster) & \
          (df["elapsedraw"] >= 0.5 * SECONDS_PER_HOUR) & \
          (df.partition.isin(partitions))].copy()
  ce = ce[ce.admincomment != {}]  # ignore running jobs
  ce = ce.merge(pr, how="left", on="netid")
  if ce.empty: return pd.DataFrame()  # prevents next line from failing
  if xpu == "cpu":
    ce[f"{xpu}-tuples"] = ce.apply(lambda row: cpu_efficiency(row["admincomment"], row["elapsedraw"], row["jobid"], row["cluster"]), axis="columns")
  else:
    ce[f"{xpu}-tuples"] = ce.apply(lambda row: gpu_efficiency(row["admincomment"], row["elapsedraw"], row["jobid"], row["cluster"]), axis="columns")
  ce[f"{xpu}-seconds-used"]  = ce[f"{xpu}-tuples"].apply(lambda x: x[0])
  ce[f"{xpu}-seconds-total"] = ce[f"{xpu}-tuples"].apply(lambda x: x[1])
  ce["interactive"] = ce["jobname"].apply(lambda x: 1 if x.startswith("sys/dashboard") or x.startswith("interactive") else 0)
  d = {"netid":np.size, f"{xpu}-seconds-used":np.sum, f"{xpu}-seconds-total":np.sum, \
       "proportion(%)":"first", f"{xpu}-seconds-all":"first", "cores":np.mean, "interactive":np.sum}
  ce = ce.groupby("netid").agg(d).rename(columns={"netid":"jobs"})
  ce = ce.sort_values(by=f"{xpu}-seconds-total", ascending=False).reset_index(drop=False)
  ce = ce.head(15)
  ce["eff(%)"] = 100.0 * ce[f"{xpu}-seconds-used"] / ce[f"{xpu}-seconds-total"]
  if ce.empty: return pd.DataFrame()  # prevents next line from failing
  ce[f"{xpu}-hours"] = ce.apply(lambda row: round(row[f"{xpu}-seconds-total"] / SECONDS_PER_HOUR), axis="columns")
  ce = ce[ce[f"{xpu}-seconds-all"] > 0]  # prevents next line from failing if cpu-only users in top 15 e.g., traverse
  ce["coverage"] = ce.apply(lambda row: round(row[f"{xpu}-seconds-total"] / row[f"{xpu}-seconds-all"], 2), axis="columns")
  ce["eff(%)"] = ce["eff(%)"].apply(lambda x: round(x))
  ce["cores"] = ce["cores"].apply(lambda x: round(x, 1))
  eff_thres = 60 if xpu == "cpu" else 15
  filters = (ce["eff(%)"] < eff_thres) & (ce["proportion(%)"] >= 4)
  ce = ce[["netid", f"{xpu}-hours", "proportion(%)", "eff(%)", "jobs", "interactive", "cores", "coverage"]][filters]
  ce.index += 1

  ###########
  ## EMAIL ##
  ###########
  if args.email:
    for netid in ce.netid:
      vfile = f"{args.files}/low_xpu_efficiency/{netid}.email.csv"
      last_write_date = datetime(1970, 1, 1)
      if os.path.exists(vfile):
        last_write_date = datetime.fromtimestamp(os.path.getmtime(vfile))
      s = f"{get_first_name(netid)},\n\n"
      if (datetime.now().timestamp() - last_write_date.timestamp() >= 6 * HOURS_PER_DAY * SECONDS_PER_HOUR):
        usr = ce[ce.netid == netid].copy()
        cols = ["netid", f"{xpu}-hours", "proportion(%)", "eff(%)"]
        renamings = {"eff(%)":"Efficiency(%)", "jobid":"JobID", "netid":"NetID", "proportion(%)":"Proportion(%)", \
                     "cpu-hours":"CPU-hours", "gpu-hours":"GPU-hours"}
        usr = usr[cols].rename(columns=renamings)
        s += "\n\n"
        # make period clear and number of jobs and rank
        s += "You are heavy user.\n"
        s += "\n".join([5 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
        s += "\n\n"
        s += "\n"
        s += textwrap.dedent(f"""
        Run the jobstats command 
             https://researchcomputing.princeton.edu/support/knowledge-base/job-stats

        The most common reasons for low CPU efficiency are:

             1. Using too many CPU-cores for parallel jobs. You can find the optimal number
                of CPU-cores by performing a scaling analysis:
                https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis

             2. Writing or reading files to/from the /tigress or /projects storage systems.
                Job output should be written to /scratch/gpfs/{netid} which is a much
                faster filesystem. For more on the filesystems:
                https://researchcomputing.princeton.edu/support/knowledge-base/data-storage

             3. Slow MPI calls potentially due to load imbalance. Make sure you are using an environment module
                and not MPICH from a Conda install.
             4. Are you using srun in place of mpirun? Binding to cores instead of NUMA nodes.

        Consult the documentation or write to mailing list of the code for additional
        reasons for low {xpu.upper()} efficiency.
        """)
        s += "\n"
        s += textwrap.dedent(f"""
        Add the following lines to your Slurm scripts to receive an email report with
        {xpu.upper()} utilization information after each job finishes:

             #SBATCH --mail-type=begin
             #SBATCH --mail-type=end
             #SBATCH --mail-user={netid}@princeton.edu
        
        Replying to this email will open a support ticket with CSES. Let us know if we
        can be of help in resolving this matter.
        """)
        send_email(s, "halverson@princeton.edu", subject=f"Low {xpu.upper()} utilization on {cluster}", sender="cses@princeton.edu")
        #send_email(s,   f"{netid}@princeton.edu", subject="Jobs with zero GPU utilization", sender="cses@princeton.edu")
        usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M")
        if os.path.exists(vfile):
          curr = pd.read_csv(vfile)
          curr = pd.concat([curr, usr]).drop_duplicates()
          curr.to_csv(vfile, index=False, header=True)
        else:
          usr.to_csv(vfile, index=False, header=True)
      else:
        print(s)
  print("Exiting low efficiency email routine")
  ###########
  return ce

def get_stats_for_running_job(jobid, cluster):
  import importlib.machinery
  import importlib.util
  print("jobid:", jobid, "cluster:", cluster)
  cluster = cluster.replace("tiger", "tiger2")
  loader = importlib.machinery.SourceFileLoader('jobstats', '/home/jdh4/.local/bin/jobstats') # until remove args.simple
  spec = importlib.util.spec_from_loader('jobstats', loader)
  mymodule = importlib.util.module_from_spec(spec)
  loader.exec_module(mymodule)
  stats = mymodule.JobStats(jobid=jobid, cluster=cluster)
  time.sleep(0.5)
  return eval(stats.report_job_json(False))

def emails_gpu_jobs_zero_util(df):
  fltr = ((df.cluster == "della")    & (df.partition == "gpu")) | \
         ((df.cluster == "tiger")    & (df.partition == "gpu")) | \
         ((df.cluster == "traverse") & (df.partition == "all"))
  em = df[fltr & \
          (df.elapsedraw >= 1 * SECONDS_PER_HOUR) & \
          (df.elapsedraw <  4 * SECONDS_PER_HOUR) & \
          (df.state == "RUNNING") & \
          (df.gpus > 0)].copy()
  em["jobstats"] = em.apply(lambda row: get_stats_for_running_job(row["jobid"], row["cluster"]), axis='columns')
  em["GPUs-Unused"] = em.jobstats.apply(num_gpus_with_zero_util)
  em["interactive"] = em["jobname"].apply(lambda x: True if x.startswith("sys/dashboard") or x.startswith("interactive") else False)
  msk = (em["interactive"]) & (em.gpus == 1) & (em["limit-minutes"] <= 8 * MINUTES_PER_HOUR)
  em = em[~msk]
  em = em[em["GPUs-Unused"] > 0][["jobid", "netid", "cluster", "gpus", "GPUs-Unused", "elapsedraw"]]
  renamings = {"gpus":"GPUs-Allocated", "jobid":"JobID", "netid":"NetID", "cluster":"Cluster"}
  em.rename(columns=renamings, inplace=True)
  for netid in em.NetID.unique():
    vfile = f"/tigress/jdh4/utilities/job_defense_shield/violations/{netid}.violations.email.csv"
    last_write_date = datetime(1970, 1, 1).date()
    if os.path.exists(vfile):
      last_write_date = datetime.fromtimestamp(os.path.getmtime(vfile)).date()
    if (last_write_date != datetime.now().date()):
      s = f"{get_first_name(netid)},\n\n"
      usr = em[em.NetID == netid].copy()

      single_job = bool(usr.shape[0] == 1)
      multi_gpu_jobs = bool(usr[usr["GPUs-Allocated"] > 1].shape[0])

      if single_job and (not multi_gpu_jobs):
        version = "the GPU"
        usr["GPU-Util"] = "0%"
        zero = (
        'We measure the utilization of each allocated GPU every 30 seconds. '
        'All measurements for the job above have been reported as 0%. '
        'You can see this by running the "jobstats" command, for example:'
        )
      elif single_job and multi_gpu_jobs and bool(usr[usr["GPUs-Allocated"] == usr["GPUs-Unused"]].shape[0]):
        version = "the GPUs"
        usr["GPU-Util"] = "0%"
        zero = (
        'We measure the utilization of each allocated GPU every 30 seconds. '
        'All measurements for the job above have been reported as 0%. '
        'You can see this by running the "jobstats" command, for example:'
        )
      elif single_job and multi_gpu_jobs and (not bool(usr[usr["GPUs-Allocated"] != usr["GPUs-Unused"]].shape[0])):
        version = "all of the GPUs"
        usr["GPU-Unused-Util"] = "0%"
        zero = (
        'We measure the utilization of each allocated GPU every 30 seconds. '
        'All measurements for at least one of the GPUs used in the job above have been reported as 0%. '
        'You can see this by running the "jobstats" command, for example:'
        )
      elif (not single_job) and (not multi_gpu_jobs):
        version = "the GPUs"
        usr["GPU-Util"] = "0%"
        zero = (
        'We measure the utilization of each allocated GPU every 30 seconds. '
        'All measurements for the GPUs used in the jobs above have been reported as 0%. '
        'You can see this by running the "jobstats" command, for example:'
        )
      else:
        version = "the GPU(s)"
        usr["GPU-Unused-Util"] = "0%"
        zero = (
        'We measure the utilization of each allocated GPU every 30 seconds. '
        'All measurements for at least one of the GPUs used in each job above have been reported as 0%. '
        'You can see this by running the "jobstats" command, for example:'
        )

      usr["Hours"] = usr.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR, 1))
      usr.drop(columns=["elapsedraw"], inplace=True)

      if single_job:
        text = (
         "You have a GPU job that has been running for more than 1 hour but\n"
        f"it appears to not be using {version}:\n\n"
        )
        s += "\n".join(textwrap.wrap(text, width=80))
      else:
        text = (
         "You have GPU jobs that have been running for more than 1 hour but\n"
        f"they appear to not be using {version}:\n\n"
        )
        s += "\n".join(textwrap.wrap(text, width=80))
      s += "\n\n"
      s += "\n".join([5 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
      s += "\n\n"

      s += "\n".join(textwrap.wrap(zero, width=80))
      s += "\n\n"
      s += f"     $ jobstats {usr.JobID.values[0]}"
      s += "\n\n"

      text = (
      'Follow the link at the bottom of the "jobstats" output for more detailed information.'
      )
      s += "\n".join(textwrap.wrap(text, width=80))
      s += "\n\n"

      version = "GPU is" if single_job and (not multi_gpu_jobs) else "GPU(s) are"
      text = (
      f'If the {version} not being used then you need to take action now to resolve this issue. '
       'Wasting resources prevents other users from getting their work done and it causes your subsequent jobs to have a lower priority. '
       'Users that continually underutilize the GPUs risk having their accounts suspended.'
      )
      s += "\n".join(textwrap.wrap(text, width=80))
      s += "\n\n"
      text = (
      'Toward resolving this issue please consult the documentation for the code that you are running. Is it GPU-enabled? '
      )
      s += "\n".join(textwrap.wrap(text, width=80))
      s += "\n"
      s += textwrap.dedent("""
      For general information about GPU computing and Slurm job statistics:

           https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing
           https://researchcomputing.princeton.edu/support/knowledge-base/job-stats
      """)
      s += "\n"
      version = "job" if single_job else "jobs"
      text = (
      f'Please consider canceling the {version} listed above using the "scancel" command, for example:'
      )
      s += "\n".join(textwrap.wrap(text, width=80))
      s += "\n\n"
      s += f"     $ scancel {usr.JobID.values[0]}"
      s += "\n"
      s += textwrap.dedent(f"""
      Add the following lines to your Slurm scripts to receive an email report with
      GPU utilization information after each job finishes:

           #SBATCH --mail-type=begin
           #SBATCH --mail-type=end
           #SBATCH --mail-user={netid}@princeton.edu
      
      Replying to this email will open a support ticket with CSES. Let us know if we
      can be of help in resolving this matter.
      """)

      # send email and append violation file
      date_today = datetime.now().strftime("%Y-%m-%d")
      cal = USFederalHolidayCalendar()
      us_holiday = date_today in cal.holidays()
      pu_holidays  = ["2022-07-05", "2022-11-25", "2022-12-23", "2022-12-26", "2022-12-30", "2023-01-02",
                      "2023-06-16", "2023-11-24", "2023-12-26", "2023-01-02", "2023-06-19"]
      pu_holiday = date_today in pu_holidays
      if args.email and not us_holiday and not pu_holiday:
        send_email(s,   f"{netid}@princeton.edu", subject="Jobs with zero GPU utilization", sender="cses@princeton.edu")
        send_email(s,  "halverson@princeton.edu", subject="Jobs with zero GPU utilization", sender="cses@princeton.edu")
        usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M")
        if "GPU-Util"        in usr.columns: usr.drop(columns=["GPU-Util"],        inplace=True)
        if "GPU-Unused-Util" in usr.columns: usr.drop(columns=["GPU-Unused-Util"], inplace=True)
        if os.path.exists(vfile):
          curr = pd.read_csv(vfile)
          curr = pd.concat([curr, usr]).drop_duplicates()
          curr.to_csv(vfile, index=False, header=True)
        else:
          usr.to_csv(vfile, index=False, header=True)
      else:
        print(s)
  print("Exiting GPUs email routine")
  return None

def gpu_jobs_zero_util(df, cluster, partitions):
  zu = df[(df.cluster == cluster) & \
          (df["elapsed-hours"] >= 1) & \
          (df.partition.isin(partitions))].copy()
  zu = zu[zu.admincomment != {}]  # ignore running jobs
  zu["interactive"] = zu["jobname"].apply(lambda x: True if x.startswith("sys/dashboard") or x.startswith("interactive") else False)
  zu["gpus-unused"] = zu.admincomment.apply(num_gpus_with_zero_util)
  zu = zu[zu["gpus-unused"] > 0].rename(columns={"elapsed-hours":"hours"}).sort_values(by="netid")
  zu.state = zu.state.apply(lambda x: JOBSTATES[x])
  return zu[["netid", "gpus", "gpus-unused", "jobid", "state", "hours", "interactive", "start-date"]]

def cpu_jobs_zero_util(df, cluster, partitions):
  zu = df[(df.cluster == cluster) & \
          (df["elapsed-hours"] > 1) & \
          (df.partition.isin(partitions))].copy()
  zu = zu[zu.admincomment != {}]  # ignore running jobs
  zu["interactive"] = zu["jobname"].apply(lambda x: True if x.startswith("sys/dashboard") or x.startswith("interactive") else False)
  def cpu_nodes_with_zero_util(d):
    ct = 0
    for node in d['nodes']:
      try:
        cpu_time = d['nodes'][node]['total_time']
      except:
        print("total_time not found")
        return 0
      else:
        if float(cpu_time) == 0: ct += 1
    return ct
  zu["nodes-unused"] = zu.admincomment.apply(cpu_nodes_with_zero_util)
  zu = zu[zu["nodes-unused"] > 0].rename(columns={"elapsed-hours":"hours"}).sort_values(by="netid")
  zu.state = zu.state.apply(lambda x: JOBSTATES[x])
  return zu[["netid", "nodes", "nodes-unused", "jobid", "state", "hours", "interactive", "start-date"]]

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
  if m.empty: return pd.DataFrame()  # prevents next line from failing
  m["eff(%)"] = m.apply(lambda row: gpu_efficiency(row["admincomment"], row["elapsedraw"], row["jobid"], row["cluster"], single=True) if row["admincomment"] != {} else "", axis="columns")
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
  filters = ~f["position"].isin(["G3", "G4", "G5", "G6", "G7", "G8"])
  return f[["netid", "position", "dept", "cluster", "jobs", "num-failed"]][filters]

def jobs_with_the_most_cores(df):
  """Top 10 users with the highest number of CPU-cores in a job. Only one job per user is shown."""
  cols = ["jobid", "netid", "cluster", "cores", "nodes", "gpus", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment", "elapsedraw"]
  c = df[cols].groupby("netid").apply(lambda d: d.iloc[d["cores"].argmax()]).copy()
  c = c.sort_values("cores", ascending=False)[:10].drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  c.state = c.state.apply(lambda x: JOBSTATES[x])
  if c.empty: return pd.DataFrame()  # prevents next line from failing
  c["eff(%)"] = c.apply(lambda row: cpu_efficiency(row["admincomment"], row["elapsedraw"], row["jobid"], row["cluster"], single=True) if row["admincomment"] != {} else "", axis="columns")
  cols = cols[:8] + ["hours", "eff(%)"]
  return c[cols]

def jobs_with_the_most_gpus(df):
  """Top 10 users with the highest number of GPUs in a job. Only one job per user is shown."""
  cols = ["jobid", "netid", "cluster", "gpus", "nodes", "cores", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment", "elapsedraw"]
  df = df[(df.partition != "cryoem") & (df.netid != "cryoem")]
  g = df[cols].groupby("netid").apply(lambda d: d.iloc[d["gpus"].argmax()]).copy()
  g = g.sort_values("gpus", ascending=False)[:10].drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  g.state = g.state.apply(lambda x: JOBSTATES[x])
  if g.empty: return pd.DataFrame()  # prevents next line from failing
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
  cols = ["jobid", "netid", "cluster", "nodes", "cores", "qos", "partition", "s-days", "e-days"]
  q = q[cols].groupby("netid").apply(lambda d: d.iloc[d["s-days"].argmax()]).sort_values("s-days", ascending=False)
  return q[q["s-days"] >= 4][:10]

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

def print_report_of_users_with_continual_underutilization():
  files = sorted(glob.glob(f"{args.files}/*.csv"))
  if len(files) == 0:
    print("No underutilization files found.")
    return None
  today = datetime.now().date()
  if args.zero_gpu_utilization:
    print("=====================================================")
    print("           ZERO GPU UTILIZATION EMAILS SENT")
    print("=====================================================")
    print("                                                  today")
    print("                                                    |")
    print("                                                    V")
    for f in files:
      netid = f.split("/")[-1].split(".")[0]
      df = pd.read_csv(f)
      df["when"] = df.email_sent.apply(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M").date())
      hits = df.when.unique()
      row = [today - timedelta(days=i) in hits for i in range(30)]
      s = " " * (8 - len(netid)) + netid + "@princeton.edu "
      s += ''.join(["X" if r else "_" for r in row])[::-1]
      print(s)
  print("\n=====================================================")
  text = (
          "\nRequestor: <>@princeton.edu\n"
          "Hi <>,\n"
          "You have been sent multiple emails over the last severals days about zero GPU utilization."
          "At this time you need to either (1) resolve the issue, (2) ask us to help you resolve "
          "the issue, or (3) stop running these jobs."
          "If you fail to take action then we will be forced to suspend your account and notify"
          "your sponsor. The GPUs are valuable resources."
      )
  print("\n".join(textwrap.wrap(text, width=80, replace_whitespace=False)))
  return None
 

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Slurm job alerts')
  parser.add_argument('-d', '--days', type=int, default=14, metavar='N',
                      help='Create report over N previous days from now (default: 14)')
  parser.add_argument('--zero-cpu-utilization', action='store_true', default=False,
                      help='Identify CPU jobs with zero utilization')
  parser.add_argument('--zero-gpu-utilization', action='store_true', default=False,
                      help='Identify running GPU jobs with zero utilization')
  parser.add_argument('--low-xpu-efficiency', action='store_true', default=False,
                      help='Identify users with low CPU/GPU efficiency')
  parser.add_argument('--datascience', action='store_true', default=False,
                      help='Identify users that unjustly used the datascience nodes')
  parser.add_argument('--low-time-efficiency', action='store_true', default=False,
                      help='Identify users that are over-allocating CPU/GPU time')
  parser.add_argument('--files', default="/tigress/jdh4/utilities/job_defense_shield/violations",
                      help='Path to the underutilization files')
  parser.add_argument('--email', action='store_true', default=False,
                      help='Send output via email')
  parser.add_argument('--check', action='store_true', default=False,
                      help='Create report of users who may be ignoring the automated emails')
  args = parser.parse_args()

  # pandas display settings
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", 1000)

  if args.check:
    _ = print_report_of_users_with_continual_underutilization()
    sys.exit()

  # convert slurm timestamps to seconds
  os.environ["SLURM_TIME_FORMAT"] = "%s"

  flags = "-L -a -X -P -n"
  start_date = datetime.now() - timedelta(days=args.days)
  # jobname must be last in line below to catch "|" chars in raw_dataframe_from_sacct()
  fields = "jobid,user,cluster,account,partition,cputimeraw,elapsedraw,timelimitraw,nnodes,ncpus,alloctres,submit,eligible,start,end,qos,state,admincomment,jobname"
  renamings = {"user":"netid", "cputimeraw":"cpu-seconds", "nnodes":"nodes", "ncpus":"cores", "timelimitraw":"limit-minutes"}
  numeric_fields = ["cpu-seconds", "elapsedraw", "limit-minutes", "nodes", "cores", "submit", "eligible"]
  raw = raw_dataframe_from_sacct(flags, start_date, fields, renamings, numeric_fields, use_cache=not args.email)

  raw = raw[~raw.cluster.isin(["tukey", "perseus"])]
  raw.cluster   =   raw.cluster.str.replace("tiger2", "tiger")
  raw.partition = raw.partition.str.replace("datascience", "datasci")
  raw.state = raw.state.apply(lambda x: "CANCELLED" if "CANCEL" in x else x)

  # df excludes pending jobs
  df = raw.copy()
  df = df[pd.notnull(df.alloctres) & (df.alloctres != "")]
  df.start = df.start.astype("int64")
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

  cls = (("della", "Della (CPU)", ("cpu", "datasci", "physics"), "cpu"), \
         ("della", "Della (GPU)", ("gpu",), "gpu"), \
         ("stellar", "Stellar (AMD)", ("bigmem", "cimes"), "cpu"), \
         ("stellar", "Stellar (Intel)", ("all", "pppl", "pu", "serial"), "cpu"), \
         ("tiger", "TigerCPU", ("cpu", "ext", "serial"), "cpu"), \
         ("tiger", "TigerGPU", ("gpu",), "gpu"), \
         ("traverse", "Traverse (GPU)", ("all",), "gpu"))
  if args.low_time_efficiency:
    ####################################
    ### used allocated cpu/gpu hours ###
    ####################################
    s += "           Unused allocated CPU/GPU-Hours (of COMPLETED 2+ hour jobs)"
    for cluster, name, partitions, xpu in cls:
      un = unused_allocated_hours_of_completed(df, cluster, partitions, xpu)
      if not un.empty:
        df_str = un.to_string(index=True, justify="center")
        s += add_dividers(df_str, title=name, pre="\n\n")

  ####### consider jobs in the last N days only #######
  #thres_days = 7
  #df = df[df.start >= time.time() - thres_days * HOURS_PER_DAY * SECONDS_PER_HOUR]
  #s += f"\n\n\n            --- the next set of tables below is for the last {thres_days} days ---"

  if args.low_xpu_efficiency:
    ##########################
    ### cpu/gpu efficiency ###
    ##########################
    s += "\n\n\n      CPU/GPU Efficiencies of top 15 users (30+ minute jobs, ignoring running)"
    for cluster, name, partitions, xpu in cls:
      un = xpu_efficiencies_of_heaviest_users(df, cluster, partitions, xpu)
      if not un.empty:
        df_str = un.to_string(index=True, justify="center")
        s += add_dividers(df_str, title=name, pre="\n\n")

  ####### consider jobs in the last N days only #######
  #thres_days = 3
  #df = df[df.start >= time.time() - thres_days * HOURS_PER_DAY * SECONDS_PER_HOUR]
  #s += f"\n\n\n            --- everything below is for the last {thres_days} days ---"

  ##########################################
  ### zero utilization on a GPU by email ###
  ##########################################
  if args.zero_gpu_utilization:
    _ = emails_gpu_jobs_zero_util(df)

    #################################
    ### zero utilization on a GPU ###
    #################################
    #first_hit = False
    #for cluster, name, partitions in [("tiger", "TigerGPU", ("gpu",)), \
    #                                  ("della", "Della (GPU)", ("gpu",)), \
    #                                  ("traverse", "Traverse (GPU)", ("all",))]:
      ############
    #  zu = gpu_jobs_zero_util(df, cluster, partitions)
    #  if not zu.empty:
    #    if not first_hit:
    #      s += "\n\n\n    Zero utilization on a GPU (1+ hour jobs, ignoring running)"
    #      first_hit = True
    #    df_str = zu.to_string(index=False, justify="center")
    #    s += add_dividers(df_str, title=name, pre="\n\n")

  if args.zero_cpu_utilization:
    ######################################
    ### zero utilization on a CPU node ###
    ######################################
    first_hit = False
    for cluster, name, partitions in [("tiger", "TigerCPU", ("cpu", "ext", "serial")), \
                                      ("della", "Della (CPU)", ("cpu", "datasci", "physics")), \
                                      ("stellar", "Stellar (Intel)", ("all", "pppl", "pu", "serial")), \
                                      ("stellar", "Stellar (AMD)", ("cimes",))]:
      zu = cpu_jobs_zero_util(df, cluster, partitions)
      if not zu.empty:
        if not first_hit:
          s += "\n\n\n    Zero utilization on a CPU (2+ hour jobs, ignoring running)"
          first_hit = True
        df_str = zu.to_string(index=False, justify="center")
        s += add_dividers(df_str, title=name, pre="\n\n")

  if False:
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

  if args.datascience:
    ###################
    ### datascience ###
    ###################
    ds = datascience_node_violators(df)
    if not ds.empty:
      df_str = ds.to_string(index=False, justify="center")
      s += add_dividers(df_str, title="Datascience jobs that didn't need to be (all jobs, 1+ hours)", pre="\n\n\n")

  if False:
    ##############################
    ### last jobs are failures ###
    ##############################
    fl = recent_jobs_all_failures(df)
    if not fl.empty:
      df_str = fl.to_string(index=False, justify="center")
      s += add_dividers(df_str, title="All jobs failed on last day (4+ jobs)")

  if False:
    ##############################
    ### large cpu and gpu jobs ###
    ##############################
    df_str = jobs_with_the_most_cores(df).to_string(index=False, justify="center")
    s += add_dividers(df_str, title="Jobs with the most CPU-cores (1 job per user)")
    df_str = jobs_with_the_most_gpus(df).to_string(index=False, justify="center")
    s += add_dividers(df_str, title="Jobs with the most GPUs (1 job per user, ignoring cryoem)")

  if False:
    ###########################
    ### longest queue times ###
    ###########################
    df_str = longest_queue_times(raw).to_string(index=False, justify="center")
    s += add_dividers(df_str, title="Longest queue times of PENDING jobs (1 job per user, ignoring job arrays)")

  if args.email:
    pass
    #send_email(s, "halverson@princeton.edu")
    #send_email(s, "kabbey@princeton.edu")
  else:
    print(s)

  print(datetime.now())
