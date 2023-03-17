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
from abc import abstractmethod
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from utils import JOBSTATES
from utils import SECONDS_PER_MINUTE
from utils import SECONDS_PER_HOUR
from utils import MINUTES_PER_HOUR
from utils import HOURS_PER_DAY
from utils import get_first_name
from utils import send_email

from efficiency import get_stats_dict
from efficiency import cpu_efficiency
from efficiency import gpu_efficiency
from efficiency import cpu_memory_usage
from efficiency import gpu_memory_usage_eff_tuples
from efficiency import num_gpus_with_zero_util
from efficiency import max_cpu_memory_used_per_node

from alert.unused_allocated_time import unused_allocated_hours_of_completed
from alert.fragmentation import multinode_cpu_fragmentation
from alert.fragmentation import multinode_gpu_fragmentation
from alert.info import jobs_with_the_most_cores
from alert.info import jobs_with_the_most_gpus
from alert.info import longest_queue_times
from alert.datascience import datascience_node_violators
from alert.xpu_efficiency import xpu_efficiencies_of_heaviest_users
from alert.zero_gpu_utilization import gpu_jobs_zero_util
from alert.zero_gpu_utilization import active_gpu_jobs_with_zero_utilization
from alert.zero_cpu_utilization import cpu_jobs_zero_util

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

def print_report_of_users_with_continual_underutilization(mydir, title, days=30):
  files = sorted(glob.glob(f"{args.files}/{mydir}/*.csv"))
  if len(files) == 0:
    print(f"No underutilization files found in {args.files}/{mydir}")
    return None
  today = datetime.now().date()
  day_ticks = days
  print("=====================================================")
  print(f"           {title} EMAILS SENT")
  print("=====================================================")
  max_netid = max([len(f.split("/")[-1].split(".")[0]) for f in files])
  print(" " * (max_netid + len("@princeton.edu") + day_ticks - 2) + "today")
  print(" " * (max_netid + len("@princeton.edu") + day_ticks - 0) + "|")
  print(" " * (max_netid + len("@princeton.edu") + day_ticks - 0) + "V")
  for f in files:
    netid = f.split("/")[-1].split(".")[0]
    df = pd.read_csv(f)
    df["when"] = df.email_sent.apply(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M").date())
    hits = df.when.unique()
    row = [today - timedelta(days=i) in hits for i in range(day_ticks)]
    s = " " * (8 - len(netid)) + netid + "@princeton.edu "
    s += ''.join(["X" if r else "_" for r in row])[::-1]
    if "X" in s: print(s)
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
  #print("\n".join(textwrap.wrap(text, width=80, replace_whitespace=False)))
  return None


class Alert:
  """Base class for all alerts."""
  def __init__(self, df, days_between_emails, violation, vpath, subject):
      self.df = df
      self.days_between_emails = days_between_emails
      self.violation = violation
      self.vpath = vpath
      self.vbase = os.path.join(self.vpath, self.violation)
      self.subject = subject
      self._filter_and_add_new_fields()
      # create directory to store user violations
      if not os.path.exists(self.vbase):
          os.mkdir(self.vbase)

  @abstractmethod
  def _filter_and_add_new_fields(self):
      """Filter the dataframe and add new fields.

      Returns:
          None
      """

  @abstractmethod
  def send_emails_to_users(self, emails: bool) -> None:
      """Send emails to the users.

      Returns:
          None
      """

  @abstractmethod
  def send_report_to_admins(self):
      """Send an email report to system administrators.

      Returns:
          None
      """

  def has_sufficient_time_passed_since_last_email(self, vfile) -> bool:
      """Return boolean specifying whether sufficient time has passed."""
      last_write_date = datetime(1970, 1, 1)
      if os.path.exists(vfile):
          last_write_date = datetime.fromtimestamp(os.path.getmtime(vfile))
      seconds_since_last_email = datetime.now().timestamp() - last_write_date.timestamp()
      seconds_threshold = self.days_between_emails * HOURS_PER_DAY * SECONDS_PER_HOUR
      return seconds_since_last_email >= seconds_threshold

  def __len__(self):
      return self.df.shape[0]

  def __str__(self):
      return self.df.to_string()

  @staticmethod
  def is_today_a_work_day() -> bool:
      """Determine if today is a work day."""
      date_today = datetime.now().strftime("%Y-%m-%d")
      cal = USFederalHolidayCalendar()
      us_holiday = date_today in cal.holidays()
      pu_holidays = ["2023-05-29", "2023-06-16", "2023-07-04", 
                     "2023-09-04", "2023-11-23", "2023-11-24",
                     "2023-12-26", "2024-01-02", "2024-01-15"]
      pu_holiday = date_today in pu_holidays
      day_of_week = datetime.strptime(date_today, "%Y-%m-%d").weekday()
      return (not us_holiday) and (not pu_holiday) and (day_of_week < 5)

  @staticmethod
  def update_violation_log(usr, vfile) -> None:
      """Append the new violations to file."""
      usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M")
      if os.path.exists(vfile):
          curr = pd.read_csv(vfile)
          curr = pd.concat([curr, usr]).drop_duplicates()
          curr.to_csv(vfile, index=False, header=True)
      else:
          usr.to_csv(vfile, index=False, header=True)


class MultiInstanceGPU(Alert):

  def __init__(self, df, days_between_emails, violation, vpath, subject):
      super().__init__(df, days_between_emails, violation, vpath, subject)

  def _filter_and_add_new_fields(self):
      # filter the dataframe
      self.df = self.df[(self.df.cluster == "della") &
                        (self.df.partition == "gpu") &
                        (self.df.cores == 1) &
                        (self.df.gpus == 1) &
                        (self.df.admincomment != {}) &
                        (self.df.state != "OUT_OF_MEMORY") &
                        (self.df["elapsed-hours"] >= 1)].copy()
      # add new fields
      self.df["gpu-memused-memalloc-eff"] = self.df.apply(lambda row:
                                            gpu_memory_usage_eff_tuples(row["admincomment"],
                                                                        row["jobid"],
                                                                        row["cluster"]),
                                                                        axis="columns")
      # next three lines are valid since only one GPU per job
      self.df["gpu-memory-used"]  = self.df["gpu-memused-memalloc-eff"].apply(lambda x: x[0][0])
      self.df["gpu-memory-alloc"] = self.df["gpu-memused-memalloc-eff"].apply(lambda x: x[0][1])
      self.df["gpu-eff"]          = self.df["gpu-memused-memalloc-eff"].apply(lambda x: x[0][2])
      # add CPU memory usage
      self.df["cpu-memused-memalloc"] = self.df.apply(lambda row:
                                        cpu_memory_usage(row["admincomment"],
                                                         row["jobid"],
                                                         row["cluster"]),
                                                         axis="columns")
      self.df["cpu-memory-used"] = self.df["cpu-memused-memalloc"].apply(lambda x: x[0])
      gpu_eff_threshold = 15 # percent
      gpu_mem_threshold = 10 # GB
      cpu_mem_threshold = 32 # GB
      self.df = self.df[(self.df["gpu-eff"] <= gpu_eff_threshold) &
                        (self.df["gpu-eff"] != 0) &
                        (self.df["gpu-memory-used"] < gpu_mem_threshold) &
                        (self.df["cpu-memory-used"] < cpu_mem_threshold)]
      self.df = self.df[["jobid", "netid", "gpu-eff", "gpu-memory-used", "cpu-memory-used", "elapsed-hours"]]

  def send_emails_to_users(self, emails):
      if emails and Alert.is_today_a_work_day():
          for user in self.df.netid.unique():
              vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
              if self.has_sufficient_time_passed_since_last_email(vfile):
                  usr = self.df[self.df.netid == user].copy()
                  renamings = {"elapsed-hours":"Hours",
                               "jobid":"JobID",
                               "netid":"NetID",
                               "gpu-memory-used":"GPU-Mem-Used",
                               "cpu-memory-used":"CPU-Mem-Used",
                               "cpu-hours":"CPU-hours",
                               "gpu-eff":"GPU-Util"}
                  usr = usr.rename(columns=renamings)
                  usr["CPU-Mem-Used"] = usr["CPU-Mem-Used"].apply(lambda x: f"{round(x)} GB")
                  usr["GPU-Mem-Used"] = usr["GPU-Mem-Used"].apply(lambda x: f"{round(x)} GB")
                  usr["GPU-Util"] = usr["GPU-Util"].apply(lambda x: f"{round(x)}%")
                  s = f"{get_first_name(user)},\n\n"
                  s += f"Below are jobs that ran on an A100 GPU on Della in the past {self.days_between_emails} days:"
                  s += "\n\n"
                  s += "\n".join([2 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
                  s += "\n"
                  s += textwrap.dedent(f"""
                  The jobs above have a low GPU utilization and they use less than 10 GB of GPU
                  memory and less than 32 GB of CPU memory. Such jobs could be run on the MIG
                  GPUs. A MIG GPU is essentially a small A100 GPU with 1/7th the performance and
                  memory of an A100. To run on a MIG GPU, add the "partition" directive to your
                  Slurm script:

                    #SBATCH --nodes=1
                    #SBATCH --ntasks=1
                    #SBATCH --cpus-per-task=1
                    #SBATCH --gres=gpu:1
                    #SBATCH --partition=mig

                  For interactive sessions use, for example:

                    $ salloc --nodes=1 --ntasks=1 --time=1:00:00 --gres=gpu:1 --partition=mig

                  If you are using Jupyter OnDemand then set the "Custom partition" to "mig" when
                  creating the session.

                  A job can use a MIG GPU when the following constraints are satisfied:

                    1. The required number of GPUs is 1
                    2. The required number of CPU-cores is 1
                    3. The required GPU memory is less than 10 GB
                    4. The required CPU memory is less than 32 GB

                  All MIG jobs are automatically allocated 32 GB of CPU memory and 10 GB of GPU
                  memory.

                  By running future jobs on the MIG GPUs you will experience shorter queue
                  times and you will help keep A100 GPUs free for jobs that need them. Since
                  your jobs satisfy the above constraints, please use the MIG GPUs. For more:

                    https://researchcomputing.princeton.edu/systems/della#gpus

                  As an alternative to MIG, you may consider trying to improve the GPU
                  utilization of your code. A good place to start is the mailing list of
                  the software you are using.

                  Replying to this email will open a support ticket with CSES. Let us know if we
                  can be of help.
                  """)
                  send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                  send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                  print(s)

                  # append the new violations to the log file
                  Alert.update_violation_log(usr, vfile)
                  

class ZeroUtilGPUHours(Alert):

  def __init__(self, df, days_between_emails, violation, vpath, subject):
      super().__init__(df, days_between_emails, violation, vpath, subject)

  def _filter_and_add_new_fields(self):
      # filter the dataframe
      self.df = self.df[(self.df.cluster == "della") &
                        (self.df.partition == "gpu") &
                        (self.df.gpus > 0) &
                        (self.df.admincomment != {}) &
                        (self.df["elapsedraw"] >= SECONDS_PER_HOUR)].copy()
      # add new fields
      self.df["gpus-unused"] = self.df.admincomment.apply(num_gpus_with_zero_util)
      self.df = self.df[self.df["gpus-unused"] > 0]
      self.df["zero-util-hours"] = self.df["gpus-unused"] * self.df["elapsedraw"] / SECONDS_PER_HOUR
      self.df["GPU-Unused-Util"] = "0%"
      self.df = self.df[["jobid", "netid", "gpus", "gpus-unused", "GPU-Unused-Util", "zero-util-hours"]]
      # for each user sum the number of GPU-hours with zero GPU utilization
      self.gp = self.df.groupby("netid").agg({"zero-util-hours":np.sum, "netid":np.size})
      self.gp = self.gp.rename(columns={"netid":"jobs"})
      self.gp = self.gp.sort_values(by="zero-util-hours", ascending=False).reset_index(drop=False)
      self.gp = self.gp[self.gp["zero-util-hours"] >= 100]
      self.df["zero-util-hours"] = self.df["zero-util-hours"].apply(round)

  def send_emails_to_users(self, emails):
      if emails and Alert.is_today_a_work_day():
          for user in self.gp.netid.unique():
              vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
              if self.has_sufficient_time_passed_since_last_email(vfile):
                  usr = self.df[self.df.netid == user].copy()
                  renamings = {"zero-util-hours":"Zero-Util-GPU-Hours",
                               "jobid":"JobID",
                               "netid":"NetID",
                               "gpus-unused":"GPUs-Unused",
                               "gpus":"GPUs"}
                  usr = usr.rename(columns=renamings)
                  zero_hours = round(self.gp[self.gp.netid == user]["zero-util-hours"].values[0])
                  s =  f"Requestor: {user}@princeton.edu\n\n"
                  s += f"{get_first_name(user)},\n\n"
                  s += f"You have consumed {zero_hours} GPU-hours at 0% GPU utilization in the past {self.days_between_emails} days on\n"
                  s +=  "Della. This is a waste of valuable resources. Please monitor your jobs using the\n"
                  s +=  "\"jobstats\" command or the web interface:\n\n"
                  s +=  "  https://researchcomputing.princeton.edu/support/knowledge-base/job-stats\n\n"
                  s +=  "Below are the jobs with 0% GPU utilization:\n\n"
                  s +=  "\n".join([2 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
                  s +=  "\n"
                  s += textwrap.dedent(f"""
                  Please investigate the reason for the GPUs not being used. For instance, is the
                  code GPU-enabled? Can you use a MIG GPU instead of a full A100?

                  For general information about GPU computing at Princeton:

                    https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing

                  Replying to this email will open a support ticket with CSES. Let us know if we
                  can be of help.
                  """)
                  send_email(s,      "cses@princeton.edu", subject=f"{self.subject}", sender="halverson@princeton.edu")
                  send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                  print(s)

                  # append the new violations to the log file
                  Alert.update_violation_log(usr, vfile)

  def send_report_to_admins(self):
      return self.gp.head(10).to_string()


class MultinodeGPUFragmentation(Alert):

  def __init__(self, df, days_between_emails, violation, vpath, subject):
      super().__init__(df, days_between_emails, violation, vpath, subject)

  def _filter_and_add_new_fields(self):
      # filter the dataframe
      self.df = self.df[(self.df.cluster == "della") &
                        (self.df.partition == "gpu") &
                        (self.df.gpus > 0) &
                        (self.df.nodes > 1) &
                        (self.df.nodes == self.df.gpus) &
                        (self.df["elapsed-hours"] >= 1)].copy()
      if not self.df.empty:
          self.df["GPU-eff"] = self.df.apply(lambda row:
                                             gpu_efficiency(row["admincomment"],
                                                            row["elapsedraw"],
                                                            row["jobid"],
                                                            row["cluster"], single=True)
                                             if row["admincomment"] != {} else 999, axis="columns")
          self.gpu_util_thres = 50
          self.has_low_gpu_util = bool(self.df[self.df["GPU-eff"] < self.gpu_util_thres].shape[0])
          self.df["GPU-eff"] = self.df["GPU-eff"].apply(lambda x: "--" if x == 999 else f"{round(x)}%")
          self.df["GPUs-per-Node"] = 1
          self.df = self.df[["netid", "jobid", "gpus", "nodes", "GPUs-per-Node", "elapsed-hours", "GPU-eff"]]

  def send_emails_to_users(self, emails):
      if emails and Alert.is_today_a_work_day() and not self.df.empty:
          for user in self.df.netid.unique():
              vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
              if self.has_sufficient_time_passed_since_last_email(vfile):
                  usr = self.df[self.df.netid == user].copy()
                  renamings = {"netid":"NetID",
                               "jobid":"JobID",
                               "nodes":"Nodes",
                               "gpus":"GPUs",
                               "elapsed-hours":"Hours"}
                  usr = usr.rename(columns=renamings)
                  edays = self.days_between_emails
                  s =  f"{get_first_name(user)},\n\n"
                  s += f"Below are jobs that ran on Della in the past {edays} days that used 1 GPU per node\n"
                  s +=  "over multiple nodes:\n\n"
                  s +=  "\n".join([2 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
                  if self.has_low_gpu_util:
                      s +=  "\n\n"
                      s += f"There is at least one job with a GPU efficiency of less than {self.gpu_util_thres}%. In these cases\n"
                      s +=  "please consider using only 1 GPU per job to improve the efficiency."
                  s += "\n"
                  s += textwrap.dedent(f"""
                  The GPU nodes on Della have either 2 GPUs per node or 4 GPUs per node. For future
                  jobs, please try to use as few nodes as possible by allocating more GPUs per node.
                  This is done by modifying the --gres Slurm directive:

                    https://researchcomputing.princeton.edu/support/knowledge-base/slurm#gpus

                  For more information about the Della GPU nodes:

                    https://researchcomputing.princeton.edu/systems/della#gpus

                  When using more than 1 GPU per job, be sure to conduct a scaling analysis to find
                  the optimal number of GPUs:

                    https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis

                  Replying to this email will open a support ticket with CSES. Let us know if we
                  can be of help.
                  """)
                  send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                  send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                  print(s)

                  # append the new violations to the log file
                  Alert.update_violation_log(usr, vfile)

  def send_report_to_admins(self):
      return self.df.to_string()



if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Slurm job alerts')
  parser.add_argument('-d', '--days', type=int, default=14, metavar='N',
                      help='Create report over N previous days from now (default: 14)')
  parser.add_argument('--zero-cpu-utilization', action='store_true', default=False,
                      help='Identify CPU jobs with zero utilization')
  parser.add_argument('--zero-gpu-utilization', action='store_true', default=False,
                      help='Identify running GPU jobs with zero utilization')
  parser.add_argument('--zero-gpu-util-hours', action='store_true', default=False,
                      help='Identify users with the most zero GPU utilization hours')
  parser.add_argument('--low-xpu-efficiency', action='store_true', default=False,
                      help='Identify users with low CPU/GPU efficiency')
  parser.add_argument('--datascience', action='store_true', default=False,
                      help='Identify users that unjustly used the datascience nodes')
  parser.add_argument('--mig', action='store_true', default=False,
                      help='Identify users that should use MIG')
  parser.add_argument('--cpu-fragmentation', action='store_true', default=False,
                      help='Identify users that are splitting CPU jobs across too many nodes')
  parser.add_argument('--gpu-fragmentation', action='store_true', default=False,
                      help='Identify users that are splitting GPU jobs across too many nodes')
  parser.add_argument('--low-time-efficiency', action='store_true', default=False,
                      help='Identify users that are over-allocating time')
  parser.add_argument('--files', default="/tigress/jdh4/utilities/job_defense_shield/violations",
                      help='Path to the underutilization files')
  parser.add_argument('--email', action='store_true', default=False,
                      help='Send output via email')
  parser.add_argument('--watch', action='store_true', default=False,
                      help='Send output via email to halverson@princeton.edu')
  parser.add_argument('--check', action='store_true', default=False,
                      help='Create report of users who may be ignoring the automated emails')
  args = parser.parse_args()

  if args.check:
    if args.datascience:
      _ = print_report_of_users_with_continual_underutilization("datascience", "DATASCIENCE")
    if args.zero_gpu_utilization:
      _ = print_report_of_users_with_continual_underutilization("zero_gpu_utilization", "ZERO GPU UTILIZATION")
    if args.mig:
      _ = print_report_of_users_with_continual_underutilization("should_be_using_mig", "       MIG")
    if args.low_xpu_efficiency:
      _ = print_report_of_users_with_continual_underutilization("low_xpu_efficiency", "LOW CPU/GPU EFFICIENCY")
    sys.exit()

  # pandas display settings
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", 1000)

  # convert slurm timestamps to seconds
  os.environ["SLURM_TIME_FORMAT"] = "%s"

  flags = "-L -a -X -P -n"
  start_date = datetime.now() - timedelta(days=args.days)
  # jobname must be last in list below to catch "|" chars in raw_dataframe_from_sacct()
  fields = ["jobid",
            "user",
            "cluster",
            "account",
            "partition",
            "cputimeraw",
            "elapsedraw",
            "timelimitraw",
            "nnodes",
            "ncpus",
            "alloctres",
            "submit",
            "eligible",
            "start",
            "end",
            "qos",
            "state",
            "admincomment",
            "jobname"]  
  fields = ",".join(fields)
  assert fields.split(",")[-1] == "jobname"
  renamings = {"user":"netid",
               "cputimeraw":"cpu-seconds",
               "nnodes":"nodes",
               "ncpus":"cores",
               "timelimitraw":"limit-minutes"}
  numeric_fields = ["cpu-seconds", "elapsedraw", "limit-minutes", "nodes", "cores", "submit", "eligible"]
  use_cache = False if (args.email or args.watch) else True
  raw = raw_dataframe_from_sacct(flags, start_date, fields, renamings, numeric_fields, use_cache)

  raw = raw[~raw.cluster.isin(["tukey", "perseus"])]
  raw.cluster   =   raw.cluster.str.replace("tiger2", "tiger")
  raw.partition = raw.partition.str.replace("datascience", "datasci")
  raw = raw[pd.notna(raw.state)]
  raw.state = raw.state.apply(lambda x: "CANCELLED" if "CANCEL" in x else x)
  raw.cores = raw.cores.astype("int64")
  raw.nodes = raw.nodes.astype("int64")

  # df excludes pending jobs
  df = raw.copy()
  df = df[pd.notnull(df.alloctres) & (df.alloctres != "")]
  df.start = df.start.astype("int64")
  df = add_new_and_derived_fields(df)
  df.reset_index(drop=True, inplace=True)

  # header
  fmt = "%a %b %-d"
  s = f"{start_date.strftime(fmt)} - {datetime.now().strftime(fmt)}\n\n"
  s += f"Total users: {raw.netid.unique().size}\n"
  s += f"Total jobs:  {raw.shape[0]}\n\n"

  if args.mig:
      mig = MultiInstanceGPU(df,
                             days_between_emails=args.days,
                             violation="should_be_using_mig",
                             vpath=args.files,
                             subject="Consider Using the MIG GPUs on Della")
      mig.send_emails_to_users(args.email)

  if args.zero_gpu_util_hours:
      zero_gpu_hours = ZeroUtilGPUHours(df,
                             days_between_emails=args.days,
                             violation="zero_util_gpu_hours",
                             vpath=args.files,
                             subject="Many GPU-Hours with Zero GPU Utilization on Della")
      zero_gpu_hours.send_emails_to_users(args.email)
      s += add_dividers(zero_gpu_hours.send_report_to_admins(),
                        title="Zero Utilization GPU-Hours (of COMPLETED 1+ Hour Jobs)")

  if args.gpu_fragmentation:
      gpu_frag = MultinodeGPUFragmentation(df,
                             days_between_emails=args.days,
                             violation="gpu_fragmentation",
                             vpath=args.files,
                             subject="Fragmented GPU Jobs on Della")
      gpu_frag.send_emails_to_users(args.email)
      s += add_dividers(gpu_frag.send_report_to_admins(),
                        title="Multinode GPU jobs with fragmentation (all jobs, 1+ hours)")

  if not args.email:
    df.info()
    print(df.describe().astype("int64").T)
    print("\nTotal NaNs:", df.isnull().sum().sum(), "\n")

  cls = (("della", "Della (CPU)", ("cpu", "datasci", "physics"), "cpu"), \
         ("della", "Della (GPU)", ("gpu",), "gpu"), \
         ("stellar", "Stellar (AMD)", ("bigmem", "cimes"), "cpu"), \
         ("stellar", "Stellar (Intel)", ("all", "pppl", "pu", "serial"), "cpu"), \
         ("tiger", "TigerCPU", ("cpu", "ext", "serial"), "cpu"), \
         ("traverse", "Traverse (GPU)", ("all",), "gpu"))
  cls = (("della", "Della (CPU)", ("cpu",), "cpu"), \
         ("della", "Della (GPU)", ("gpu",), "gpu"), \
         ("stellar", "Stellar (Intel)", ("all", "pppl", "pu", "serial"), "cpu"), \
         ("tiger", "TigerCPU", ("cpu", "ext", "serial"), "cpu"))

  if args.low_xpu_efficiency:
    first_hit = False
    for cluster, cluster_name, partitions, xpu in cls:
      un = xpu_efficiencies_of_heaviest_users(df, cluster, cluster_name, partitions, xpu, args.email, args.files)
      if not un.empty:
        if not first_hit:
          s += "\n\n\n      CPU/GPU Efficiencies of top 15 users (30+ minute jobs, ignoring running)"
          first_hit = True
        df_str = un.to_string(index=True, justify="center")
        s += add_dividers(df_str, title=cluster_name, pre="\n\n")

  if args.low_time_efficiency:
    first_hit = False
    for cluster, cluster_name, partitions, xpu in cls:
      un = unused_allocated_hours_of_completed(df, cluster, cluster_name, partitions, xpu, args.email)
      if not un.empty:
        if not first_hit:
          s += "\n\n\n           Unused allocated CPU/GPU-Hours (of COMPLETED 2+ hour jobs)"
          first_hit = True
        df_str = un.to_string(index=True, justify="center")
        s += add_dividers(df_str, title=cluster_name, pre="\n\n")

  if args.zero_gpu_utilization:
    em = active_gpu_jobs_with_zero_utilization(df, args.email, args.files)
    if False and not em.empty:
      df_str = em.to_string(index=True, justify="center")
      s += add_dividers(df_str, title="ZERO GPU", pre="\n\n")

  if False:
    name = "    Zero utilization on a GPU (1+ hour jobs, ignoring running)"
    zu = gpu_jobs_zero_util(df)
    if not zu.empty:
      df_str = zu.to_string(index=False, justify="center")
      s += add_dividers(df_str, title=name)

  if args.zero_cpu_utilization:
    first_hit = False
    for cluster, cluster_name, partitions in [("tiger", "TigerCPU", ("cpu", "ext", "serial")), \
                                      ("della", "Della (CPU)", ("cpu", "datasci", "physics")), \
                                      ("stellar", "Stellar (Intel)", ("all", "pppl", "pu", "serial")), \
                                      ("stellar", "Stellar (AMD)", ("cimes",))]:
      zu = cpu_jobs_zero_util(df, cluster, partitions)
      if not zu.empty:
        if not first_hit:
          s += "\n\n\n    Zero utilization on a CPU (2+ hour jobs, ignoring running)"
          first_hit = True
        df_str = zu.to_string(index=False, justify="center")
        s += add_dividers(df_str, title=cluster_name, pre="\n\n")

  if args.cpu_fragmentation:
    fg = multinode_cpu_fragmentation(df, args.email, args.files)
    if not fg.empty:
      df_str = fg.to_string(index=False, justify="center")
      s += add_dividers(df_str, title="Multinode CPU jobs with < 14 cores per node (all jobs, 2+ hours)", pre="\n\n\n")
 
  if args.datascience:
    ds = datascience_node_violators(df, args.email, args.files)
    if False and not ds.empty:
      df_str = ds.to_string(index=False, justify="center")
      s += add_dividers(df_str, title="Datascience jobs that didn't need to be (all jobs, 1+ hours)", pre="\n\n\n")

  if True:
    df_str = jobs_with_the_most_cores(df).to_string(index=False, justify="center")
    s += add_dividers(df_str, title="Jobs with the most CPU-cores (1 job per user)")
    df_str = jobs_with_the_most_gpus(df).to_string(index=False, justify="center")
    s += add_dividers(df_str, title="Jobs with the most GPUs (1 job per user, ignoring cryoem)")

  if True:
    df_str = longest_queue_times(raw).to_string(index=False, justify="center")
    s += add_dividers(df_str, title="Longest queue times of PENDING jobs (1 job per user, ignoring job arrays)")

  if args.watch:
    send_email(s, "halverson@princeton.edu", subject="Cluster utilization report", sender="halverson@princeton.edu")

  print(s)
  print(datetime.now())
