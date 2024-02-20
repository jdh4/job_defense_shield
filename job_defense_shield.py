import os
import sys
import argparse
import subprocess
from datetime import datetime
from datetime import timedelta
import pandas as pd
import yaml

from utils import SECONDS_PER_MINUTE
from utils import SECONDS_PER_HOUR
from utils import is_today_a_work_day
from utils import send_email
from utils import show_history_of_emails_sent

from efficiency import get_stats_dict
from alert.datascience import datascience_node_violators

from alert.zero_gpu_utilization import ZeroGpuUtilization
from alert.mig import MultiInstanceGPU
from alert.zero_util_gpu_hours import ZeroUtilGPUHours
from alert.gpu_fragmentation import MultinodeGPUFragmentation
from alert.excess_cpu_memory import ExcessCPUMemory
from alert.hard_warning_cpu_memory import HardWarningCPUMemory
from alert.zero_cpu_utilization import ZeroCPU
from alert.most_gpus import MostGPUs
from alert.most_cores import MostCores
from alert.utilization_overview import UtilizationOverview
from alert.utilization_by_slurm_account import UtilizationBySlurmAccount
from alert.longest_queued import LongestQueuedJobs
from alert.jobs_overview import JobsOverview
from alert.excessive_time_limits import ExcessiveTimeLimits
from alert.serial_code_using_multiple_cores import SerialCodeUsingMultipleCores
from alert.fragmentation import MultinodeCPUFragmentation
from alert.xpu_efficiency import LowEfficiency
from alert.active_cpu_memory import ActiveCPUMemory


def raw_dataframe_from_sacct(flags, start_date, fields, renamings=[], numeric_fields=[], use_cache=False):
    fname = f"cache_sacct_{start_date.strftime('%Y%m%d')}.csv"
    if use_cache and os.path.exists(fname):
        print("\nUsing cache file.\n", flush=True)
        rw = pd.read_csv(fname, low_memory=False)
    else:
        ymd = start_date.strftime('%Y-%m-%d')
        hms = start_date.strftime('%H:%M:%S')
        cmd = f"sacct {flags} -S {ymd}T{hms} -E now -o {fields}"
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


def gpus_per_job(tres: str) -> int:
    """Return the number of GPUs used."""
    # billing=8,cpu=4,mem=16G,node=1
    # billing=112,cpu=112,gres/gpu=16,mem=33600M,node=4
    if "gres/gpu=" in tres:
        for part in tres.split(","):
            if "gres/gpu=" in part:
                gpus = int(part.split("=")[-1])
                assert gpus > 0
                return gpus
        raise Exception(f"GPU count not extracted for {tres}")
    else:
        return 0


def add_new_and_derived_fields(df):
    df["cpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["cores"], axis='columns')
    df["gpus"] = df.alloctres.apply(gpus_per_job)
    df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')
    def is_gpu_job(tres):
        return 1 if "gres/gpu=" in tres and "gres/gpu=0" not in tres else 0
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


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Job Defense Shield')
  parser.add_argument('--zero-cpu-utilization', action='store_true', default=False,
                      help='Identify completed CPU jobs with zero utilization')
  parser.add_argument('--zero-gpu-utilization', action='store_true', default=False,
                      help='Identify running GPU jobs with zero utilization')
  parser.add_argument('--zero-util-gpu-hours', action='store_true', default=False,
                      help='Identify users with the most zero GPU utilization hours')
  parser.add_argument('--low-xpu-efficiency', action='store_true', default=False,
                      help='Identify users with low CPU/GPU efficiency')
  parser.add_argument('--datascience', action='store_true', default=False,
                      help='Identify jobs that unjustly used the datascience nodes')
  parser.add_argument('--excess-cpu-memory', action='store_true', default=False,
                      help='Identify users that are allocating too much CPU memory')
  parser.add_argument('--hard-warning-cpu-memory', action='store_true', default=False,
                      help='Send a hard warning email to users that are allocating too much CPU memory')
  parser.add_argument('--active-cpu-memory', action='store_true', default=False,
                      help='Identify running jobs that are allocating too much CPU memory')
  parser.add_argument('--mig', action='store_true', default=False,
                      help='Identify jobs that should use MIG')
  parser.add_argument('--cpu-fragmentation', action='store_true', default=False,
                      help='Identify CPU jobs that are split across too many nodes')
  parser.add_argument('--gpu-fragmentation', action='store_true', default=False,
                      help='Identify GPU jobs that are split across too many nodes')
  parser.add_argument('--excessive-time', action='store_true', default=False,
                      help='Identify users with excessive run time limits')
  parser.add_argument('--serial-using-multiple', action='store_true', default=False,
                      help='Indentify serial codes using multiple CPU-cores')
  parser.add_argument('--utilization-overview', action='store_true', default=False,
                      help='Generate a utilization report by cluster and partition')
  parser.add_argument('--utilization-by-slurm-account', action='store_true', default=False,
                      help='Generate a utilization report by cluster, partition and account')
  parser.add_argument('--longest-queued', action='store_true', default=False,
                      help='List the longest queued jobs')
  parser.add_argument('--most-cores', action='store_true', default=False,
                      help='List the largest jobs by number of allocated CPU-cores')
  parser.add_argument('--most-gpus', action='store_true', default=False,
                      help='List the largest jobs by number of allocated GPUs')
  parser.add_argument('--jobs-overview', action='store_true', default=False,
                      help='List the users with the most jobs')
  parser.add_argument('-d', '--days', type=int, default=14, metavar='N',
                      help='Use job data over N previous days from now (default: 14)')
  parser.add_argument('-M', '--clusters', type=str, default="all",
                      help='Specify cluster(s) (e.g., --clusters=della,traverse)')
  parser.add_argument('-r', '--partition', type=str, default="",
                      help='Specify partition(s) (e.g., --partition=gpu,mig)')
  parser.add_argument('--num-top-users', type=int, default=15,
                      help='Specify the number of users to consider')
  parser.add_argument('--files', type=str, default="/tigress/jdh4/utilities/job_defense_shield/violations",
                      help='Path to the underutilization log files')
  parser.add_argument('--config-file', type=str, default=None,
                      help='Absolute path to the configuration file')
  parser.add_argument('--email', action='store_true', default=False,
                      help='Send email alerts to users')
  parser.add_argument('--report', action='store_true', default=False,
                      help='Send an email report to administrators')
  parser.add_argument('--check', action='store_true', default=False,
                      help='Show the history of emails sent to users')
  args = parser.parse_args()

  # read configuration file
  jds_path = os.path.join(os.path.dirname(__file__), "config.yaml")
  cwd_path = os.path.join(os.getcwd(), "config.yaml")
  if args.config_file and os.path.isfile(args.config_file):
      print(f"Configuration file: {args.config_file}")
      with open(args.config_file, "r", encoding="utf-8") as fp:
          cfg = yaml.safe_load(fp)
  elif args.config_file and not os.path.isfile(args.config_file):
      print(f"Configuration file does not exist: {args.config_file}. Exiting ...")
      sys.exit()
  elif args.config_file is None and os.path.isfile(jds_path):
      print(f"Configuration file: {jds_path}")
      with open(jds_path, "r", encoding="utf-8") as fp:
          cfg = yaml.safe_load(fp)
  elif args.config_file is None and os.path.isfile(cwd_path):
      print(f"Configuration file: {cwd_path}")
      with open(cwd_path, "r", encoding="utf-8") as fp:
          cfg = yaml.safe_load(fp)
  else:
      print("Configuration file not found. Exiting ...")
      sys.exit()

  if args.email and (os.environ["USER"] not in ["jdh4", "slurm"]):
      print("The --email flag can currently only used by jdh4 and slurm to send emails. Exiting ...")
      sys.exit()

  #######################
  ## CHECK EMAILS SENT ##
  #######################
  if args.check:
      if args.zero_gpu_utilization:
          show_history_of_emails_sent(args.files,
                                      "zero_gpu_utilization",
                                      "ZERO GPU UTILIZATION OF A RUNNING JOB",
                                      args.days)
      if args.zero_cpu_utilization:
          show_history_of_emails_sent(args.files,
                                      "zero_cpu_utilization",
                                      "ZERO CPU UTILIZATION OF A RUNNING JOB",
                                      args.days)
      if args.mig:
          show_history_of_emails_sent(args.files,
                                      "should_be_using_mig",
                                      "SHOULD HAVE USED MIG",
                                      args.days)
      if args.low_xpu_efficiency:
          show_history_of_emails_sent(args.files,
                                      "low_xpu_efficiency",
                                      "LOW CPU/GPU EFFICIENCY",
                                      args.days)
      if args.zero_util_gpu_hours:
          show_history_of_emails_sent(args.files,
                                      "zero_util_gpu_hours",
                                      "ZERO UTILIZATION GPU-HOURS",
                                      args.days)
      if args.gpu_fragmentation:
          show_history_of_emails_sent(args.files,
                                      "gpu_fragmentation",
                                      "1 GPU PER NODE",
                                      args.days)
      if args.excess_cpu_memory:
          show_history_of_emails_sent(args.files,
                                      "excess_cpu_memory",
                                      "EXCESS CPU MEMORY",
                                      args.days)
      if args.hard_warning_cpu_memory:
          show_history_of_emails_sent(args.files,
                                      "hard_warning_cpu_memory",
                                      "HARD WARNING CPU MEMORY",
                                      args.days)
      if args.cpu_fragmentation:
          show_history_of_emails_sent(args.files,
                                      "cpu_fragmentation",
                                      "CPU FRAGMENTATION PER NODE",
                                      args.days)
      if args.datascience:
          show_history_of_emails_sent(args.files,
                                      "datascience",
                                      "DATASCIENCE",
                                      args.days) 
      if args.serial_using_multiple:
          show_history_of_emails_sent(args.files,
                                      "serial_using_multiple",
                                      "SERIAL CODE USING MULTIPLE CPU-CORES",
                                      args.days)
      if args.excessive_time:
          show_history_of_emails_sent(args.files,
                                      "excessive_time_limits",
                                      "EXCESSIVE TIME LIMITS",
                                      args.days)
      if args.most_gpus or args.most_cores or args.longest_queued:
          print("Nothing to check for --most-gpus, --most-cores or --longest-queued.")
      sys.exit()

  # pandas display settings
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", 1000)

  # convert slurm timestamps to seconds
  os.environ["SLURM_TIME_FORMAT"] = "%s"

  flags = f"-a -X -P -n --clusters={args.clusters.replace('tiger', 'tiger2')}"
  if args.partition:
      flags = f"{flags} --partition={args.partition}"
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
  numeric_fields = ["cpu-seconds",
                    "elapsedraw",
                    "limit-minutes",
                    "nodes",
                    "cores",
                    "submit",
                    "eligible"]
  use_cache = False if (args.email or args.report) else True
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
  print(f"Number of rows (before): {df.shape[0]}")
  df.start = df.apply(lambda row: row["eligible"]
                                  if row["start"] == "Unknown"
                                  else row["start"],
                                  axis="columns")
  df = df[pd.notnull(df.alloctres) &
          (df.alloctres != "") &
          pd.notnull(df.start) &
          (~df.start.isin(["", "None"]))]
  print(f"Number of rows  (after): {df.shape[0]}")
  df.start = df.start.astype("int64")

  correction = False
  if correction:
      df["secs-from-start"] = df["start"] - start_date.timestamp()
      df["secs-from-start"] = df["secs-from-start"].apply(lambda x: x if x < 0 else 0)
      df["elapsedraw"] = df["elapsedraw"] + df["secs-from-start"]

  df = add_new_and_derived_fields(df)
  df.reset_index(drop=True, inplace=True)

  #if not args.email:
  #if debug:
  if False:
      df.info()
      print(df.describe().astype("int64").T)
      print("\nTotal NaNs:", df.isnull().sum().sum())

  fmt = "%a %b %-d"
  s = f"{start_date.strftime(fmt)} - {datetime.now().strftime(fmt)}"
 
  ############################################
  ## RUNNING JOBS WITH ZERO GPU UTILIZATION ##
  ############################################
  if args.zero_gpu_utilization:
      alerts = [alert for alert in cfg.keys() if "zero-gpu-utilization" in alert]
      for alert in alerts:
          zero_gpu = ZeroGpuUtilization(df,
                                        days_between_emails=args.days,
                                        violation="zero_gpu_utilization",
                                        vpath=args.files,
                                        subject="Jobs with Zero GPU Utilization",
                                        **cfg[alert])
      if args.email:
          zero_gpu.send_emails_to_users()

  ######################### 
  ## ZERO UTIL GPU-HOURS ##
  ######################### 
  if args.zero_util_gpu_hours:
      zero_gpu_hours = ZeroUtilGPUHours(df,
                             days_between_emails=args.days,
                             violation="zero_util_gpu_hours",
                             vpath=args.files,
                             subject="WARNING OF ACCOUNT SUSPENSION: Underutilization of the GPUs on Della")
      if args.email and is_today_a_work_day():
          zero_gpu_hours.send_emails_to_users()
      title="Zero Utilization GPU-Hours (of COMPLETED 1+ Hour Jobs)"
      s += zero_gpu_hours.generate_report_for_admins(title)

  #######################
  ## GPU FRAGMENTATION ##
  #######################
  if args.gpu_fragmentation:
      gpu_frag = MultinodeGPUFragmentation(df,
                             days_between_emails=args.days,
                             violation="gpu_fragmentation",
                             vpath=args.files,
                             subject="Fragmented GPU Jobs on Della",
                             cluster="della",
                             partition="gpu")
      if args.email and is_today_a_work_day():
          gpu_frag.send_emails_to_users()
      title = "Multinode GPU jobs with fragmentation (all jobs, 1+ hours)"
      s += gpu_frag.generate_report_for_admins(title)

  ############################
  ## LOW CPU/GPU EFFICIENCY ##
  ############################
  if args.low_xpu_efficiency:
      alerts = [alert for alert in cfg.keys() if "low-xpu-efficiency" in alert]
      for alert in alerts:
          low_eff = LowEfficiency(df,
                                  days_between_emails=args.days,
                                  violation="low_xpu_efficiency",
                                  vpath=args.files,
                                  subject="Jobs with Low Efficiency",
                                  **cfg[alert])
          if args.email and is_today_a_work_day():
              low_eff.send_emails_to_users()
          title = f"{low_eff.cluster_name} efficiencies of top {low_eff.num_top_users} users (30+ minute jobs, ignoring running)"
          s += low_eff.generate_report_for_admins(title, keep_index=True)

  #################
  ## DATASCIENCE ##
  #################
  if args.datascience:
      ds = datascience_node_violators(df, args.email, args.files)

  #######################
  ## EXCESS CPU MEMORY ##
  #######################
  if args.excess_cpu_memory:
      alerts = [alert for alert in cfg.keys() if "excess-cpu-memory" in alert]
      for alert in alerts:
          mem_hours = ExcessCPUMemory(df,
                                      days_between_emails=args.days,
                                      violation="excess_cpu_memory",
                                      vpath=args.files,
                                      subject="Requesting Too Much CPU Memory",
                                      **cfg[alert])
          if args.email and is_today_a_work_day():
              mem_hours.send_emails_to_users()
          title = "TB-Hours (1+ hour jobs, ignoring approximately full node jobs)"
          s += mem_hours.generate_report_for_admins(title, keep_index=True)

  if args.hard_warning_cpu_memory:
      mem = HardWarningCPUMemory(df,
                                 days_between_emails=90,
                                 violation="hard_warning_cpu_memory",
                                 vpath=args.files,
                                 subject="ACTION REQUIRED: Requesting Too Much CPU Memory",
                                 cluster="della",
                                 partition="cpu")
      if args.email and is_today_a_work_day():
          mem.send_emails_to_users()

  #######################
  ## ACTIVE CPU MEMORY ##
  #######################
  if args.active_cpu_memory:
      mem = ActiveCPUMemory(df,
                            days_between_emails=args.days,
                            violation="active_cpu_memory",
                            vpath=args.files,
                            subject="Requesting Too Much CPU Memory for Jobs on Della")
      #if args.email and is_today_a_work_day():
      mem.send_emails_to_users()
      title = "Actively running jobs allocating too much memory"
      s += mem.generate_report_for_admins(title)

  ###########################
  ## EXCESSIVE TIME LIMITS ##
  ###########################
  if args.excessive_time:
      low_time = ExcessiveTimeLimits(df,
                             days_between_emails=args.days,
                             violation="excessive_time_limits",
                             vpath=args.files,
                             subject="Requesting Too Much Time for Jobs on Della",
                             cluster="della",
                             partition="cpu")
      if args.email and is_today_a_work_day():
          low_time.send_emails_to_users()
      title = "Excessive time limits (all jobs, 1+ hours)"
      s += low_time.generate_report_for_admins(title)

  ######################################
  ## SERIAL CODE USING MULTIPLE CORES ##
  ######################################
  if args.serial_using_multiple:
      serial = SerialCodeUsingMultipleCores(df,
                                        days_between_emails=args.days,
                                        violation="serial_using_multiple",
                                        vpath=args.files,
                                        subject="Serial Jobs Using Multiple CPU-cores")
      if args.email and is_today_a_work_day():
          serial.send_emails_to_users()
      title = "Potential serial codes using multiple CPU-cores (Della cpu)"
      s += serial.generate_report_for_admins(title, keep_index=True)

  ###################
  ## ZERO CPU UTIL ##
  ###################
  if args.zero_cpu_utilization:
      zero_cpu = ZeroCPU(df,
                         days_between_emails=args.days,
                         violation="zero_cpu_utilization",
                         vpath=args.files,
                         subject="Jobs with Zero CPU Utilization")
      if args.email and is_today_a_work_day():
          zero_cpu.send_emails_to_users()
      title = "Jobs with Zero CPU Utilization (1+ hours)"
      s += zero_cpu.generate_report_for_admins(title, keep_index=False)

  ######################
  ## CPU FRAGMENTATION #
  ######################
  if args.cpu_fragmentation:
      cpu_frag = MultinodeCPUFragmentation(df,
                                          days_between_emails=args.days,
                                          violation="cpu_fragmentation",
                                          vpath=args.files,
                                          subject="Jobs Using Too Many Nodes")
      if args.email and is_today_a_work_day():
          cpu_frag.send_emails_to_users()
      title = "CPU fragmentation (1+ hours)"
      s += cpu_frag.generate_report_for_admins(title, keep_index=False)

  ####################################
  ## JOBS THAT SHOULD HAVE USED MIG ##
  ####################################
  if args.mig:
      alerts = [alert for alert in cfg.keys() if "should-be-using-mig" in alert]
      for alert in alerts:
          mig = MultiInstanceGPU(df,
                                 days_between_emails=args.days,
                                 violation="should_be_using_mig",
                                 vpath=args.files,
                                 subject="Consider Using the MIG GPUs on Della",
                                 **cfg[alert])
          if args.email and is_today_a_work_day():
              mig.send_emails_to_users()
          s += mig.generate_report_for_admins("Could Have Been MIG Jobs")


  ##########################
  ## UTILIZATION OVERVIEW ##
  ##########################
  if args.utilization_overview:
      util = UtilizationOverview(df,
                                 days_between_emails=args.days,
                                 violation="null",
                                 vpath=args.files,
                                 subject="")
      title = "Utilization Overview"
      s += util.generate_report_for_admins(title)

  ##################################
  ## UTILIZATION BY SLURM ACCOUNT ##
  ##################################
  if args.utilization_by_slurm_account:
      util = UtilizationBySlurmAccount(df,
                                       days_between_emails=args.days,
                                       violation="null",
                                       vpath=args.files,
                                       subject="")
      title = "Utilization by Slurm Account"
      s += util.generate_report_for_admins(title)

  #########################
  ## LONGEST QUEUED JOBS ##
  #########################
  if args.longest_queued:
      queued = LongestQueuedJobs(raw,
                           days_between_emails=args.days,
                           violation="null",
                           vpath=args.files,
                           subject="")
      title = "Longest queue times (1 job per user, ignoring job arrays, 4+ days)"
      s += queued.generate_report_for_admins(title)

  ###################
  ## JOBS OVERVIEW ##
  ###################
  if args.jobs_overview:
      jobs = JobsOverview(df,
                          days_between_emails=args.days,
                          violation="null",
                          vpath=args.files,
                          subject="")
      title = "Most jobs (1 second or longer -- ignoring running and pending)"
      s += jobs.generate_report_for_admins(title)

  ################
  ## MOST CORES ##
  ################
  if args.most_cores:
      most_cores = MostCores(df,
                             days_between_emails=args.days,
                             violation="null",
                             vpath=args.files,
                             subject="")
      title = "Jobs with the most CPU-cores (1 job per user)"
      s += most_cores.generate_report_for_admins(title)

  ###############
  ## MOST GPUS ##
  ###############
  if args.most_gpus:
      most_gpus = MostGPUs(df,
                           days_between_emails=args.days,
                           violation="null",
                           vpath=args.files,
                           subject="")
      title = "Jobs with the most GPUs (1 job per user, ignoring cryoem)"
      s += most_gpus.generate_report_for_admins(title)

  ########################## 
  ## SEND EMAIL TO ADMINS ##
  ########################## 
  if args.report:
    send_email(s, "halverson@princeton.edu", subject="Cluster utilization report", sender="halverson@princeton.edu")

  print(s, end="\n\n")
  print(datetime.now())
