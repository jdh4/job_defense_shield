import os
import textwrap
from datetime import datetime
from time import sleep
import pandas as pd
from utils import JOBSTATES
from utils import SECONDS_PER_HOUR
from utils import MINUTES_PER_HOUR
from utils import get_first_name
from utils import send_email
from efficiency import num_gpus_with_zero_util
from pandas.tseries.holiday import USFederalHolidayCalendar

def get_stats_for_running_job(jobid, cluster):
  """Get the job statistics for running jobs by calling jobstats"""
  import importlib.machinery
  import importlib.util
  cluster = cluster.replace("tiger", "tiger2")
  loader = importlib.machinery.SourceFileLoader('jobstats', '/usr/local/bin/jobstats')
  spec = importlib.util.spec_from_loader('jobstats', loader)
  mymodule = importlib.util.module_from_spec(spec)
  loader.exec_module(mymodule)
  stats = mymodule.JobStats(jobid=jobid, cluster=cluster, prom_server="http://vigilant2:8480")
  sleep(1)
  return eval(stats.report_job_json(False))

def active_gpu_jobs_with_zero_utilization(df, email, vpath):
  fltr = ((df.cluster == "della")    & (df.partition == "gpu")) | \
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

  if email:
    for netid in em.NetID.unique():
      vfile = f"{vpath}/zero_gpu_utilization/{netid}.violations.email.csv"
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

        version = "job" if single_job else "jobs"
        text = (
        f'Please consider canceling the {version} listed above by using the "scancel" command, for example:'
        )
        s += "\n".join(textwrap.wrap(text, width=80))
        s += "\n\n"
        s += f"     $ scancel {usr.JobID.values[0]}"
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
        For general information about GPU computing at Princeton:

             https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing

        Please monitor your jobs using the "jobstats" command and the web interface:

             https://researchcomputing.princeton.edu/support/knowledge-base/job-stats
        """)
        s += textwrap.dedent(f"""
        Add the following lines to your Slurm scripts to receive an email report with
        GPU utilization information after each job finishes:

             #SBATCH --mail-type=end
             #SBATCH --mail-user={netid}@princeton.edu

        Consider attending an in-person Research Computing help session for assistance:

             https://researchcomputing.princeton.edu/support/help-sessions
        
        Replying to this automated email will open a support ticket with Research
        Computing. Let us know if we can be of help.
        """)

        # send email and append violation file
        date_today = datetime.now().strftime("%Y-%m-%d")
        cal = USFederalHolidayCalendar()
        us_holiday = date_today in cal.holidays()
        pu_holidays = ["2022-07-05", "2022-11-25", "2022-12-23", "2022-12-26", "2022-12-30", "2023-01-02",
                       "2023-06-16", "2023-11-24", "2023-12-26", "2023-01-02", "2023-06-19"]
        pu_holiday = date_today in pu_holidays
        if not us_holiday and not pu_holiday:
          send_email(s,  f"{netid}@princeton.edu", subject="Jobs with zero GPU utilization", sender="cses@princeton.edu")
          send_email(s, "halverson@princeton.edu", subject="Jobs with zero GPU utilization", sender="cses@princeton.edu")
          usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M")
          if "GPU-Util"        in usr.columns: usr.drop(columns=["GPU-Util"],        inplace=True)
          if "GPU-Unused-Util" in usr.columns: usr.drop(columns=["GPU-Unused-Util"], inplace=True)
          if os.path.exists(vfile):
            curr = pd.read_csv(vfile)
            curr = pd.concat([curr, usr]).drop_duplicates()
            curr.to_csv(vfile, index=False, header=True)
          else:
            usr.to_csv(vfile, index=False, header=True)
  return em
