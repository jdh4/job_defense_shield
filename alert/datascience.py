import os
import textwrap
from datetime import datetime
import numpy as np
import pandas as pd
from efficiency import cpu_memory_usage
from utils import JOBSTATES
from utils import SECONDS_PER_HOUR
from utils import HOURS_PER_DAY
from utils import get_first_name
from utils import send_email
from pandas.tseries.holiday import USFederalHolidayCalendar

def large_memory_needed(usage, account, safety=1.0):
  """Boolean function that decides if the DS node was needed."""
  cascade_max_mem = safety * 190
  physics_max_mem = safety * 380
  if (usage >= cascade_max_mem and account != "physics") or (usage >= physics_max_mem and account == "physics"):
    return "Yes"
  else:
    return "No"

def datascience_node_violators(df, email, vpath):
  ds = df[(df.cluster == "della") &
          (df.partition == "datasci") &
          (df.state != "OUT_OF_MEMORY") &
          (df.admincomment != {}) &
          (df["elapsed-hours"] >= 1)].copy()

  ds["memory-tuple"] = ds.apply(lambda row: cpu_memory_usage(row["admincomment"], row["jobid"], row["cluster"]), axis="columns")
  ds["memory-used"]  = ds["memory-tuple"].apply(lambda x: x[0])
  ds["memory-alloc"] = ds["memory-tuple"].apply(lambda x: x[1])
  ds["Large-Memory-Needed?"] = ds.apply(lambda row: large_memory_needed(row["memory-used"], row["account"]), axis="columns")
  ds["within-safety"] = ds.apply(lambda row: True if large_memory_needed(row["memory-used"], row["account"], safety=0.8) == "Yes" else False, axis="columns")
  ds.state = ds.state.apply(lambda x: JOBSTATES[x])

  ### EMAIL
  if email:
    for netid in np.sort(ds.netid.unique()):
      usr = ds[ds.netid == netid].copy()
      total_jobs = usr.shape[0]
      bad_jobs           = usr[usr["Large-Memory-Needed?"] == "No"].shape[0]
      jobs_within_safety = usr[usr["within-safety"]].shape[0]
      is_physics = "physics" in usr.account.unique().tolist()
      small_physics = usr[usr["memory-alloc"] < 380].shape[0]
      if bad_jobs > 0:
        vfile = f"{vpath}/datascience/{netid}.email.csv"
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
            overall memory requirement of each job is less than {max_mem}. You should use the
            smallest value possible but include an extra 20% for safety.

            For more information on the large-memory nodes and allocating CPU memory:

               https://researchcomputing.princeton.edu/systems/della#large_memory
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
            each job is less than {max_mem}. You should use the smallest value possible but include
            an extra 20% for safety.

            For more information on the large-memory nodes and allocating CPU memory:

               https://researchcomputing.princeton.edu/systems/della#large_memory
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
            overall memory requirement of each job is less than {max_mem}. You should use the
            smallest value possible but include an extra 20% for safety.

            We understand that for some jobs it can be very difficult or impossible to estimate
            the memory requirements. For those jobs please disregard this email.

            For more information on the large-memory nodes and allocating CPU memory:

               https://researchcomputing.princeton.edu/systems/della#large_memory
               https://researchcomputing.princeton.edu/support/knowledge-base/memory
            """)

          # if a physics user does not specify -p physics and they request more than 194G then it will go to datascience
          if is_physics and small_physics:
            s += textwrap.dedent(f"""
            Your Slurm account is physics. This means that you have acccess to nodes with 380 GB
            of memory that are not available to other users. Please add the following line to
            your Slurm scripts:
            
               #SBATCH --partition=physics
            """)
        
          s += textwrap.dedent(f"""
          Add the following lines to your Slurm scripts to receive an email report with
          memory usage information after each job finishes:

             #SBATCH --mail-type=end
             #SBATCH --mail-user={netid}@princeton.edu

          One can also see memory usage information by using the following command:

             $ jobstats {usr.JobID.values[0]}
          
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
          if email and not us_holiday and not pu_holiday:
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

  cols = ["netid", "jobid", "state", "memory-used", "memory-alloc", "elapsed-hours", "cores", "Large-Memory-Needed?"]
  return ds[cols][ds["Large-Memory-Needed?"] == "No"].sort_values(by="memory-used", ascending=False)
