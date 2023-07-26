import os
import textwrap
from datetime import datetime
import numpy as np
import pandas as pd
from utils import get_first_name
from utils import SECONDS_PER_HOUR
from utils import HOURS_PER_DAY
from utils import send_email
from efficiency import cpu_efficiency
from efficiency import gpu_efficiency
from pandas.tseries.holiday import USFederalHolidayCalendar

def xpu_efficiencies_of_heaviest_users(df, cluster, cluster_name, partitions, xpu, email, vpath, num_top_users):
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
  def uniq_list(series):
    return ",".join(sorted(set(series)))
  d = {"netid":np.size, f"{xpu}-seconds-used":np.sum, f"{xpu}-seconds-total":np.sum, "partition":uniq_list, \
       "proportion(%)":"first", f"{xpu}-seconds-all":"first", "cores":np.mean, "interactive":np.sum}
  ce = ce.groupby("netid").agg(d).rename(columns={"netid":"jobs"})
  ce = ce.sort_values(by=f"{xpu}-seconds-total", ascending=False).reset_index(drop=False)
  ce = ce.head(num_top_users)
  ce["eff(%)"] = 100.0 * ce[f"{xpu}-seconds-used"] / ce[f"{xpu}-seconds-total"]
  if ce.empty: return pd.DataFrame()  # prevents next line from failing
  ce[f"{xpu}-hours"] = ce.apply(lambda row: round(row[f"{xpu}-seconds-total"] / SECONDS_PER_HOUR), axis="columns")
  ce = ce[ce[f"{xpu}-seconds-all"] > 0]  # prevents next line from failing if cpu-only users in num_top_users e.g., traverse
  ce["coverage"] = ce.apply(lambda row: round(row[f"{xpu}-seconds-total"] / row[f"{xpu}-seconds-all"], 2), axis="columns")
  ce["eff(%)"] = ce["eff(%)"].apply(lambda x: round(x))
  ce["cores"] = ce["cores"].apply(lambda x: round(x, 1))
  ce.index += 1
  eff_thres = 60 if xpu == "cpu" else 15
  filters = (ce["eff(%)"] <= eff_thres) & (ce["proportion(%)"] >= 2)
  de = ce[["netid", "partition", f"{xpu}-hours", "proportion(%)", "eff(%)", "jobs", "interactive", "cores", "coverage"]].copy()
  ce = ce[["netid", "partition", f"{xpu}-hours", "proportion(%)", "eff(%)", "jobs", "interactive", "cores", "coverage"]][filters]

  if email:
    rank_text = {1:"the most", 2:"the 2nd most", 3:"the 3rd most"}
    for netid in ce.netid:
      vfile = f"{vpath}/low_xpu_efficiency/{netid}.email.csv"
      last_write_date = datetime(1970, 1, 1)
      if os.path.exists(vfile):
        # instead of next line, could read datetime from last line and use that
        last_write_date = datetime.fromtimestamp(os.path.getmtime(vfile))
      s = f"{get_first_name(netid)},\n\n"
      if (datetime.now().timestamp() - last_write_date.timestamp() >= 8 * HOURS_PER_DAY * SECONDS_PER_HOUR):
        usr = ce[ce.netid == netid].copy()
        rank = ce.index[ce.netid == netid].tolist()[0]
        usr[f"{xpu.upper()}-rank"] = f"{rank}/{pr.shape[0]}"
        usr["eff(%)"] = usr["eff(%)"].apply(lambda x: f"{x}%")
        if xpu == "cpu":
          cols = ["netid", "partition", "jobs", f"{xpu}-hours", f"{xpu.upper()}-rank", "eff(%)", "cores"]
        if xpu == "gpu":
          cols = ["netid", "partition", "jobs", f"{xpu}-hours", f"{xpu.upper()}-rank", "eff(%)"]
        renamings = {"eff(%)":"Efficiency", "jobid":"JobID", "netid":"NetID", "proportion(%)":"Proportion(%)", \
                     "cpu-hours":"CPU-hours", "gpu-hours":"GPU-hours", "jobs":"Jobs", "cores":"AvgCores", \
                     "partition":"Partition(s)"}
        usr = usr[cols].rename(columns=renamings)
        usage = usr["CPU-hours"] if xpu == "cpu" else usr["GPU-hours"]
        usr["AvgCores"] = usr["AvgCores"].apply(lambda x: str(x).replace(".0", ""))
        usage = usage.values[0]
        myrank = f"the {rank}th most" if rank > 3 else rank_text[rank]
        s +=f"Over the last 7 days you have used {myrank} {xpu.upper()}-hours on {cluster_name} but\n"
        s +=f"your mean {xpu.upper()} efficiency is only {usr['Efficiency'].values[0]}:\n\n"
        s += "\n".join([5 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
        s += "\n"
        if xpu == "cpu":
          s += textwrap.dedent(f"""
          A good target value for CPU-Util is 90% and above. Please investigate the reason(s)
          for the low efficiency. Common reasons for low {xpu.upper()} efficiency include:

            1. Running a serial code using multiple CPU-cores. Make sure that your code is
               written to run in parallel before using multiple CPU-cores. Learn more:
               https://researchcomputing.princeton.edu/support/knowledge-base/parallel-code

            2. Using too many CPU-cores for parallel jobs. You can find the optimal number
               of CPU-cores by performing a scaling analysis:
               https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis

            3. Writing job output to the /tigress or /projects storage systems. Actively
               running jobs should be writing output files to /scratch/gpfs/{netid} which is
               a much faster filesystem. For more information:
               https://researchcomputing.princeton.edu/support/knowledge-base/data-storage

            4. Using the MPICH library instead of an MPI library that was built for our
               clusters. Some software installed using \"conda\" is built against an MPI
               library that is not optimized for our systems. Run \"conda list\" after
               activating the environment and look for \"mpich\" to see if you are using
               this library.

            5. Using \"mpirun\" instead of \"srun\" for parallel codes. Please use \"srun\".
               For more information on Slurm:
               https://researchcomputing.princeton.edu/support/knowledge-base/slurm
          """)
        elif xpu == "gpu":
          s += textwrap.dedent(f"""
          A good target value for GPU-Util is 50% and above. Please investigate the reason(s)
          for the low efficiency. Common reasons for low {xpu.upper()} efficiency include:

            1. Misconfigured application scripts. Be sure to read the documentation of the
               software to make sure that you are using it properly. This includes creating
               the appropriate software environment. For a general overview of GPU computing:
               https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing

            2. Using an A100 GPU when a MIG GPU would be sufficient. Some codes do not have
               enough work to keep an A100 GPU busy. If you encounter this on the Della
               cluster then consider using a MIG GPU:
               https://researchcomputing.princeton.edu/systems/della#gpus

            3. Training deep learning models using only a single CPU-core. Codes such as
               PyTorch and TensorFlow show performance benefits when multiple CPU-cores are
               used for data loading. For PyTorch see:
               https://researchcomputing.princeton.edu/support/knowledge-base/pytorch#multi

            4. Using too many GPUs for a job. You can find the optimal number of GPUs and
               CPU-cores by performing a scaling analysis:
               https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis

            5. Writing job output to the /tigress or /projects storage systems. Actively
               running jobs should be writing output files to /scratch/gpfs/{netid} which is
               a much faster filesystem. For more information:
               https://researchcomputing.princeton.edu/support/knowledge-base/data-storage
          """)
        s += textwrap.dedent(f"""
        Consult the documentation or write to the mailing list of the software that you
        are using for additional reasons for low {xpu.upper()} efficiency and for potential
        solutions. You may also consider attending an in-person Research Computing help
        session:

             https://researchcomputing.princeton.edu/support/help-sessions
        """)
        s += textwrap.dedent(f"""
        Add the following lines to your Slurm scripts to receive an email report with {xpu.upper()}
        efficiency information after each job finishes:

             #SBATCH --mail-type=end
             #SBATCH --mail-user={netid}@princeton.edu
        
        You can check the efficiency of completed and actively running jobs by using the
        \"jobstats\" command:

             https://researchcomputing.princeton.edu/support/knowledge-base/job-stats

        Replying to this automated email will open a support ticket with Research
        Computing. Let us know if we can be of help.
        """)
        send_email(s, "halverson@princeton.edu", subject=f"Low {xpu.upper()} efficiency on {cluster_name}", sender="cses@princeton.edu")
        send_email(s,  f"{netid}@princeton.edu", subject=f"Low {xpu.upper()} efficiency on {cluster_name}", sender="cses@princeton.edu")
        usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M")
        if os.path.exists(vfile):
          curr = pd.read_csv(vfile)
          curr = pd.concat([curr, usr]).drop_duplicates()
          curr.to_csv(vfile, index=False, header=True)
        else:
          usr.to_csv(vfile, index=False, header=True)
  return de
