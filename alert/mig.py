import textwrap
from base import Alert
from utils import add_dividers
from utils import get_first_name
from utils import send_email
from efficiency import cpu_memory_usage
from efficiency import gpu_memory_usage_eff_tuples
import numpy as np


class MultiInstanceGPU(Alert):

  """Find jobs that could have used the MIG GPUs."""

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
      # next two lines are valid since only one GPU per job
      self.df["GPU-Mem-Used"]  = self.df["gpu-memused-memalloc-eff"].apply(lambda x: x[0][0])
      self.df["GPU-Util"]      = self.df["gpu-memused-memalloc-eff"].apply(lambda x: x[0][2])
      # add CPU memory usage
      self.df["cpu-memused-memalloc"] = self.df.apply(lambda row:
                                        cpu_memory_usage(row["admincomment"],
                                                         row["jobid"],
                                                         row["cluster"]),
                                                         axis="columns")
      self.df["CPU-Mem-Used"] = self.df["cpu-memused-memalloc"].apply(lambda x: x[0])
      # find jobs that could have used mig
      gpu_eff_threshold = 15 # percent
      gpu_mem_threshold = 10 # GB
      cpu_mem_threshold = 32 # GB
      self.df = self.df[(self.df["GPU-Util"] <= gpu_eff_threshold) &
                        (self.df["GPU-Util"] != 0) &
                        (self.df["GPU-Mem-Used"] < gpu_mem_threshold) &
                        (self.df["CPU-Mem-Used"] < cpu_mem_threshold)]
      self.df["CPU-Mem-Used"] = self.df["CPU-Mem-Used"].apply(lambda x: f"{round(x)} GB")
      self.df["GPU-Mem-Used"] = self.df["GPU-Mem-Used"].apply(lambda x: f"{round(x)} GB")
      self.df["GPU-Util"]     = self.df["GPU-Util"].apply(lambda x: f"{round(x)}%")
      renamings = {"elapsed-hours":"Hours", "jobid":"JobID", "netid":"NetID"}
      self.df = self.df.rename(columns=renamings)
      self.df = self.df[["JobID", "NetID", "GPU-Util", "GPU-Mem-Used", "CPU-Mem-Used", "Hours"]]

  def send_emails_to_users(self):
      for user in self.df.NetID.unique():
          vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
          if self.has_sufficient_time_passed_since_last_email(vfile):
              usr = self.df[self.df.NetID == user].copy()
              s =  f"{get_first_name(user)},\n\n"
              s += f"Below are jobs that ran on an A100 GPU on Della in the past {self.days_between_emails} days:"
              s +=  "\n\n"
              s +=  "\n".join([2 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
              s +=  "\n"
              s += textwrap.dedent(f"""
              The jobs above have a low GPU utilization and they use less than 10 GB of GPU
              memory and less than 32 GB of CPU memory. Such jobs could be run on the MIG
              GPUs. A MIG GPU has 1/7th the performance and memory of an A100. To run on a
              MIG GPU, add the "partition" directive to your Slurm script:

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

              Add the following lines to your Slurm scripts to receive an email report with
              GPU utilization information after each job finishes:

                #SBATCH --mail-type=end
                #SBATCH --mail-user={user}@princeton.edu

              Replying to this email will open a support ticket with CSES. Let us know if we
              can be of help.
              """)
              send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
              send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
              print(s)

              # append the new violations to the log file
              Alert.update_violation_log(usr, vfile)
 
  def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
      if self.df.empty:
          return ""
      else:
          self.gp = self.df.groupby("NetID").agg({"Hours":np.sum, "NetID":np.size})
          self.gp = self.gp.rename(columns={"NetID":"Jobs", "Hours":"Full-A100-GPU-Hours"})
          self.gp = self.gp.sort_values(by="Full-A100-GPU-Hours", ascending=False)
          self.gp.reset_index(drop=False, inplace=True)
          self.gp.index += 1
          return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
