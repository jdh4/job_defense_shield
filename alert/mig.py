import textwrap
import pandas as pd
from base import Alert
from utils import add_dividers
from utils import get_first_name
from utils import send_email
from efficiency import cpu_memory_usage
from efficiency import gpu_memory_usage_eff_tuples
import numpy as np


class MultiInstanceGPU(Alert):

    """Find jobs that could have used the MIG GPUs."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition == self.partition) &
                          (self.df.cores == 1) &
                          (self.df.gpus == 1) &
                          (self.df.admincomment != {}) &
                          (~self.df.netid.isin(self.excluded_users)) &
                          (self.df.state != "OUT_OF_MEMORY") &
                          (self.df["elapsed-hours"] >= 1)].copy()
        # add new fields
        self.df["gpu-tuple"] = self.df.apply(lambda row:
                               gpu_memory_usage_eff_tuples(row["admincomment"],
                                                           row["jobid"],
                                                           row["cluster"]),
                                                           axis="columns")
        self.df["error_code"] = self.df["gpu-tuple"].apply(lambda x: x[1])
        self.df = self.df[self.df["error_code"] == 0]
        # next two lines are valid since only one GPU per job
        self.df["GPU-Mem-Used"] = self.df["gpu-tuple"].apply(lambda tpl: tpl[0][0][0])
        self.df["GPU-Util"]     = self.df["gpu-tuple"].apply(lambda tpl: tpl[0][0][2])
        # add CPU memory usage
        self.df["memory-tuple"] = self.df.apply(lambda row:
                                  cpu_memory_usage(row["admincomment"],
                                                   row["jobid"],
                                                   row["cluster"]),
                                                   axis="columns")
        cols = ["CPU-Mem-Used", "mem-alloc", "error_code"]
        self.df[cols] = pd.DataFrame(self.df["memory-tuple"].tolist(), index=self.df.index)
        self.df = self.df[self.df["error_code"] == 0]
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
        self.df["GPU-Util"]     = self.df["GPU-Util"].apply(lambda x: f"{round(x)}%" if x > 0.5 else f"{round(x, 1)}%")
        renamings = {"elapsed-hours":"Hours", "jobid":"JobID", "netid":"NetID"}
        self.df = self.df.rename(columns=renamings)
        self.df = self.df[["JobID", "NetID", "GPU-Util", "GPU-Mem-Used", "CPU-Mem-Used", "Hours"]]

    def send_emails_to_users(self):
        for user in self.df.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                usr["Hours"] = usr["Hours"].apply(lambda x: str(x).replace(".0", ""))
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

                If you are using Jupyter OnDemand then set the "Node type" to "mig" when
                creating the session.

                By running jobs on the MIG GPUs you will experience shorter queue times and
                you will help keep A100 GPUs free for jobs that need them. For more info:

                  https://researchcomputing.princeton.edu/systems/della#gpus

                As an alternative to MIG, you may consider trying to improve the GPU
                utilization of your code. A good target value is greater than 50%. Consider
                writing to the mailing list of the software that you are using or attend
                an in-person Research Computing help session:

                  https://researchcomputing.princeton.edu/support/help-sessions

                For general information about GPU computing at Princeton:

                  https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing

                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)
                send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email(s, "alerts-jobs-aaaalegbihhpknikkw2fkdx6gi@princetonrc.slack.com", subject=f"{self.subject}", sender="cses@princeton.edu")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)
   
    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            self.admin = self.df.groupby("NetID").agg({"Hours":np.sum, "NetID":np.size})
            self.admin = self.admin.rename(columns={"NetID":"Jobs", "Hours":"Full-A100-GPU-Hours"})
            self.admin = self.admin.sort_values(by="Full-A100-GPU-Hours", ascending=False)
            self.admin.reset_index(drop=False, inplace=True)
            self.admin.index += 1
            self.admin["email90"] = self.admin["NetID"].apply(lambda netid:
                                                   self.get_emails_sent_count(netid,
                                                                              self.violation,
                                                                              days=90))
            return add_dividers(self.admin.to_string(index=keep_index, justify="center"), title)
