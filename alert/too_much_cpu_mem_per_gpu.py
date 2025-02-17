import pandas as pd
from base import Alert
from utils import send_email
from utils import add_dividers
from utils import MINUTES_PER_HOUR as mph
from efficiency import cpu_memory_usage
from greeting import GreetingFactory
from email_translator import EmailTranslator


class TooMuchCpuMemPerGpu(Alert):

    """Find jobs that allocating too much CPU memory per GPU."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition.isin(self.partitions)) &
                          (self.df.gpus > 0) &
                          (self.df.admincomment != {}) &
                          (self.df.state != "OUT_OF_MEMORY") &
                          (~self.df.user.isin(self.excluded_users)) &
                          (self.df["elapsed-hours"] >= self.min_run_time / mph)].copy()
        self.df.rename(columns={"user":"User"}, inplace=True)
        # add new fields
        if not self.df.empty:
            self.df["memory-tuple"] = self.df.apply(lambda row:
                                           cpu_memory_usage(row["admincomment"],
                                                            row["jobid"],
                                                            row["cluster"]),
                                                            axis="columns")
            cols = ["CPU-Mem-Used", "mem-alloc", "error_code"]
            self.df[cols] = pd.DataFrame(self.df["memory-tuple"].tolist(), index=self.df.index)
            self.df = self.df[self.df["error_code"] == 0]
            self.df["CPU-Mem-per-GPU"] = self.df["mem-alloc"] / self.df.gpus
            self.df = self.df[self.df["CPU-Mem-per-GPU"] > self.cpu_mem_per_gpu_limit]
            self.df["Mem-Eff"] = self.df["CPU-Mem-Used"] / self.df["mem-alloc"]
            # should user be ignored if even just one of their jobs is efficient?
            self.df = self.df[self.df["Mem-Eff"] < self.mem_eff_thres]
            self.df["CPU-Mem-per-GPU"] = self.df["CPU-Mem-per-GPU"].apply(lambda x:
                                              str(round(x, 1)).replace(".0", "") + " GB")
            self.df["CPU-Mem-per-GPU-Target"] = f"{self.cpu_mem_per_gpu_target} GB"
            self.df["mem-alloc"] = self.df["mem-alloc"].apply(round)
            cols = ["jobid",
                    "User",
                    "elapsed-hours",
                    "Mem-Eff",
                    "mem-alloc",
                    "gpus",
                    "CPU-Mem-per-GPU",
                    "CPU-Mem-per-GPU-Target",
                    "CPU-Mem-Used"]
            self.df = self.df[cols]
            renamings = {"jobid":"JobID",
                         "mem-alloc":"CPU-Mem",
                         "gpus":"GPUs",
                         "elapsed-hours":"Hours"}
            self.df = self.df.rename(columns=renamings)
            self.df["Hours"] = self.df["Hours"].apply(lambda x: str(round(x, 1))
                                                      if x < 5 else str(round(x)))
            self.df = self.df.sort_values("User", ascending=True)

    def send_emails_to_users(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.df.User.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.User == user].copy()
                # use the rounded values since those will appear in the email
                usr["Mem-Eff"] = usr["Mem-Eff"].apply(lambda x: round(100 * x))
                usr["mem-per-gpu-safety"] = 1.2 * (usr["Mem-Eff"] / 100) * usr["CPU-Mem"] / usr.GPUs
                mem_per_gpu = max(8, round(usr["mem-per-gpu-safety"].max()))
                usr.drop(columns=["User", "mem-per-gpu-safety", "CPU-Mem-Used"], inplace=True)
                usr["Mem-Eff"] = usr["Mem-Eff"].apply(lambda x: f"{x}%")
                usr["CPU-Mem"] = usr["CPU-Mem"].apply(lambda x: f"{x} GB")
                indent = 3 * " "
                table = usr.to_string(index=False, justify="center").split("\n")
                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<CLUSTER>"] = self.cluster
                tags["<NAME>"] = self.cluster_name
                tags["<TARGET>"] = str(self.cpu_mem_per_gpu_target)
                tags["<CORES>"] = str(self.cores_per_node)
                tags["<MEMORY>"] = str(self.cpu_mem_per_node)
                tags["<SLURM>"] = str(mem_per_gpu)
                tags["<GPUS>"] = str(self.gpus_per_node)
                tags["<DAYS>"] = str(self.days_between_emails)
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                tags["<PARTITIONS>"] = ",".join(self.partitions)
                translator = EmailTranslator(self.email_file, tags)
                s = translator.replace_tags()

                send_email(s, f"{user}@princeton.edu", subject=f"{self.subject}")
                for email in self.admin_emails:
                    send_email(s, f"{email}", subject=f"{self.subject}")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
