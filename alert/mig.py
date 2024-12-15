import pandas as pd
from base import Alert
from utils import add_dividers
from utils import send_email
from utils import MINUTES_PER_HOUR as mph
from efficiency import cpu_memory_usage
from efficiency import gpu_memory_usage_eff_tuples
from greeting import GreetingFactory
from email_translator import EmailTranslator


class MultiInstanceGPU(Alert):

    """Find jobs that could have used the MIG GPUs."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition == self.partition) &
                          (self.df.cores == self.num_cores_threshold) &
                          (self.df.gpus == 1) &
                          (self.df.admincomment != {}) &
                          (~self.df.user.isin(self.excluded_users)) &
                          (self.df.state != "OUT_OF_MEMORY") &
                          (self.df["elapsed-hours"] >= self.min_run_time / mph)].copy()
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
        renamings = {"elapsed-hours":"Hours", "jobid":"JobID", "user":"User"}
        self.df = self.df.rename(columns=renamings)
        self.df = self.df[["JobID", "User", "GPU-Util", "GPU-Mem-Used", "CPU-Mem-Used", "Hours"]]
        # where is groupby and then compare to abs threshold

    def send_emails_to_users(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.df.User.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.User == user].copy()
                usr["Hours"] = usr["Hours"].apply(lambda hrs: round(hrs, 1))
                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<DAYS>"] = str(self.days_between_emails)
                tags["<CLUSTER>"] = self.cluster
                tags["<PARTITION>"] = self.partition  # multiple partitions?
                tags["<TARGET>"] = self.gpu_util_target
                tags["<NUM-JOBS>"] = str(len(usr))
                indent = 2 * " "
                table = usr.to_string(index=False, justify="center").split("\n")
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                translator = EmailTranslator("email/mig.txt", tags)
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
        else:
            def jobid_list(series):
                ellipsis = "+" if len(series) > self.max_num_jobid_admin else ""
                return ",".join(series[:self.max_num_jobid_admin]) + ellipsis
            d = {"Hours":"sum", "User":"size", "JobID":jobid_list}
            self.admin = self.df.groupby("User").agg(d)
            self.admin["Hours"] = self.admin["Hours"].apply(lambda hrs: round(hrs, 1))
            renamings = {"User":"Jobs", "Hours":"GPU-Hours"}
            self.admin = self.admin.rename(columns=renamings)
            self.admin = self.admin.sort_values(by="GPU-Hours", ascending=False)
            self.admin.reset_index(drop=False, inplace=True)
            self.admin.index += 1
            self.admin["email90"] = self.admin["User"].apply(lambda user:
                                                   self.get_emails_sent_count(user,
                                                                              self.violation,
                                                                              days=90))
            return add_dividers(self.admin.to_string(index=keep_index, justify="center"), title)
