from datetime import datetime
import pandas as pd
from base import Alert
from utils import add_dividers
from utils import MINUTES_PER_HOUR as mph
from efficiency import num_gpus_with_zero_util
from greeting import GreetingFactory
from email_translator import EmailTranslator


class ZeroUtilGPUHours(Alert):

    """Identify users with many GPU-hours at 0% GPU utilization."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        self.excluded_users = []
        self.user_emails_bcc = []
        self.report_emails = []
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition.isin(self.partitions)) &
                          (self.df.gpus > 0) &
                          (~self.df.user.isin(self.excluded_users)) &
                          (self.df["elapsed-hours"] >= self.min_run_time / mph)].copy()
        if self.include_running_jobs:
            self.df.admincomment = Alert.get_admincomment_for_running_jobs(self)
        self.df = self.df[self.df.admincomment != {}]
        self.gp = pd.DataFrame({"User":[]})
        self.admin = pd.DataFrame()
        # add new fields
        if not self.df.empty:
            self.df["zero-tuple"] = self.df.apply(lambda row:
                                    num_gpus_with_zero_util(row["admincomment"],
                                                            row["jobid"],
                                                            row["cluster"]),
                                                            axis="columns")
            cols = ["GPUs-Unused", "error_code"]
            self.df[cols] = pd.DataFrame(self.df["zero-tuple"].tolist(), index=self.df.index)
            self.df = self.df[(self.df["error_code"] == 0) & (self.df["GPUs-Unused"] > 0)]
            self.df["Zero-Util-GPU-Hours"] = self.df["GPUs-Unused"] * self.df["elapsed-hours"]
            self.df["GPU-Unused-Util"] = "0%"
            cols = ["jobid",
                    "user",
                    "partition",
                    "gpus",
                    "GPUs-Unused",
                    "GPU-Unused-Util",
                    "Zero-Util-GPU-Hours"]
            self.df = self.df[cols]
            renamings = {"jobid":"JobID", "user":"User", "gpus":"GPUs"}
            self.df = self.df.rename(columns=renamings)
            def jobid_list(series):
                ellipsis = "+" if len(series) > self.max_num_jobid_admin else ""
                return ",".join(series[:self.max_num_jobid_admin]) + ellipsis
            # for each user sum the number of GPU-hours with zero GPU utilization
            self.gp = self.df.groupby("User").agg({"Zero-Util-GPU-Hours":"sum",
                                                   "User":"size",
                                                   "JobID":jobid_list})
            self.gp = self.gp.rename(columns={"User":"Jobs"})
            self.gp.reset_index(drop=False, inplace=True)
            self.admin = self.gp[self.gp["Zero-Util-GPU-Hours"] >= self.gpu_hours_threshold_admin].copy()
            # apply a threshold to focus on the heaviest offenders
            self.gp = self.gp[self.gp["Zero-Util-GPU-Hours"] >= self.gpu_hours_threshold_user]
            self.df["Zero-Util-GPU-Hours"] = self.df["Zero-Util-GPU-Hours"].apply(round)

    def create_emails(self, method):
        # self.gp is not needed here (could use df)
        g = GreetingFactory().create_greeting(method)
        for user in self.gp.User.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.User == user].copy()
                zero_hours = round(self.gp[self.gp.User == user]["Zero-Util-GPU-Hours"].values[0])
                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<GPU-HOURS>"] = str(zero_hours)
                tags["<DAYS>"] = str(self.days_between_emails)
                tags["<CLUSTER>"] = self.cluster
                tags["<PARTITIONS>"] = ",".join(sorted(set(usr.partition)))
                usr.drop(columns=["partition"], inplace=True)
                tags["<NUM-JOBS>"] = str(len(usr))
                indent = 4 * " "
                table = usr.to_string(index=False, justify="center").split("\n")
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                translator = EmailTranslator(self.email_file, tags)
                email = translator.replace_tags()
                self.emails.append((user, email, usr))

    def generate_report_for_admins(self,
                                   title: str,
                                   start_date: datetime,
                                   end_date: datetime,
                                   keep_index: bool=False) -> str:
        if self.admin.empty:
            return ""
        else:
            self.admin["emails"] = self.admin.User.apply(lambda user:
                                        self.get_emails_sent_count(user, self.violation))
            self.admin.emails = self.format_email_counts(self.admin.emails)
            self.admin = self.admin.sort_values(by="Zero-Util-GPU-Hours", ascending=False)
            self.admin["Zero-Util-GPU-Hours"] = self.admin["Zero-Util-GPU-Hours"].apply(round)
            self.admin = self.admin.rename(columns={"Zero-Util-GPU-Hours":"GPU-Hours-At-0%"})
            self.admin.reset_index(drop=True, inplace=True)
            self.admin.index += 1
            post  = f"   Cluster: {self.cluster}\n" 
            post += f"Partitions: {', '.join(self.partitions)}\n" 
            fmt = "%a %b %d, %Y at %I:%M %p"
            post += f"     Start: {start_date.strftime(fmt)}\n" 
            post += f"       End: {end_date.strftime(fmt)}\n" 
            return add_dividers(self.admin.to_string(index=keep_index, justify="center"), title, post=post)
