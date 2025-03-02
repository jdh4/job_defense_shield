from datetime import datetime
import pandas as pd
from base import Alert
from utils import add_dividers
from utils import MINUTES_PER_HOUR as mph
from efficiency import num_gpus_with_zero_util
from greeting import GreetingFactory
from email_translator import EmailTranslator


class ZeroUtilGPUHours(Alert):

    """Identify users with many GPU-hours at 0% utilization."""

    def __init__(self, df, days_between_emails, violation, vpath, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, **kwargs)

    def _add_required_fields(self):
        if not hasattr(self, "email_subject"):
            self.email_subject = "GPU-hours at 0% Utilization"
        if not hasattr(self, "report_title"):
            self.report_title = "GPU-hours at 0% Utilization"
        if not hasattr(self, "max_num_jobid_admin"):
            self.max_num_jobid_admin = 4

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

    def generate_report_for_admins(self, keep_index: bool=False) -> str:
        if self.admin.empty:
            column_names = ["User",
                            "GPU-Hours-At-0%",
                            "Jobs",
                            "JobID",
                            "Emails"]
            self.admin = pd.DataFrame(columns=column_names)
            return add_dividers(self.create_empty_report(self.admin), self.report_title)
        self.admin["Emails"] = self.admin.User.apply(lambda user:
                                    self.get_emails_sent_count(user, self.violation))
        self.admin.Emails = self.format_email_counts(self.admin.Emails)
        self.admin = self.admin.sort_values(by="Zero-Util-GPU-Hours", ascending=False)
        self.admin["Zero-Util-GPU-Hours"] = self.admin["Zero-Util-GPU-Hours"].apply(round)
        self.admin = self.admin.rename(columns={"Zero-Util-GPU-Hours":"GPU-Hours-At-0%"})
        self.admin.reset_index(drop=True, inplace=True)
        self.admin.index += 1
        report_str = self.admin.to_string(index=keep_index, justify="center")
        return add_dividers(report_str, self.report_title)
