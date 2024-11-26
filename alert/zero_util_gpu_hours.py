from datetime import datetime
import textwrap
import pandas as pd
import utils
from base import Alert
from utils import SECONDS_PER_HOUR
from utils import SECONDS_PER_MINUTE
from utils import send_email_cses
from utils import add_dividers
from efficiency import num_gpus_with_zero_util
from greeting import Greeting


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
                          (self.df.admincomment != {}) &
                          (~self.df.netid.isin(self.excluded_users)) &
                          (self.df["elapsedraw"] >= self.min_run_time * SECONDS_PER_MINUTE)].copy()
        self.gp = pd.DataFrame({"NetID":[]})
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
            self.df["Zero-Util-GPU-Hours"] = self.df["GPUs-Unused"] * self.df["elapsedraw"] / SECONDS_PER_HOUR
            self.df["GPU-Unused-Util"] = "0%"
            cols = ["jobid",
                    "netid",
                    "gpus",
                    "GPUs-Unused",
                    "GPU-Unused-Util",
                    "Zero-Util-GPU-Hours"]
            self.df = self.df[cols]
            renamings = {"jobid":"JobID", "netid":"NetID", "gpus":"GPUs"}
            self.df = self.df.rename(columns=renamings)
            def jobid_list(series):
                ellipsis = "+" if len(series) > self.max_num_jobid else ""
                return ",".join(series[:self.max_num_jobid]) + ellipsis
            # for each user sum the number of GPU-hours with zero GPU utilization
            self.gp = self.df.groupby("NetID").agg({"Zero-Util-GPU-Hours":"sum",
                                                    "NetID":"size",
                                                    "JobID":jobid_list})
            self.gp = self.gp.rename(columns={"NetID":"Jobs"})
            self.gp.reset_index(drop=False, inplace=True)
            self.admin = self.gp[self.gp["Zero-Util-GPU-Hours"] >= self.gpu_hours_threshold_admin].copy()
            # apply a threshold to focus on the heaviest offenders
            self.gp = self.gp[self.gp["Zero-Util-GPU-Hours"] >= self.gpu_hours_threshold_user]
            self.df["Zero-Util-GPU-Hours"] = self.df["Zero-Util-GPU-Hours"].apply(round)

    def send_emails_to_users(self):
        for user in self.gp.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                zero_hours = round(self.gp[self.gp.NetID == user]["Zero-Util-GPU-Hours"].values[0])
                emails_sent = self.get_emails_sent_count(user, "zero_gpu_utilization")
                s = f"{Greeting(user).greeting()}"
                s += f"You have consumed {zero_hours} GPU-hours at 0% GPU utilization in the past {self.days_between_emails} days on\n"
                s += "Della. This is a waste of valuable resources.\n"
                s += textwrap.dedent(f"""
                See this webpage for three common reasons why a user may encounter 0% GPU
                utilization:

                  https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing

                Consider attending an in-person Research Computing help session for assistance:

                  https://researchcomputing.princeton.edu/support/help-sessions

                It is your responsibility to ensure that the GPUs and other resources are being
                used efficiently by your jobs. Please monitor your jobs using the "jobstats"
                command:
                """)
                s += "\n"
                s += f"  $ jobstats {usr.JobID.values[0]}"
                s += "\n\n"
                s += f"\nBelow are the jobs that ran in the past {self.days_between_emails} days with 0% GPU utilization:\n\n"
                s +=  "\n".join([2 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
                s +=  "\n\nPlease reply to this support ticket if you would like assistance in resolving"
                s +=  "\nthis issue."

                if emails_sent <= 1:
                    self.subject = "Underutilization of the GPUs on Della"
                else:
                    self.subject = "WARNING OF ACCOUNT SUSPENSION: Underutilization of the GPUs on Della"
                send_email_cses(s,      "cses@princeton.edu", subject=f"{self.subject}", sender="jdh4@princeton.edu")
                send_email_cses(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, start_date, keep_index: bool=False) -> str:
        if self.admin.empty:
            return ""
        else:
            self.admin = self.admin.sort_values(by="Zero-Util-GPU-Hours", ascending=False)
            self.admin["Zero-Util-GPU-Hours"] = self.admin["Zero-Util-GPU-Hours"].apply(round)
            self.admin = self.admin.rename(columns={"Zero-Util-GPU-Hours":"0%-GPU-Hours"})
            self.admin.reset_index(drop=True, inplace=True)
            self.admin.index += 1
            post  = f"                  Cluster: {self.cluster}\n" 
            post += f"               Partitions: {', '.join(self.partitions)}\n" 
            fmt = "%a %b %-d, %Y at %-I:%M %p"
            post += f"                    Start: {start_date.strftime(fmt)}\n" 
            post += f"                      End: {datetime.now().strftime(fmt)}\n" 
            post += f"             min_run_time: {self.min_run_time} minutes\n" 
            post += f"gpu_hours_threshold_admin: {self.gpu_hours_threshold_admin} GPU-hours\n" 
            post += f"            max_num_jobid: {self.max_num_jobid}\n"
            post += "* This report applies to completed jobs only"
            return add_dividers(self.admin.to_string(index=keep_index, justify="center"), title, post=post)
