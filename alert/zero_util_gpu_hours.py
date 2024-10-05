import textwrap
import pandas as pd

import utils
from base import Alert
from utils import SECONDS_PER_HOUR
from utils import get_first_name
from utils import send_email_cses
from utils import add_dividers
from efficiency import num_gpus_with_zero_util


class ZeroUtilGPUHours(Alert):

    """Identify users with many GPU-hours at 0% GPU utilization."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == "della") &
                          (self.df.partition == "gpu") &
                          (self.df.gpus > 0) &
                          (self.df.admincomment != {}) &
                          (self.df["elapsedraw"] >= SECONDS_PER_HOUR)].copy()
        self.gp = pd.DataFrame({"NetID":[]})
        self.admin = pd.DataFrame()
        x = utils.HOURS_PER_DAY
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
            self.df = self.df[["jobid", "netid", "gpus", "GPUs-Unused", "GPU-Unused-Util", "Zero-Util-GPU-Hours"]]
            renamings = {"jobid":"JobID", "netid":"NetID", "gpus":"GPUs"}
            self.df = self.df.rename(columns=renamings)
            # for each user sum the number of GPU-hours with zero GPU utilization
            self.gp = self.df.groupby("NetID").agg({"Zero-Util-GPU-Hours":"sum", "NetID":"size"})
            self.gp = self.gp.rename(columns={"NetID":"Jobs"})
            self.gp.reset_index(drop=False, inplace=True)
            self.admin = self.gp[self.gp["Zero-Util-GPU-Hours"] >= 25].copy()
            # apply a threshold to focus on the heaviest offenders
            self.gp = self.gp[self.gp["Zero-Util-GPU-Hours"] >= 100]
            self.df["Zero-Util-GPU-Hours"] = self.df["Zero-Util-GPU-Hours"].apply(round)

    def send_emails_to_users(self):
        for user in self.gp.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                zero_hours = round(self.gp[self.gp.NetID == user]["Zero-Util-GPU-Hours"].values[0])
                emails_sent = self.get_emails_sent_count(user, "zero_gpu_utilization")
                s =  f"Requestor: {user}@princeton.edu\n\n"
                s += f"{get_first_name(user, formal=True)},\n\n"
                if emails_sent == 0:
                    s += f"You have consumed {zero_hours} GPU-hours at 0% GPU utilization in the past {self.days_between_emails} days on\n"
                    s +=  "Della. This is a waste of valuable resources.\n"
                elif emails_sent == 1:
                    s += f"You have consumed {zero_hours} GPU-hours at 0% GPU utilization in the past {self.days_between_emails} days on\n"
                    s +=  "Della. Additionally, in the last 30 days you have been sent a warning email\n"
                    s +=  "with the subject \"Jobs with zero GPU utilization\".\n"
                else:
                    s += f"You have consumed {zero_hours} GPU-hours at 0% GPU utilization in the past {self.days_between_emails} days on\n"
                    s += f"Della. Additionally, in the last 30 days you have been sent {emails_sent} warning emails\n"
                    s +=  "with the subject \"Jobs with zero GPU utilization\".\n"
                if emails_sent <= 1:
                    s += textwrap.dedent(f"""
                    Be aware that your account can be suspended when you waste this amount of
                    GPU resources. Since this appears to be an isolated incident, no action
                    will be taken and your account will remain active.
                    """)
                else:
                    s += textwrap.dedent(f"""
                    At this time you need to stop underutilizing the GPUs or YOUR ACCOUNT WILL BE
                    SUSPENDED and your sponsor will be contacted. The GPUs are valuable resources
                    and they must be used efficiently.
                    """)
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

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.admin.empty:
            return ""
        else:
            self.admin = self.admin.sort_values(by="Zero-Util-GPU-Hours", ascending=False)
            self.admin["Zero-Util-GPU-Hours"] = self.admin["Zero-Util-GPU-Hours"].apply(round)
            self.admin.reset_index(drop=True, inplace=True)
            self.admin.index += 1
            return add_dividers(self.admin.to_string(index=keep_index, justify="center"), title)
