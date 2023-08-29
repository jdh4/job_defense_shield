import textwrap
import pandas as pd
from base import Alert
from utils import SECONDS_PER_HOUR
from utils import get_first_name
from utils import send_email_cses
from utils import add_dividers
from efficiency import num_gpus_with_zero_util


class ZeroUtilGPUHours(Alert):

    """Identify users with many GPU-hours at 0% GPU utilization."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == "della") &
                          (self.df.partition == "gpu") &
                          (self.df.gpus > 0) &
                          (self.df.admincomment != {}) &
                          (self.df["elapsedraw"] >= SECONDS_PER_HOUR)].copy()
        self.gp = pd.DataFrame({"NetID":[]})
        # add new fields
        if not self.df.empty:
            self.df["GPUs-Unused"] = self.df.admincomment.apply(num_gpus_with_zero_util)
            self.df = self.df[self.df["GPUs-Unused"] > 0]
            self.df["Zero-Util-GPU-Hours"] = self.df["GPUs-Unused"] * self.df["elapsedraw"] / SECONDS_PER_HOUR
            self.df["GPU-Unused-Util"] = "0%"
            self.df = self.df[["jobid", "netid", "gpus", "GPUs-Unused", "GPU-Unused-Util", "Zero-Util-GPU-Hours"]]
            renamings = {"jobid":"JobID", "netid":"NetID", "gpus":"GPUs"}
            self.df = self.df.rename(columns=renamings)
            # for each user sum the number of GPU-hours with zero GPU utilization
            self.gp = self.df.groupby("NetID").agg({"Zero-Util-GPU-Hours":"sum", "NetID":"size"})
            self.gp = self.gp.rename(columns={"NetID":"Jobs"})
            self.gp.reset_index(drop=False, inplace=True)
            self.rw = self.gp.copy()
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
                s += f"{get_first_name(user)},\n\n"
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
                Please reply to this email if you would like assistance in resolving this issue.

                Below are three common reasons why a user may encounter 0% GPU utilization:

                  1. Is your code GPU-enabled? Only codes that have been explicitly written
                     to use GPUs can take advantage of them. Please consult the documentation
                     for your software. If your code is not GPU-enabled then please remove the
                     --gres Slurm directive when submitting jobs. For more information:

                       https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing

                  2. Make sure your software environment is properly configured. In some cases
                     certain libraries must be available for your code to run on GPUs. The
                     solution can be to load an environment module or to install a specific
                     software dependency. If your code uses CUDA then the CUDA Toolkit 11 or
                     higher should be used on Della. Please check your software environment
                     against the installation directions of your code.

                  3. Please do not create "salloc" sessions for long periods of time. For
                     example, allocating a GPU for 24 hours is wasteful unless you plan to work
                     intensively during the entire period. For interactive work, please
                     consider using the MIG GPUs:

                       https://researchcomputing.princeton.edu/systems/della#mig

                Consider attending an in-person Research Computing help session for assistance:

                  https://researchcomputing.princeton.edu/support/help-sessions

                It is your responsibility to ensure that the GPUs and other resources are being
                used efficiently by your jobs. Please monitor your jobs using the "jobstats"
                command and the web interface:

                  https://researchcomputing.princeton.edu/support/knowledge-base/job-stats
                """)
                s += f"\nBelow are the jobs that ran in the past {self.days_between_emails} days with 0% GPU utilization:\n\n"
                s +=  "\n".join([2 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
                s +=  "\n\nPlease reply to this support ticket if you need assistance."

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
        if self.rw.empty:
            return ""
        else:
            self.rw = self.rw[self.rw["Zero-Util-GPU-Hours"] >= 25]
            self.rw = self.rw.sort_values(by="Zero-Util-GPU-Hours", ascending=False)
            self.rw["Zero-Util-GPU-Hours"] = self.rw["Zero-Util-GPU-Hours"].apply(round)
            self.rw.reset_index(drop=True, inplace=True)
            self.rw.index += 1
            return add_dividers(self.rw.to_string(index=keep_index, justify="center"), title)
