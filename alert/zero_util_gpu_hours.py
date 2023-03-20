from base import Alert
from utils import SECONDS_PER_HOUR
from utils import get_first_name
from utils import send_email
from utils import add_dividers
from efficiency import num_gpus_with_zero_util
from efficiency import gpu_efficiency
import numpy as np


class ZeroUtilGPUHours(Alert):

  """Identify users with many GPU-hours at 0% GPU utilization."""

  def __init__(self, df, days_between_emails, violation, vpath, subject):
      super().__init__(df, days_between_emails, violation, vpath, subject)

  def _filter_and_add_new_fields(self):
      # filter the dataframe
      self.df = self.df[(self.df.cluster == "della") &
                        (self.df.partition == "gpu") &
                        (self.df.gpus > 0) &
                        (self.df.admincomment != {}) &
                        (self.df["elapsedraw"] >= SECONDS_PER_HOUR)].copy()
      # add new fields
      self.df["GPUs-Unused"] = self.df.admincomment.apply(num_gpus_with_zero_util)
      self.df = self.df[self.df["GPUs-Unused"] > 0]
      self.df["Zero-Util-GPU-Hours"] = self.df["GPUs-Unused"] * self.df["elapsedraw"] / SECONDS_PER_HOUR
      self.df["GPU-Unused-Util"] = "0%"
      self.df = self.df[["jobid", "netid", "gpus", "GPUs-Unused", "GPU-Unused-Util", "Zero-Util-GPU-Hours"]]
      renamings = {"jobid":"JobID", "netid":"NetID", "gpus":"GPUs"}
      self.df = self.df.rename(columns=renamings)
      # for each user sum the number of GPU-hours with zero GPU utilization
      self.gp = self.df.groupby("NetID").agg({"Zero-Util-GPU-Hours":np.sum, "NetID":np.size})
      self.df["Zero-Util-GPU-Hours"] = self.df["Zero-Util-GPU-Hours"].apply(round)

  def send_emails_to_users(self):
      for user in self.gp.netid.unique():
          vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
          if self.has_sufficient_time_passed_since_last_email(vfile):
              usr = self.df[self.df.NetID == user].copy()
              zero_hours = round(self.gp[self.gp.NetID == user]["Zero-Util-GPU-Hours"].values[0])
              s =  f"Requestor: {user}@princeton.edu\n\n"
              s += f"{get_first_name(user)},\n\n"
              s += f"You have consumed {zero_hours} GPU-hours at 0% GPU utilization in the past {self.days_between_emails} days on\n"
              s +=  "Della. This is a waste of valuable resources. Please monitor your jobs using the\n"
              s +=  "\"jobstats\" command or the web interface:\n\n"
              s +=  "  https://researchcomputing.princeton.edu/support/knowledge-base/job-stats\n\n"
              s +=  "Below are the jobs with 0% GPU utilization:\n\n"
              s +=  "\n".join([2 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
              s +=  "\n"
              s += textwrap.dedent(f"""
              Please investigate the reason for the GPUs not being used. For instance, is the
              code GPU-enabled? Can you use a MIG GPU instead of a full A100?

              For general information about GPU computing at Princeton:

                https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing

              Replying to this email will open a support ticket with CSES. Let us know if we
              can be of help.
              """)
              #send_email(s,      "cses@princeton.edu", subject=f"{self.subject}", sender="halverson@princeton.edu")
              send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
              print(s)

              # append the new violations to the log file
              #Alert.update_violation_log(usr, vfile)

  def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
      self.gp = self.gp.rename(columns={"NetID":"Jobs"})
      self.gp = self.gp.sort_values(by="Zero-Util-GPU-Hours", ascending=False)
      self.gp.reset_index(drop=False, inplace=True)
      self.gp = self.gp[self.gp["Zero-Util-GPU-Hours"] >= 100]
      self.gp["Zero-Util-GPU-Hours"] = self.gp["Zero-Util-GPU-Hours"].apply(round)
      self.gp.index += 1
      return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
