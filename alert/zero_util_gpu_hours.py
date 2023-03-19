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
      self.df["gpus-unused"] = self.df.admincomment.apply(num_gpus_with_zero_util)
      self.df = self.df[self.df["gpus-unused"] > 0]
      self.df["zero-util-hours"] = self.df["gpus-unused"] * self.df["elapsedraw"] / SECONDS_PER_HOUR
      self.df["GPU-Unused-Util"] = "0%"
      self.df = self.df[["jobid", "netid", "gpus", "gpus-unused", "GPU-Unused-Util", "zero-util-hours"]]
      # for each user sum the number of GPU-hours with zero GPU utilization
      self.gp = self.df.groupby("netid").agg({"zero-util-hours":np.sum, "netid":np.size})
      self.gp = self.gp.rename(columns={"netid":"jobs"})
      self.gp = self.gp.sort_values(by="zero-util-hours", ascending=False).reset_index(drop=False)
      self.gp = self.gp[self.gp["zero-util-hours"] >= 100]
      self.df["zero-util-hours"] = self.df["zero-util-hours"].apply(round)

  def send_emails_to_users(self):
      for user in self.gp.netid.unique():
          vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
          if self.has_sufficient_time_passed_since_last_email(vfile):
              usr = self.df[self.df.netid == user].copy()
              renamings = {"zero-util-hours":"Zero-Util-GPU-Hours",
                           "jobid":"JobID",
                           "netid":"NetID",
                           "gpus-unused":"GPUs-Unused",
                           "gpus":"GPUs"}
              usr = usr.rename(columns=renamings)
              zero_hours = round(self.gp[self.gp.netid == user]["zero-util-hours"].values[0])
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

  def generate_report_for_admins(self, title: str, keep_index: bool=True) -> str:
      self.gp["zero-util-hours"] = self.gp["zero-util-hours"].apply(round)
      self.gp = self.gp.rename(columns={"netid":"NetID", "jobs":"Jobs", "zero-util-hours":"Zero-Util-GPU-Hours"})
      self.gp.index += 1
      return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
