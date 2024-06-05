import textwrap
from base import Alert
from efficiency import gpu_efficiency
from utils import get_first_name
from utils import send_email
from utils import add_dividers


class TooManyCoresPerGpu(Alert):

    """Find jobs that using too many CPU-cores per GPU."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          self.df.partition.isin(self.partitions) &
                          (self.df.gpus > 0) &
                          (self.df.cores > self.cores_per_gpu_thres * self.df.gpus) &
                          (self.df["elapsed-hours"] >= 1)].copy()
        self.df.rename(columns={"netid":"NetID"}, inplace=True)
        # add new fields
        if not self.df.empty:
            self.df["Cores-per-GPU"] = self.df.cores / self.df.gpus
            self.df["Cores-per-GPU"] = self.df["Cores-per-GPU"].apply(lambda x:
                                                                      str(round(x, 1)).replace(".0", ""))
            self.df["Cores-per-GPU-Target"] = self.cores_per_gpu_target
            cols = ["jobid",
                    "NetID",
                    "elapsed-hours",
                    "cores",
                    "gpus",
                    "Cores-per-GPU",
                    "Cores-per-GPU-Target"]
            self.df = self.df[cols]
            renamings = {"jobid":"JobID",
                         "cores":"Cores",
                         "gpus":"GPUs",
                         "elapsed-hours":"Hours"}
            self.df = self.df.rename(columns=renamings)

    def send_emails_to_users(self):
        for user in self.df.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                edays = self.days_between_emails
                s =  f"{get_first_name(user)},\n\n"
                s += f"Below are jobs that ran on Della in the past {edays} days that might be using\n"
                s +=  "more CPU-cores per GPU than necessary:\n\n"
                s +=  "\n".join([4 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
                s += "\n"
                s += textwrap.dedent(f"""
                Each PLI node on Della has 96 CPU-cores and 8 GPUs. If possible please try
                to use only up to 12 CPU-cores per GPU. This will prevent the situation
                where there are free GPUs but not enough CPU-cores to accept new jobs on a
                given node. For instance, three jobs that each allocate 32 CPU-cores and 1
                GPU will cause the remaining 5 GPUs on the node to be unavailable.

                For more information about the PLI nodes on Della:

                    https://researchcomputing.princeton.edu/systems/della#pli

                Consider attending an in-person Research Computing help session for assistance:

                    https://researchcomputing.princeton.edu/support/help-sessions

                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)
                #send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                for email in self.admin_emails:
                    send_email(s, f"{email}", subject=f"{self.subject}", sender="cses@princeton.edu")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
