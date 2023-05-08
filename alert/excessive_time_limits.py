import textwrap
from base import Alert
from utils import get_first_name
from utils import send_email
from utils import add_dividers
import numpy as np


class ExcessiveTimeLimits(Alert):

    """Over-allocating run time."""

    def __init__(self, df, days_between_emails, violation, vpath, subject):
        super().__init__(df, days_between_emails, violation, vpath, subject)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == "della") &
                          (self.df.partition == "cpu") &
                          (self.df.state == "COMPLETED") &
                          (self.df["elapsed-hours"] >= 1)].copy()
        # add new fields
        xpu = "cpu"
        if not self.df.empty:
            self.df["ratio"] = 100 * self.df[f"{xpu}-hours"] / self.df[f"{xpu}-alloc-hours"]
            d = {f"{xpu}-waste-hours":np.sum,
                 f"{xpu}-alloc-hours":np.sum,
                 f"{xpu}-hours":np.sum,
                 "netid":np.size,
                 "partition":lambda series: ",".join(sorted(set(series))),
                 "ratio":"median"}
            self.gp = self.df.groupby("netid").agg(d).rename(columns={"netid":"jobs", "ratio":"median(%)"})
            self.gp = self.gp.sort_values(by=f"{xpu}-hours", ascending=False).reset_index(drop=False)
            self.gp["rank"] = self.gp.index + 1
            self.gp = self.gp.sort_values(by=f"{xpu}-waste-hours", ascending=False).reset_index(drop=False)
            self.gp = self.gp.head(5)
            self.gp.index += 1
            self.gp[f"{xpu}-hours"] = self.gp[f"{xpu}-hours"].apply(round).astype("int64")
            self.gp["mean(%)"] = 100 * self.gp[f"{xpu}-hours"] / self.gp[f"{xpu}-alloc-hours"]
            self.gp["mean(%)"] = self.gp["mean(%)"].apply(round).astype("int64")
            self.gp["median(%)"] = self.gp["median(%)"].apply(round).astype("int64")
            self.gp = self.gp[(self.gp[f"{xpu}-waste-hours"] > 10000) &
                              (self.gp["mean(%)"] < 20) &
                              (self.gp["median(%)"] < 20) &
                              (self.gp["rank"] < 10)]
            cols = ["netid",
                    f"{xpu}-waste-hours",
                    f"{xpu}-hours",
                    f"{xpu}-alloc-hours",
                    "mean(%)",
                    "median(%)",
                    "rank",
                    "jobs",
                    "partition"]
            self.gp = self.gp[cols]
            renamings = {"netid":"NetID",
                         f"{xpu}-waste-hours":"CPU-Hours-Unused"}
            self.gp = self.gp.rename(columns=renamings)

    def send_emails_to_users(self):
        for user in self.gp.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.gp[self.gp.NetID == user].copy()
                edays = self.days_between_emails
                s =  f"{get_first_name(user)},\n\n"
                s += f"Below are jobs that ran on Della in the past {edays} days that allocated too much time\n"
                s +=  "\n".join([4 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
                s += "\n"
                s += textwrap.dedent(f"""
                Please lower the value of the --time Slurm directive.

                    https://researchcomputing.princeton.edu/support/knowledge-base/slurm

                Replying to this email will open a support ticket with CSES. Let us know if we
                can be of help.
                """)
                #send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        """Rename some of the columns."""
        if self.gp.empty:
            return ""
        else:
            xpu = "cpu"
            renamings = {f"{xpu}-hours":"used",
                         f"{xpu}-alloc-hours":"total"}
            self.gp = self.gp.rename(columns=renamings)
            return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
