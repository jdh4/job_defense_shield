import textwrap
from base import Alert
from utils import SECONDS_PER_MINUTE
from utils import seconds_to_slurm_time_format
from utils import get_first_name
from utils import send_email
from utils import add_dividers
import numpy as np


class ExcessiveTimeLimits(Alert):

    """Over-allocating run time."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == "della") &
                          (self.df.partition == "cpu") &
                          (self.df.state == "COMPLETED") &
                          (self.df["elapsed-hours"] >= 1)].copy()
        # add new fields
        if not self.df.empty:
            xpu = "cpu"
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
                         f"{xpu}-waste-hours":f"{xpu.upper()}-Hours-Unused"}
            self.gp = self.gp.rename(columns=renamings)

    def send_emails_to_users(self):
        for user in self.gp.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.gp[self.gp.NetID == user].copy()
                jobs = self.df[self.df.netid == user].copy()
                xpu = "cpu"
                num_disp = 10
                total_jobs = jobs.shape[0]
                case = f"{num_disp} of your {total_jobs} jobs" if total_jobs > num_disp else "your jobs"
                jobs = jobs.sort_values(by=f"{xpu}-waste-hours", ascending=False).head(num_disp)
                jobs["Time-Used"] = jobs["elapsedraw"].apply(seconds_to_slurm_time_format)
                jobs["Time-Allocated"] = jobs["limit-minutes"].apply(lambda x: seconds_to_slurm_time_format(SECONDS_PER_MINUTE * x))
                jobs["Percent-Used"] = jobs["ratio"].apply(lambda x: f"{round(x)}%")
                jobs = jobs[["jobid", "netid", "Time-Used", "Time-Allocated", "Percent-Used", "cores"]].sort_values(by="jobid")
                renamings = {"jobid":"JobID", "netid":"NetID", "cores":"CPU-Cores"}
                jobs = jobs.rename(columns=renamings)
                edays = self.days_between_emails
                s =  f"{get_first_name(user)},\n\n"
                s += f"Below are {case} that ran on Della ({xpu.upper()}) in the past {edays} days:\n\n"
                s +=  "\n".join([4 * " " + row for row in jobs.to_string(index=False, justify="center").split("\n")])
                s += "\n"
                unused_hours = usr[f"{xpu.upper()}-Hours-Unused"].values[0]
                s += textwrap.dedent(f"""
                It appears that you are requesting too much time for your jobs since you are
                only using on average {usr['mean(%)'].values[0]}% of the allocated time (for the {total_jobs} jobs). This has
                resulted in {unused_hours} {xpu.upper()}-hours that you scheduled but did not use (it was made
                available to other users, however).

                Please request less time by modifying the --time Slurm directive. This will
                lower your queue times and allow the Slurm job scheduler to work more
                effectively for all users. For instance, if your job requires 8 hours then use:

                    #SBATCH --time=10:00:00

                The value above includes an extra 20% for safety. A good target for Percent-Used
                is 80%.

                Time-Used is the time (wallclock) that the job needed. The total time allocated
                for the job is Time-Allocated. The format is DD-HH:MM:SS where DD is days,
                HH is hours, MM is minutes and SS is seconds. Percent-Used is Time-Used
                divided by Time-Allocated.

                For more information on allocating time via Slurm:

                    https://researchcomputing.princeton.edu/support/knowledge-base/slurm

                Consider attending an in-person Research Computing help session for assistance:

                    https://researchcomputing.princeton.edu/support/help-sessions

                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)
                send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
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
