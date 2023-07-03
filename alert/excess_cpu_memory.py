import os
from datetime import datetime
from datetime import timedelta
import textwrap
from base import Alert
from efficiency import cpu_memory_usage
from utils import get_first_name
from utils import send_email
from utils import add_dividers
import pandas as pd


class ExcessCPUMemory(Alert):

    """Cumulative memory use per user."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition == self.partition) &
                          (self.df.admincomment != {}) &
                          (self.df.state != "RUNNING") &
                          (self.df.state != "OUT_OF_MEMORY") &
                          (self.df["elapsed-hours"] >= 1)].copy()
        self.ad = pd.DataFrame()
        # add new fields
        if not self.df.empty:
            # only consider jobs that do not use (approximately) full nodes
            self.df = self.df[self.df["cores"] / self.df["nodes"] < self.cores_per_node]
            self.df["memory-tuple"] = self.df.apply(lambda row:
                                                    cpu_memory_usage(row["admincomment"],
                                                                     row["jobid"],
                                                                     row["cluster"]),
                                                                     axis="columns")
            self.df["mem-used"]   = self.df["memory-tuple"].apply(lambda x: x[0])
            self.df["mem-alloc"]  = self.df["memory-tuple"].apply(lambda x: x[1])
            self.df["mem-unused"] = self.df["mem-alloc"] - self.df["mem-used"]
            GB_per_TB = 1000
            self.df["mem-hrs-used"]   = self.df["mem-used"] * self.df["elapsed-hours"] / GB_per_TB
            self.df["mem-hrs-alloc"]  = self.df["mem-alloc"] * self.df["elapsed-hours"] / GB_per_TB
            self.df["mem-hrs-unused"] = self.df["mem-hrs-alloc"] - self.df["mem-hrs-used"]
            self.df["mean-ratio"]     = self.df["mem-hrs-used"] / self.df["mem-hrs-alloc"]
            self.df["median-ratio"]   = self.df["mem-hrs-used"] / self.df["mem-hrs-alloc"]
            # compute various quantities by grouping by user
            d = {"mem-hrs-used":"sum",
                 "mem-hrs-alloc":"sum",
                 "mem-hrs-unused":"sum",
                 "elapsed-hours":"sum",
                 "cores":"mean",
                 "cpu-hours":"sum",
                 "account":pd.Series.mode,
                 "mean-ratio":"mean",
                 "median-ratio":"median",
                 "netid":"size"}
            self.gp = self.df.groupby("netid").agg(d)
            self.gp = self.gp.rename(columns={"netid":"jobs"})
            self.gp.reset_index(drop=False, inplace=True)
            # next line is safeguard against division by zero
            total_mem_hours = max(1, self.gp["mem-hrs-alloc"].sum())
            self.gp["proportion"] = self.gp["mem-hrs-alloc"] / total_mem_hours
            self.gp["ratio"] = self.gp["mem-hrs-used"] / self.gp["mem-hrs-alloc"]
            self.gp = self.gp.sort_values("mem-hrs-unused", ascending=False)
            cols = ["netid",
                    "account",
                    "proportion",
                    "mem-hrs-unused",
                    "mem-hrs-used",
                    "mem-hrs-alloc",
                    "ratio",
                    "median-ratio",
                    "mean-ratio",
                    "elapsed-hours",
                    "cores",
                    "cpu-hours",
                    "jobs"]
            self.gp = self.gp[cols]
            renamings = {"netid":"NetID",
                         "elapsed-hours":"hrs",
                         "cores":"avg-cores",
                         "cpu-hours":"cpu-hrs"}
            self.gp = self.gp.rename(columns=renamings)
            self.gp["emails"] = self.gp["NetID"].apply(self.get_emails_sent_count)
            cols = ["mem-hrs-unused", "mem-hrs-used", "mem-hrs-alloc", "cpu-hrs"]
            self.gp[cols] = self.gp[cols].apply(round).astype("int64")
            cols = ["proportion", "ratio", "mean-ratio", "median-ratio"]
            self.gp[cols] = self.gp[cols].apply(lambda x: round(x, 2))
            self.gp["avg-cores"] = self.gp["avg-cores"].apply(lambda x: round(x, 1))
            self.gp.reset_index(drop=True, inplace=True)
            self.gp.index += 1
            self.gp = self.gp.head(self.num_top_users)
            self.ad = self.gp.copy()
            self.gp = self.gp[(self.gp["mem-hrs-unused"] > 15 * self.days_between_emails) & 
                              (self.gp["ratio"] < 0.2) & 
                              (self.gp["mean-ratio"] < 0.2) & 
                              (self.gp["median-ratio"] < 0.2)]

    def get_emails_sent_count(self, user: str) -> int:
        """Return the number of datascience emails sent in the last 30 days."""
        prev_violations = f"{self.vpath}/datascience/{user}.email.csv"
        if os.path.exists(prev_violations):
            d = pd.read_csv(prev_violations, parse_dates=["email_sent"])
            start_date = datetime.now() - timedelta(days=30)
            return d[d["email_sent"] >= start_date]["email_sent"].unique().size
        else:
            return 0

    def send_emails_to_users(self):
        for user in self.gp.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.gp[self.gp.NetID == user].copy()
                jobs = self.df[self.df.netid == user].copy()
                num_disp = 10
                total_jobs = jobs.shape[0]
                case = f"{num_disp} of your {total_jobs} jobs" if total_jobs > num_disp else "your jobs"
                pct = round(100 * usr["mean-ratio"].values[0])
                unused = usr["mem-hrs-unused"].values[0]
                jobs = jobs.sort_values(by="mem-hrs-unused", ascending=False).head(num_disp)
                jobs = jobs[["jobid",
                             "mem-used",
                             "mem-alloc",
                             "mean-ratio",
                             "cores",
                             "elapsed-hours"]]
                jobs["mem-used"]   = jobs["mem-used"].apply(lambda x: f"{round(x)} GB")
                jobs["mem-alloc"]  = jobs["mem-alloc"].apply(lambda x: f"{round(x)} GB")
                jobs["mean-ratio"] = jobs["mean-ratio"].apply(lambda x: f"{round(100 * x)}%")
                jobs = jobs.sort_values(by="jobid")
                renamings = {"jobid":"JobID",
                             "mem-used":"Memory-Used",
                             "mem-alloc":"Memory-Allocated",
                             "mean-ratio":"Percent-Used",
                             "cores":"Cores",
                             "elapsed-hours":"Hours"}
                jobs = jobs.rename(columns=renamings)
                edays = self.days_between_emails
                s =  f"{get_first_name(user)},\n\n"
                s += f"Below are {case} that ran on Della (CPU) in the past {edays} days:\n\n"
                s +=  "\n".join([4 * " " + row for row in jobs.to_string(index=False, justify="center").split("\n")])
                s += "\n"
                s += textwrap.dedent(f"""
                It appears that you are requesting too much CPU memory for your jobs since you
                are only using on average {pct}% of the allocated memory (for the {total_jobs} jobs). This
                has resulted in {unused} TB-hours of unused memory. A TB-hour is the allocation
                of 1 terabyte of memory for 1 hour.

                Please request less memory by modifying the --mem-per-cpu or --mem Slurm
                directive. This will lower your queue times and make the resources available
                to other users. For instance, if your job requires 8 GB per node then use:

                    #SBATCH --mem=10G

                The value above includes an extra 20% for safety. A good target value for
                Percent-Used is 80%. For more on allocating CPU memory with Slurm:

                    https://researchcomputing.princeton.edu/support/knowledge-base/memory

                You can check the CPU memory utilization of completed and actively running jobs
                by using the \"jobstats\" command. For example:

                    $ jobstats {jobs['JobID'].values[0]}

                The command above can also be used to see suggested values for the --mem-per-cpu
                and --mem Slurm directives.

                Add the following lines to your Slurm scripts to receive an email report with
                CPU memory utilization information after each job finishes:

                    #SBATCH --mail-type=end
                    #SBATCH --mail-user={user}@princeton.edu

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
        """Drop and rename some of the columns."""
        if self.ad.empty:
            return ""
        else:
            cols = ["NetID",
                    "proportion",
                    "mem-hrs-unused",
                    "mem-hrs-used",
                    "ratio",
                    "mean-ratio",
                    "median-ratio",
                    "cpu-hrs",
                    "jobs",
                    "emails"]
            self.ad = self.ad[cols]
            renamings = {"mem-hrs-unused":"TB-hrs-unused",
                         "mem-hrs-used":"TB-hrs-used",
                         "mean-ratio":"mean",
                         "median-ratio":"median"}
            self.ad = self.ad.rename(columns=renamings)
            return add_dividers(self.ad.to_string(index=keep_index, justify="center"), title)
