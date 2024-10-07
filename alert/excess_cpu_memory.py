import textwrap
import pandas as pd
from base import Alert
from efficiency import cpu_memory_usage
from utils import get_first_name
from utils import send_email
from utils import add_dividers


class ExcessCPUMemory(Alert):

    """Cumulative memory use per user. Note that df is referenced in the
       send email method. If using this alert for multiple clusters then
       will need to prune df for each cluster. Similarly for proportion.

       Filter out jobs where used is greater than allocated?
       Should jobs that use the default CPU memory be ignored?"""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.admincomment != {}) &
                          (self.df.state != "RUNNING") &
                          (self.df.state != "OUT_OF_MEMORY") &
                          self.df.cluster.isin(self.clusters) &
                          self.df.partition.isin(self.partition) &
                          (~self.df.netid.isin(self.excluded_users)) &
                          (self.df["elapsed-hours"] >= 1)].copy()
        if self.combine_partitions:
            self.df["partition"] = ",".join(sorted(self.partition))
        self.gp = pd.DataFrame({"NetID":[]})
        self.admin = pd.DataFrame()
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
            # filter out jobs using approximately default memory
            self.df["GB-per-core"] = self.df["mem-alloc"] / self.df["cores"]
            self.df = self.df[self.df["GB-per-core"] > 5]
            # ignore jobs using default memory or less?
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
            self.gp = self.df.groupby(["cluster", "partition", "netid"]).agg(d)
            self.gp = self.gp.rename(columns={"netid":"jobs"})
            self.gp.reset_index(drop=False, inplace=True)
            total_mem_hours = self.gp["mem-hrs-alloc"].sum()
            assert total_mem_hours != 0, "total_mem_hours found to be zero"
            self.gp["proportion"] = self.gp["mem-hrs-alloc"] / total_mem_hours
            self.gp["ratio"] = self.gp["mem-hrs-used"] / self.gp["mem-hrs-alloc"]
            self.gp = self.gp.sort_values("mem-hrs-unused", ascending=False)
            cols = ["cluster",
                    "partition",
                    "netid",
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
            # should email90 be computed for a specific cluster and partition?
            self.gp["email90"] = self.gp["NetID"].apply(lambda netid:
                                                 self.get_emails_sent_count(netid,
                                                                            self.violation,
                                                                            days=90))
            cols = ["mem-hrs-unused", "mem-hrs-used", "mem-hrs-alloc", "cpu-hrs"]
            self.gp[cols] = self.gp[cols].apply(round).astype("int64")
            cols = ["proportion", "ratio", "mean-ratio", "median-ratio"]
            self.gp[cols] = self.gp[cols].apply(lambda x: round(x, 2))
            self.gp["avg-cores"] = self.gp["avg-cores"].apply(lambda x: round(x, 1))
            self.gp.reset_index(drop=True, inplace=True)
            self.gp.index += 1
            self.gp = self.gp.head(self.num_top_users)
            self.admin = self.gp.copy()
            self.gp = self.gp[(self.gp["mem-hrs-unused"] > self.tb_hours_per_day * self.days_between_emails) &
                              (self.gp["ratio"] < self.ratio_threshold) &
                              (self.gp["mean-ratio"] < self.mean_ratio_threshold) &
                              (self.gp["median-ratio"] < self.median_ratio_threshold)]

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
                hours_per_week = 7 * 24
                tb_mem_per_node = 190 / 1e3
                num_wasted_nodes = round(unused / hours_per_week / tb_mem_per_node)
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
                s += f"Below are {case} that ran on Della (cpu) in the past {edays} days:\n\n"
                jobs_str = jobs.to_string(index=False, justify="center")
                s +=  "\n".join([4 * " " + row for row in jobs_str.split("\n")])
                s += "\n"
                s += textwrap.dedent(f"""
                It appears that you are requesting too much CPU memory for your jobs since you
                are only using on average {pct}% of the allocated memory (for the {total_jobs} jobs). This
                has resulted in {unused} TB-hours of unused memory which is equivalent to making
                {num_wasted_nodes} nodes unavailable to all users (including yourself) for one week! A TB-hour is
                the allocation of 1 terabyte of memory for 1 hour.

                AS PER THE RESEARCH COMPUTING ADVISORY GROUP, IN THE COMING WEEKS, EXCESSIVE
                CPU MEMORY ALLOCATION WILL RESULT IN YOUR ADVISOR BEING CONTACTED. AFTER THAT
                IF THE MATTER IS NOT RESOLVED WITHIN 7 DAYS THEN YOUR ACCOUNT WILL BE
                SUSPENDED.

                Please request less memory by modifying the --mem-per-cpu or --mem Slurm
                directive. This will lower your queue times and increase the overall throughput
                of your jobs. For instance, if your job requires 8 GB per node then use:

                    #SBATCH --mem=10G

                The value above includes an extra 20% for safety. A good target value for
                Percent-Used is 80%. For more on allocating CPU memory with Slurm:

                    https://researchcomputing.princeton.edu/support/knowledge-base/memory

                You can check the CPU memory utilization of completed and actively running jobs
                by using the \"jobstats\" command. For example:

                    $ jobstats {jobs['JobID'].values[0]}

                The command above can also be used to see suggested values for the --mem-per-cpu
                and --mem Slurm directives.

                Consider attending an in-person Research Computing help session for assistance:

                    https://researchcomputing.princeton.edu/support/help-sessions

                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)
                send_email(s, f"{user}@princeton.edu", subject=f"{self.subject}")
                for email in self.admin_emails:
                    send_email(s, f"{email}", subject=f"{self.subject}")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        """Drop and rename some of the columns."""
        if self.admin.empty:
            return ""
        else:
            cols = ["cluster",
                    "partition",
                    "NetID",
                    "proportion",
                    "mem-hrs-unused",
                    "mem-hrs-used",
                    "ratio",
                    "mean-ratio",
                    "median-ratio",
                    "cpu-hrs",
                    "jobs",
                    "email90"]
            self.admin = self.admin[cols]
            renamings = {"mem-hrs-unused":"unused",
                         "mem-hrs-used":"used",
                         "mean-ratio":"mean",
                         "median-ratio":"median"}
            self.admin = self.admin.rename(columns=renamings)
            return add_dividers(self.admin.to_string(index=keep_index, justify="center"), title)
