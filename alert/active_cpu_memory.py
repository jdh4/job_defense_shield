import re
import textwrap
from time import sleep
import pandas as pd
from base import Alert
from efficiency import cpu_memory_usage
from utils import MINUTES_PER_HOUR
from utils import HOURS_PER_DAY
from utils import get_first_name
from utils import send_email
from utils import add_dividers

class ActiveCPUMemory(Alert):

    """Find actively running jobs that not using the CPU memory. Consider only
       single-node jobs to get started. To reduce calls to jobstats, use
       alloctres to get memory allocated."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    @staticmethod
    def get_stats_for_running_job(jobid, cluster):
        """Get the job statistics for running jobs by calling jobstats"""
        import importlib.machinery
        import importlib.util
        cluster = cluster.replace("tiger", "tiger2")
        loader = importlib.machinery.SourceFileLoader('jobstats', '/usr/local/bin/jobstats')
        spec = importlib.util.spec_from_loader('jobstats', loader)
        mymodule = importlib.util.module_from_spec(spec)
        loader.exec_module(mymodule)
        stats = mymodule.JobStats(jobid=jobid, cluster=cluster, prom_server="http://vigilant2:8480")
        sleep(1)
        return eval(stats.report_job_json(False))

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[self.df.cluster.isin(["della"]) &
                          self.df.partition.isin(["cpu"]) &
                          (self.df.state == "RUNNING") &
                          (self.df.nodes == 1) &
                          (self.df.cores < 8) &
                          (self.df["limit-minutes"] >= 2 * HOURS_PER_DAY * MINUTES_PER_HOUR) &
                          (self.df["elapsed-hours"] >= 4) &
                          (self.df["elapsed-hours"] < 8)].copy()

        # get allocated memory from alloctres field and filter
        def mem_from_alloctres(tres: str) -> int:
            matches = re.findall(r"mem=\d+\.?\d*T", tres)
            if len(matches) == 1:
                return int(matches[0][:-1].split("=")[-1]) * 1000
            matches = re.findall(r"mem=\d+\.?\d*G", tres)
            if len(matches) == 1:
                return int(matches[0][:-1].split("=")[-1])
            matches = re.findall(r"mem=\d+\.?\d*M", tres)
            if len(matches) == 1:
                return int(matches[0][:-1].split("=")[-1]) / 1000
            return 0
        self.df["mem"] = self.df["alloctres"].apply(mem_from_alloctres)
        self.df = self.df[self.df["mem"] >= 50]
        
        self.admin = pd.DataFrame()
        if not self.df.empty:
            self.df["jobstats"] = self.df.apply(lambda row:
                                       self.get_stats_for_running_job(row["jobid"],
                                                                      row["cluster"]),
                                                                      axis='columns')
            self.df["memory-tuple"] = self.df.apply(lambda row:
                                                    cpu_memory_usage(row["jobstats"],
                                                                     row["jobid"],
                                                                     row["cluster"]),
                                                                     axis="columns")
            self.df["mem-used"]  = self.df["memory-tuple"].apply(lambda x: x[0])
            self.df["mem-alloc"] = self.df["memory-tuple"].apply(lambda x: x[1])
            # next line guards against division by zero when computing self.df.ratio
            self.df = self.df[self.df["mem-alloc"] >= 50]
            self.df["ratio"] = self.df["mem-used"] / self.df["mem-alloc"]
            self.df = self.df[self.df["ratio"] < 0.2]
            self.df["Limit-Hours"] = self.df["limit-minutes"].apply(lambda x:
                                                                    round(x / MINUTES_PER_HOUR))
            self.admin = self.df.copy()

    def send_emails_to_users(self):
        for user in self.df.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.netid == user].copy()
                cols = ["jobid",
                        "mem-used",
                        "mem-alloc",
                        "cores",
                        "elapsed-hours",
                        "Limit-Hours"] 
                renamings = {"jobid":"JobID",
                             "mem-used":"Memory-Used",
                             "mem-alloc":"Memory-Allocated",
                             "cores":"Cores",
                             "elapsed-hours":"Elapsed-Hours"}
                usr = usr[cols].rename(columns=renamings)
                s =  f"{get_first_name(user)},\n\n"
                s += "Below are jobs currently running on Della (cpu):\n\n"
                usr_str = usr.to_string(index=False, justify="center")
                s +=  "\n".join([4 * " " + row for row in usr_str.split("\n")])
                s += "\n"
                s += textwrap.dedent(f"""
                It appears that the jobs above have allocated too much memory.

                Consider attending an in-person Research Computing help session for assistance:

                    https://researchcomputing.princeton.edu/support/help-sessions
       
                For more on allocating CPU memory with Slurm:

                    https://researchcomputing.princeton.edu/support/knowledge-base/memory

                You can check the CPU memory utilization of completed and actively running jobs
                by using the \"jobstats\" command. For example:

                    $ jobstats {usr['JobID'].values[0]}

                The command above can also be used to see suggested values for the --mem-per-cpu
                and --mem Slurm directives.
 
                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)

                #send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                print(s)

                # append the new violations to the log file
                #Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        """Drop and rename some of the columns."""
        if self.admin.empty:
            return ""
        else:
            cols = ["jobid",
                    "netid",
                    "cluster",
                    "partition",
                    "mem-alloc",
                    "mem-used",
                    "ratio",
                    "elapsed-hours",
                    "Limit-Hours"]
            self.admin = self.admin[cols]
            renamings = {"jobid":"JobID"}
            self.admin = self.admin.rename(columns=renamings)
            return add_dividers(self.admin.to_string(index=keep_index, justify="center"), title)
