import re
import math
import textwrap
from time import sleep
import pandas as pd
from base import Alert
from efficiency import cpu_memory_usage
from utils import MINUTES_PER_HOUR
from utils import HOURS_PER_DAY
from utils import get_first_name
from utils import send_email
from utils import send_email_html
from utils import add_dividers

class ActiveCPUMemory(Alert):

    """Find actively running jobs that not using the CPU memory. Consider only
       single-node jobs to get started. To reduce calls to jobstats, use
       alloctres to get memory allocated. Note that the memory value from alloctres
       may not be equal to the reqmem field. It appears that reqmem is less than
       or equal to the alloctres value."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    @staticmethod
    def rounded_memory_with_safety(mem_used: float) -> int:
        """Return a rounded version of the suggested memory in GB including
           20% safety."""
        mem_with_safety = math.ceil(1.2 * mem_used)
        if mem_with_safety > 1000:
            mem_suggested = round(mem_with_safety, -2)
            if mem_suggested - mem_with_safety < 0: mem_suggested += 100
        elif mem_with_safety > 100:
            mem_suggested = round(mem_with_safety, -1)
            if mem_suggested - mem_with_safety < 0: mem_suggested += 10
        elif mem_with_safety > 30:
            mem_suggested = round(mem_with_safety, -1)
            if mem_suggested - mem_with_safety < 0: mem_suggested += 5
        else:
            return max(1, mem_with_safety)
        return mem_suggested

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
        stats = mymodule.JobStats(jobid=jobid,
                                  cluster=cluster,
                                  prom_server="http://vigilant2:8480")
        sleep(1)
        return eval(stats.report_job_json(False))

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[self.df.cluster.isin(["della"]) &
                          self.df.partition.isin(["cpu"]) &
                          (self.df.state == "RUNNING") &
                          (self.df.nodes == 1) &
                          (self.df.cores < 28) &
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
            self.df["GB-per-core"] = self.df["mem-alloc"] / self.df["cores"]
            self.df["ratio"] = self.df["mem-used"] / self.df["mem-alloc"]
            self.df = self.df[(self.df["ratio"] < 0.4) & (self.df["GB-per-core"] > 5)]
            self.df["blocked-cores"] = 32 - self.df["cores"] - (190 - self.df["mem-alloc"]) / 4
            self.df["Limit-Hours"] = self.df["limit-minutes"].apply(lambda x:
                                                                    round(x / MINUTES_PER_HOUR))
            self.admin = self.df.copy()

    def send_emails_to_users(self):
        for user in self.df.netid.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.netid == user].copy()
                mem_used = float(usr['mem-used'].values[0])
                cores = int(usr['cores'].values[0])
                usr["mem-used"]  =  usr["mem-used"].apply(lambda x: f'{str(x).replace(".0", "")} GB')
                usr["mem-alloc"] = usr["mem-alloc"].apply(lambda x: f'{str(x).replace(".0", "")} GB')
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
                #usr_str = usr.to_string(index=False, justify="center")
                #s +=  "\n".join([4 * " " + row for row in usr_str.split("\n")])
                s += usr.to_html(index=False, border=1, justify="center").replace('<td>', '<td align="center">')
                s += "\n"
                s += textwrap.dedent(f"""
                It appears that the jobs above have allocated too much memory since
                "Memory-Used" is much less than "Memory-Allocated". Ideally, "Memory-Used"
                should be about 80% of "Memory-Allocated".

                For instance, for {usr['JobID'].values[0]} use a Slurm directive such as --mem-per-cpu={self.rounded_memory_with_safety(mem_used/cores)}G
                or --mem={self.rounded_memory_with_safety(mem_used)}G.

                For more on allocating CPU memory with Slurm:

                    https://researchcomputing.princeton.edu/support/knowledge-base/memory

                By using accurate values for the allocated memory you will experience
                shorter queues times. When you allocate excess CPU memory you prevent
                yourself and other resources from using the remaingin CPU-cores on the
                node.

                You can check the CPU memory utilization of completed and actively running jobs
                by using the \"jobstats\" command. For example:

                    $ jobstats {usr['JobID'].values[0]}

                Consider attending an in-person Research Computing help session for assistance:

                    https://researchcomputing.princeton.edu/support/help-sessions
       
                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)

                #send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                s = '<p align="left">Hi Alan:</p>' + \
                    '<p align="left">Below is a table showing underutilization on the clusters:</p>' + \
                    '<font face="Courier New, Courier, monospace">' + \
                    usr.to_html(index=False, border=0, justify="center").replace('<td>', '<td align="center">') + \
                    '</font>' + \
                    '<p align="left">By using accurate values for the allocated memory you will experience shorter queues times. When you allocate excess CPU memory you prevent yourself and other resources from using the remaingin CPU-cores on the node.</p>' + \
                    '<p align="left">You can check the CPU memory utilization of completed and actively running jobs by using the "jobstats" command. For example:</p>' + \
                    '<p align="left"><font face="Courier New, Courier, monospace">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ jobstats 1234567</font></p>' + \
                    '<p align="left">Learn more about <a href="https://researchcomputing.princeton.edu/support/knowledge-base/memory">allocating CPU memory with Slurm</a>.</p>' + \
                    '<p align="left">Consider attending an in-person <a href="https://researchcomputing.princeton.edu/support/help-sessions">Research Computing help session</a> for assistance. ' + \
                    'Replying to this automated email will open a support ticket with <a href="https://researchcomputing.princeton.edu">Research Computing</a>. Let us know if we can be of help.</p>'
                send_email_html(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email_html(s, "halverson.jonathan@gmail.com", subject=f"{self.subject}", sender="cses@princeton.edu")
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
                    "blocked-cores",
                    "elapsed-hours",
                    "Limit-Hours"]
            self.admin = self.admin[cols]
            renamings = {"jobid":"JobID"}
            self.admin = self.admin.rename(columns=renamings)
            return add_dividers(self.admin.to_string(index=keep_index, justify="center"), title)
