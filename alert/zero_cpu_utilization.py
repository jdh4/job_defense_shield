import pandas as pd
from base import Alert
from utils import add_dividers
from utils import MINUTES_PER_HOUR as mph
from efficiency import cpu_nodes_with_zero_util
from greeting import GreetingFactory
from email_translator import EmailTranslator


class ZeroCPU(Alert):

    """CPU jobs with zero utilization on one or more nodes."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        self.excluded_users = []
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition.isin(self.partitions)) &
                          (self.df.admincomment != {}) &
                          (~self.df.user.isin(self.excluded_users)) &
                          (self.df["elapsed-hours"] >= self.min_run_time / mph)].copy()
        # add new fields
        if not self.df.empty:
            self.df["nodes-tuple"] = self.df.apply(lambda row:
                                     cpu_nodes_with_zero_util(row["admincomment"],
                                                              row["jobid"],
                                                              row["cluster"]),
                                                              axis="columns")
            cols = ["nodes-unused", "error_code"]
            self.df[cols] = pd.DataFrame(self.df["nodes-tuple"].tolist(), index=self.df.index)
            self.df = self.df[(self.df["error_code"] == 0) & (self.df["nodes-unused"] > 0)]
            def is_interactive(jobname):
                if jobname.startswith("sys/dashboard") or jobname.startswith("interactive"):
                    return True
                return False
            self.df["interactive"] = self.df["jobname"].apply(is_interactive)
            self.df["CPU-Util-Unused"] = "0%"
            cols = ["jobid",
                    "user",
                    "cluster",
                    "nodes",
                    "nodes-unused",
                    "CPU-Util-Unused",
                    "cores",
                    "elapsed-hours"]
            self.df = self.df[cols]
            renamings = {"jobid":"JobID",
                         "user":"User",
                         "cluster":"Cluster",
                         "nodes":"Nodes",
                         "nodes-unused":"Nodes-Unused",
                         "cores":"Cores",
                         "elapsed-hours":"Hours"}
            self.df = self.df.rename(columns=renamings)
            self.df["Hours"] = self.df["Hours"].apply(lambda x: str(round(x, 1))
                                                      if x < 5 else str(round(x)))

    def create_emails(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.df.User.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.User == user].copy()
                if len(usr) == 1 and \
                   usr.Nodes.values[0] == 1 and \
                   usr.Cores.values[0] < 4:
                    continue
                usr.drop(columns=["User"], inplace=True)
                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<DAYS>"] = str(self.days_between_emails)
                tags["<CLUSTER>"] = self.cluster
                tags["<PARTITIONS>"] = ",".join(self.partitions)
                tags["<NUM-JOBS>"] = str(len(usr))
                indent = 4 * " "
                table = usr.to_string(index=False, justify="center").split("\n")
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                translator = EmailTranslator(self.email_file, tags)
                email = translator.replace_tags()
                self.emails.append((user, email, usr))

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        """Rename some of the columns."""
        if self.df.empty:
            return ""
        else:
            self.df = self.df.sort_values(["User", "JobID"])
            self.df["emails"] = self.df.User.apply(lambda user:
                                     self.get_emails_sent_count(user, self.violation))
            self.df.emails = self.format_email_counts(self.df.emails)
            return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
