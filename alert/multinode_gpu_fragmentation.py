import pandas as pd
from base import Alert
from efficiency import gpu_efficiency
from utils import add_dividers
from utils import JOBSTATES
from utils import MINUTES_PER_HOUR as mph
from greeting import GreetingFactory
from email_translator import EmailTranslator


class MultinodeGpuFragmentation(Alert):

    """Find multinode GPU jobs that use too many nodes."""

    def __init__(self, df, days_between_emails, violation, vpath, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, **kwargs)

    def _add_required_fields(self):
        if not hasattr(self, "email_subject"):
            self.email_subject = "Multinode GPU Jobs with Fragmentation"
        if not hasattr(self, "report_title"):
            self.report_title = "Multinode GPU Jobs with Fragmentation"

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition.isin(self.partitions)) &
                          (self.df.gpus > 0) &
                          (self.df.nodes > 1) &
                          (self.df.gpus / self.df.nodes != self.gpus_per_node) &
                          (~self.df.user.isin(self.excluded_users)) &
                          (self.df.admincomment != {}) &
                          (self.df["elapsed-hours"] >= self.min_run_time / mph)].copy()
        self.df.rename(columns={"user":"User"}, inplace=True)
        # add new fields
        if not self.df.empty:
            self.df["GPU-eff-tpl"] = self.df.apply(lambda row:
                                                   gpu_efficiency(row["admincomment"],
                                                                  row["elapsedraw"],
                                                                  row["jobid"],
                                                                  row["cluster"],
                                                                  single=True),
                                                                  axis="columns")
            self.df["error-code"] = self.df["GPU-eff-tpl"].apply(lambda tpl: tpl[1])
            # drop jobs with non-zero error code
            self.df = self.df[self.df["error-code"] == 0]
            self.df["GPU-eff"] = self.df["GPU-eff-tpl"].apply(lambda tpl: tpl[0])
            self.df = self.df.drop(columns=["GPU-eff-tpl", "error-code"])

            self.df["GPUs-per-Node"] = self.df.gpus / self.df.nodes
            cols = ["jobid",
                    "User",
                    "gpus",
                    "nodes",
                    "partition",
                    "GPUs-per-Node",
                    "elapsed-hours",
                    "state",
                    "GPU-eff"]
            self.df = self.df[cols]
            self.df.state = self.df.state.apply(lambda x: JOBSTATES[x])
            renamings = {"jobid":"JobID",
                         "nodes":"Nodes",
                         "partition":"Partition",
                         "gpus":"GPUs",
                         "state":"State",
                         "elapsed-hours":"Hours"}
            self.df = self.df.rename(columns=renamings)
            self.df["GPU-eff"] = self.df["GPU-eff"].apply(lambda x: f"{round(x)}%")
            self.df["Hours"] = self.df["Hours"].apply(lambda x: str(round(x, 1))
                                                      if x < 5 else str(round(x)))
            self.df["GPUs-per-Node"] = self.df["GPUs-per-Node"].apply(lambda gpn:
                                                                str(round(gpn, 1)).replace(".0", ""))

    def create_emails(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.df.User.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.User == user].copy()
                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<DAYS>"] = str(self.days_between_emails)
                tags["<CLUSTER>"] = self.cluster
                tags["<PARTITIONS>"] = ",".join(sorted(set(usr.Partition)))
                usr.drop(columns=["Partition"], inplace=True)
                tags["<GPUS-PER-NODE>"] = str(self.gpus_per_node)
                indent = 4 * " "
                table = usr.to_string(index=False, justify="center").split("\n")
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                translator = EmailTranslator(self.email_file, tags)
                email = translator.replace_tags()
                self.emails.append((user, email, usr))

    def generate_report_for_admins(self, keep_index: bool=False) -> str:
        if self.df.empty:
            column_names = ["JobID",
                            "User",
                            "GPUs",
                            "Nodes",
                            "GPUs-per-Node",
                            "Hours",
                            "State", 
                            "GPU-Eff",
                            "Emails"]
            self.df = pd.DataFrame(columns=column_names)
            return add_dividers(self.create_empty_report(self.df), self.report_title)
        self.df["Emails"] = self.df.User.apply(lambda user:
                                 self.get_emails_sent_count(user, self.violation))
        self.df.Emails = self.format_email_counts(self.df.Emails)
        self.df = self.df.drop(columns=["Partition"])
        report_str = self.df.to_string(index=keep_index, justify="center")
        return add_dividers(report_str, self.report_title)
