from base import Alert
from efficiency import gpu_efficiency
from utils import send_email
from utils import add_dividers
from utils import JOBSTATES
from utils import MINUTES_PER_HOUR as mph
from greeting import GreetingFactory
from email_translator import EmailTranslator


class MultinodeGpuFragmentation(Alert):

    """Find multinode GPU jobs that use 1 GPU per node."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        self.excluded_users = []
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

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
            self.df["Hours"] = self.df["Hours"].apply(lambda hrs: round(hrs, 1))
            self.df["GPUs-per-Node"] = self.df["GPUs-per-Node"].apply(lambda gpn:
                                                                str(round(gpn, 1)).replace(".0", ""))

    def send_emails_to_users(self, method):
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
                s = translator.replace_tags()

                send_email(s, f"{user}@princeton.edu", subject=f"{self.subject}")
                for email in self.admin_emails:
                    send_email(s, email, subject=f"{self.subject}")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            self.df["email"] = self.df.User.apply(lambda user:
                                            self.get_emails_sent_count(user, self.violation))
            self.df = self.df.drop(columns=["Partition"])
            post  = f"   Cluster: {self.cluster}\n" 
            post += f"Partitions: {', '.join(self.partitions)}\n" 
            return add_dividers(self.df.to_string(index=keep_index, justify="center"), title, post=post)
