import textwrap
from base import Alert
from efficiency import gpu_efficiency
from utils import send_email
from utils import add_dividers
from utils import JOBSTATES


class MultinodeGPUFragmentation(Alert):

    """Find multinode GPU jobs that use 1 GPU per node."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition == self.partition) &
                          (self.df.state != "RUNNING") &
                          (self.df.gpus > 0) &
                          (self.df.nodes > 1) &
                          (self.df.nodes == self.df.gpus) &
                          (self.df["elapsed-hours"] >= 1)].copy()
        self.df.rename(columns={"netid":"NetID"}, inplace=True)
        # add new fields
        if not self.df.empty:
            self.df["GPU-eff-tpl"] = self.df.apply(lambda row:
                                                   gpu_efficiency(row["admincomment"],
                                                                  row["elapsedraw"],
                                                                  row["jobid"],
                                                                  row["cluster"], single=True)
                                                   if row["admincomment"] != {} else (999, 0), axis="columns")
            self.df["error-code"] = self.df["GPU-eff-tpl"].apply(lambda tpl: tpl[1])
            # drop jobs with non-zero error code
            self.df = self.df[self.df["error-code"] == 0]
            self.df["GPU-eff"] = self.df["GPU-eff-tpl"].apply(lambda tpl: tpl[0])
            self.df = self.df.drop(columns=["GPU-eff-tpl", "error-code"])

            self.gpu_util_thres = 50
            self.df["GPUs-per-Node"] = 1
            cols = ["jobid",
                    "NetID",
                    "gpus",
                    "nodes",
                    "GPUs-per-Node",
                    "elapsed-hours",
                    "state",
                    "GPU-eff"]
            self.df = self.df[cols]
            self.df.state = self.df.state.apply(lambda x: JOBSTATES[x])
            renamings = {"jobid":"JobID",
                         "nodes":"Nodes",
                         "gpus":"GPUs",
                         "state":"State",
                         "elapsed-hours":"Hours"}
            self.df = self.df.rename(columns=renamings)

    def send_emails_to_users(self):
        for user in self.df.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                has_low_gpu_util = bool(usr[usr["GPU-eff"] < self.gpu_util_thres].shape[0])
                usr["GPU-eff"] = usr["GPU-eff"].apply(lambda x: "--" if x == 999 else f"{round(x)}%")
                usr["Hours"] = usr["Hours"].apply(lambda hrs: round(hrs, 1))
                edays = self.days_between_emails
                s = f"{Greeting(user).greeting()}"
                s += f"Below are jobs that ran on Della in the past {edays} days that used 1 GPU per node\n"
                s +=  "over multiple nodes:\n\n"
                s +=  "\n".join([4 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
                if has_low_gpu_util:
                    s +=  "\n\n"
                    s += f"There is at least one job above with a GPU efficiency of less than {self.gpu_util_thres}%. In these\n"
                    s +=  "cases please consider using only 1 GPU per job to improve the efficiency."
                s += "\n"
                s += textwrap.dedent(f"""
                The GPU nodes on Della have either 2 GPUs per node or 4 GPUs per node. For future
                jobs, please try to use as few nodes as possible by allocating more GPUs per node.
                This is done by modifying the --gres Slurm directive as explained here:

                    https://researchcomputing.princeton.edu/support/knowledge-base/slurm#gpus

                For more information about the Della GPU nodes:

                    https://researchcomputing.princeton.edu/systems/della#gpus

                When using more than 1 GPU per job, be sure to conduct a scaling analysis to find
                the optimal number of GPUs:

                    https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis

                Consider attending an in-person Research Computing help session for assistance:

                    https://researchcomputing.princeton.edu/support/help-sessions

                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)
                send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}")
                send_email(s, "alerts-jobs-aaaalegbihhpknikkw2fkdx6gi@princetonrc.slack.com", subject=f"{self.subject}")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            self.df["GPU-eff"] = self.df["GPU-eff"].apply(lambda x: "--" if x == 999 else f"{round(x)}%")
            return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
