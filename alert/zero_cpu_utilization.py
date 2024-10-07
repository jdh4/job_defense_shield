import textwrap
import pandas as pd
from base import Alert
from utils import get_first_name
from utils import send_email
from utils import add_dividers
from efficiency import cpu_nodes_with_zero_util


class ZeroCPU(Alert):

    """CPU jobs with zero utilization on one or more nodes."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster != "traverse") &
                          (self.df.partition != "gpu") &
                          (self.df.state != "RUNNING") &
                          (self.df.admincomment != {}) &
                          (self.df["elapsed-hours"] >= 1)].copy()
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
                    "netid",
                    "cluster",
                    "nodes",
                    "nodes-unused",
                    "CPU-Util-Unused",
                    "cores",
                    "elapsed-hours"]
            self.df = self.df[cols]
            renamings = {"jobid":"JobID",
                         "netid":"NetID",
                         "cluster":"Cluster",
                         "nodes":"Nodes",
                         "nodes-unused":"Nodes-Unused",
                         "cores":"Cores",
                         "elapsed-hours":"Hours"}
            self.df = self.df.rename(columns=renamings)

    def send_emails_to_users(self):
        for user in self.df.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                usr.drop(columns=["NetID"], inplace=True)
                usr["Nodes-Used"] = usr["Nodes"] - usr["Nodes-Unused"]
                num_jobs = usr.shape[0]
                all_single = bool(usr.shape[0] == usr[usr["Nodes"] == 1].shape[0])
                all_multi =  bool(usr.shape[0] == usr[usr["Nodes"] > 1].shape[0])
                all_used_1 = bool(usr.shape[0] == usr[usr["Nodes-Used"] == 1].shape[0])
                usr.drop(columns=["Nodes-Used"], inplace=True)
                if num_jobs == 1 and all_single:
                    continue
                s =  f"{get_first_name(user)},\n\n"
                s += f"Below are your recent jobs that did not use all of the allocated nodes:\n\n"
                usr_str = usr.to_string(index=False, justify="center")
                s +=  "\n".join([4 * " " + row for row in usr_str.split("\n")])
                s += "\n"
                s += textwrap.dedent(f"""
                The CPU utilization was found to be 0% on each of the unused nodes. You can see
                this by running the \"jobstats\" command, for example:

                    $ jobstats {usr['JobID'].values[0]}
                """)
                if all_single:
                    s += textwrap.dedent(f"""
                    Please investigate the reason(s) that the code is not using all of the allocated
                    nodes before running additional jobs.
                    """)
                else:
                    s += textwrap.dedent(f"""
                    Please investigate the reason(s) that the code is not using all of the allocated
                    nodes before running additional jobs. When appropriate, please allocate all of
                    the cores on a single node before requesting multiple nodes.
                    """)
                if all_multi and all_used_1:
                    s += textwrap.dedent(f"""
                    Only 1 node was used for each of the jobs above. This suggests that your
                    code may not be capable of using multiple nodes. Please consult the
                    documentation or write to the mailing list of your software to confirm that
                    it is capable of using multiple nodes. If your code cannot use multiple
                    nodes then please use the following Slurm directive:

                        #SBATCH --nodes=1
                    """)
                s += textwrap.dedent(f"""
                See the following webpage for information about Slurm:

                    https://researchcomputing.princeton.edu/support/knowledge-base/slurm

                Consider attending an in-person Research Computing help session for assistance:

                    https://researchcomputing.princeton.edu/support/help-sessions

                After resolving this issue, consider conducting a scaling analysis to find
                the optimal number of CPU-cores to use:

                    https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis

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
        """Rename some of the columns."""
        if self.df.empty:
            return ""
        else:
            self.df = self.df.sort_values(["NetID", "JobID"]) 
            return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
