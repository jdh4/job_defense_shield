import textwrap
from glob import glob
import pandas as pd
from datetime import datetime
from datetime import timedelta
from base import Alert
from utils import get_first_name
from utils import send_email_cses


class HardWarningCPUMemory(Alert):

    """Hard warning about underutilizing CPU memory."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    def _filter_and_add_new_fields(self):
        pass

    def send_emails_to_users(self):
        xfiles = glob(f"{self.vpath}/excess_cpu_memory/*.email.csv")
        for xfile in xfiles:
            user = xfile.split("/")[-1].split(".email")[0]
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = pd.read_csv(xfile, parse_dates=["email_sent"])
                if "cluster" not in usr.columns:
                    usr["cluster"] = "della"
                if "partition" not in usr.columns:
                    usr["partition"] = "cpu"
                usr = usr[(usr.cluster == self.cluster) & (usr.partition == self.partition)]
                start_date = datetime.now() - timedelta(days=90)
                usr = usr[usr.email_sent > start_date]
                num_warnings = usr.shape[0]
                if num_warnings >= 3:
                    tb_hrs_unused = usr["mem-hrs-unused"].sum()
                    tb_hrs_used   = usr["mem-hrs-used"].sum()
                    hours_per_week = 7 * 24
                    num_wasted_nodes = round(tb_hrs_unused / (0.190 * hours_per_week))
                    usr["Week-Ending-On"] = usr["email_sent"].apply(lambda d: d.strftime("%-m/%-d/%Y"))
                    dt = datetime.now() - usr.email_sent.min()
                    assert dt.days > 1
                    s =  f"Requestor: {user}@princeton.edu\n\n"
                    s += f"{get_first_name(user, formal=True)},\n\n"
                    s += f'Over the past {dt.days+1} days you were sent {num_warnings} emails with the subject "Requesting Too\n'
                    s += f'Much CPU Memory". The data associated with these emails is shown below:\n\n'
                    cols = ["NetID",
                            "cluster",
                            "partition",
                            "mem-hrs-unused",
                            "mem-hrs-used",
                            "Week-Ending-On"]
                    renamings = {"cluster":"Cluster",
                                 "partition":"Partition",
                                 "mem-hrs-unused":"Mem-Unused",
                                 "mem-hrs-used":"Mem-Used",
                                 "email_sent":"Email Sent"}
                    usr = usr[cols].rename(columns=renamings)
                    usr["Mem-Unused"] = usr["Mem-Unused"].apply(lambda x: f"{x} TB-hours")
                    usr["Mem-Used"]   = usr["Mem-Used"].apply(lambda x: f"{x} TB-hours")
                    usr_str = usr.to_string(index=False, justify="center", col_space=14)
                    s +=  "\n".join([1 * " " + row for row in usr_str.split("\n")])
                    s += "\n"
                    s += textwrap.dedent(f"""
                    Each row in the table above represents a 1-week period. Your Slurm allocations
                    have resulted in a total of {tb_hrs_unused} TB-hours of unused memory which is equivalent to
                    making {num_wasted_nodes} nodes unavailable to all users (including yourself) for one week!

                    At this time you need to either (1) stop allocating excessive CPU memory or
                    (2) reply to this support ticket with an explanation of why this is necessary for
                    your work.

                    We will be forced to prevent you from running jobs if no action is taken.
                    """)
                    send_email_cses(s,      "cses@princeton.edu", subject=f"{self.subject}", sender="jdh4@princeton.edu")
                    send_email_cses(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                    print(s)

                    # append the new violations to the log file
                    usr = pd.DataFrame({"netid":[user],
                                        "tb_hours_unused":[tb_hrs_unused],
                                        "tb_hours_used":[tb_hrs_used],
                                        "num_warnings":[num_warnings]})
                    Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        """There is no report for this alert."""
        return ""
