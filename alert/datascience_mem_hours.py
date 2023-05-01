import os
import textwrap
from datetime import datetime
from datetime import timedelta
from base import Alert
from efficiency import cpu_memory_usage
from utils import JOBSTATES
from utils import add_dividers
import numpy as np
import pandas as pd


class DataScienceMemoryHours(Alert):

    """Cumulative memory use of the datascience nodes."""

    def __init__(self, df, days_between_emails, violation, vpath, subject):
        super().__init__(df, days_between_emails, violation, vpath, subject)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == "della") &
                          (self.df.partition == "datasci") &
                          (self.df.admincomment != {}) &
                          (self.df.state != "RUNNING") &
                          (self.df.state != "OUT_OF_MEMORY") &
                          (self.df["elapsed-hours"] >= 1)].copy()
        # add new fields
        if not self.df.empty:
            self.df["memory-tuple"] = self.df.apply(lambda row:
                                                    cpu_memory_usage(row["admincomment"],
                                                                     row["jobid"],
                                                                     row["cluster"]),
                                                                     axis="columns")
            self.df["mem-used"]  = self.df["memory-tuple"].apply(lambda x: x[0])
            self.df["mem-alloc"] = self.df["memory-tuple"].apply(lambda x: x[1])
            GB_per_TB = 1000
            self.df["mem-hrs-used"]   = self.df["mem-used"] * self.df["elapsed-hours"] / GB_per_TB
            self.df["mem-hrs-alloc"]  = self.df["mem-alloc"] * self.df["elapsed-hours"] / GB_per_TB
            self.df["mem-hrs-unused"] = self.df["mem-hrs-alloc"] - self.df["mem-hrs-used"]
            self.df["median-ratio"]   = self.df["mem-hrs-used"] / self.df["mem-hrs-alloc"]
            self.df.state = self.df.state.apply(lambda x: JOBSTATES[x])
            # compute various quantities by grouping by user
            d = {"mem-hrs-used":np.sum,
                 "mem-hrs-alloc":np.sum,
                 "mem-hrs-unused":np.sum,
                 "elapsed-hours":np.sum,
                 "cpu-hours":np.sum,
                 "account":pd.Series.mode,
                 "median-ratio":"median",
                 "netid":np.size}
            self.gp = self.df.groupby("netid").agg(d)
            self.gp = self.gp.rename(columns={"netid":"jobs"})
            total_mem_hours = max(1, self.gp["mem-hrs-alloc"].sum())
            self.gp["proportion"] = self.gp["mem-hrs-alloc"] / total_mem_hours
            self.gp["ratio"] = self.gp["mem-hrs-used"] / self.gp["mem-hrs-alloc"]
            self.gp.reset_index(drop=False, inplace=True)
            self.gp = self.gp.sort_values("mem-hrs-unused", ascending=False)
            cols = ["netid",
                    "account",
                    "proportion",
                    "mem-hrs-unused",
                    "mem-hrs-used",
                    "mem-hrs-alloc",
                    "ratio",
                    "median-ratio",
                    "elapsed-hours",
                    "cpu-hours",
                    "jobs"]
            self.gp = self.gp[cols]
            renamings = {"elapsed-hours":"hrs", "cpu-hours":"cpu-hrs"}
            self.gp = self.gp.rename(columns=renamings)
            self.gp["emails"] = self.gp["netid"].apply(self.get_emails_sent_count)
            cols = ["mem-hrs-unused", "mem-hrs-used", "mem-hrs-alloc", "cpu-hrs"]
            self.gp[cols] = self.gp[cols].apply(round).astype("int64")
            cols = ["proportion", "ratio", "median-ratio"]
            self.gp[cols] = self.gp[cols].apply(lambda x: round(x, 2))
            self.gp.reset_index(drop=True, inplace=True)
            self.gp.index += 1

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
        pass

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.gp.empty:
            return ""
        else:
            cols = ["netid",
                    "account",
                    "proportion",
                    "mem-hrs-unused",
                    "ratio",
                    "median-ratio",
                    "elapsed-hours",
                    "jobs",
                    "emails"]
            self.gp = self.gp[cols].head(5)
            return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
