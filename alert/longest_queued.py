import time
from utils import SECONDS_PER_HOUR
from utils import HOURS_PER_DAY
from utils import add_dividers
from base import Alert


class LongestQueuedJobs(Alert):

    """Find the pending jobs with the longest queue times while ignoring
       array jobs."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[self.df.state == "PENDING"].copy()
        # remove array jobs
        self.df = self.df[~self.df.jobid.str.contains("_")]
        # add new fields
        self.df["s-days"] = round((time.time() - self.df["submit"])   / SECONDS_PER_HOUR / HOURS_PER_DAY)
        self.df["e-days"] = round((time.time() - self.df["eligible"]) / SECONDS_PER_HOUR / HOURS_PER_DAY)
        self.df["s-days"] = self.df["s-days"].astype("int64")
        self.df["e-days"] = self.df["e-days"].astype("int64")
        cols = ["jobid", "user", "cluster", "nodes", "cores", "qos", "partition", "s-days", "e-days"]
        self.df = self.df[cols].groupby("user").apply(lambda d: d.iloc[d["s-days"].argmax()])
        self.df.sort_values("s-days", ascending=False, inplace=True)
        self.df = self.df[self.df["s-days"] >= 4][:10]

    def create_emails(self):
        pass

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
