from base import Alert
from utils import add_dividers
from utils import SECONDS_PER_HOUR


class JobsOverview(Alert):

    """Users with the most jobs. Only non-running jobs with greater than
       zero elapsed seconds are considered."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        self.df = self.df[self.df["elapsedraw"] > 0].copy()
        cols = ["jobid",
                "user",
                "cluster",
                "cores",
                "state",
                "partition",
                "cpu-seconds",
                "gpu-seconds",
                "gpu-job"]
        self.df = self.df[cols]
        # add new fields
        self.df.state  = self.df.state.apply(lambda s: "CANCELLED" if "CANCEL" in s else s)
        self.df["CLD"] = self.df.state.apply(lambda s: s == "CANCELLED")
        self.df["COM"] = self.df.state.apply(lambda s: s == "COMPLETED")
        self.df["OOM"] = self.df.state.apply(lambda s: s == "OUT_OF_MEMORY")
        self.df["TO"]  = self.df.state.apply(lambda s: s == "TIMEOUT")
        self.df["F"]   = self.df.state.apply(lambda s: s == "FAILED")
        d = {"user":"size",
             "COM":"sum",
             "CLD":"sum",
             "F":"sum",
             "OOM":"sum",
             "TO":"sum",
             "cpu-seconds":"sum",
             "gpu-seconds":"sum",
             "gpu-job":"sum",
             "partition":lambda series: ",".join(sorted(set(series)))}
        self.gp = self.df.groupby(["cluster", "user"]).agg(d)
        self.gp = self.gp.rename(columns={"user":"jobs"})
        self.gp = self.gp.reset_index(drop=False)
        self.gp = self.gp.sort_values("jobs", ascending=False)
        self.gp = self.gp.rename(columns={"partition":"partitions", "gpu-job":"gpu"})
        self.gp["cpu"] = self.gp["jobs"] - self.gp["gpu"]
        self.gp["cpu-hours"] = self.gp["cpu-seconds"] / SECONDS_PER_HOUR
        self.gp["gpu-hours"] = self.gp["gpu-seconds"] / SECONDS_PER_HOUR
        self.gp["cpu-hours"] = self.gp["cpu-hours"].apply(round)
        self.gp["gpu-hours"] = self.gp["gpu-hours"].apply(round)
        self.gp.drop(columns=["cpu-seconds", "gpu-seconds"], inplace=True)
        cols = ["user",
                "cluster",
                "jobs",
                "cpu",
                "gpu",
                "COM",
                "CLD",
                "F",
                "OOM",
                "TO",
                "cpu-hours",
                "gpu-hours",
                "partitions"]
        self.gp = self.gp[cols]

    def send_emails_to_users(self):
        """There are no emails for this alert."""
        pass

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        return add_dividers(self.gp.head(10).to_string(index=keep_index, justify="center"), title)
