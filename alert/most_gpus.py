from base import Alert
from utils import add_dividers
from efficiency import gpu_efficiency
from utils import JOBSTATES


class MostGPUs(Alert):

    """Top 10 users by the highest number of allocated GPUs in a job. Only
     one job per user is shown."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        cols = ["jobid", "user", "cluster", "gpus", "nodes", "cores", "state",
              "partition", "elapsed-hours", "admincomment", "elapsedraw"]
        self.gp = self.df[cols].groupby("user").apply(lambda d: d.iloc[d["gpus"].argmax()])
        self.gp = self.gp.sort_values("gpus", ascending=False)[:10]
        self.gp = self.gp.rename(columns={"elapsed-hours":"hours"})
        self.gp.state = self.gp.state.apply(lambda x: JOBSTATES[x])
        self.gp["GPU-eff-tpl"] = self.gp.apply(lambda row:
                                             gpu_efficiency(row["admincomment"],
                                                            row["elapsedraw"],
                                                            row["jobid"],
                                                            row["cluster"], single=True)
                                             if row["admincomment"] != {} else ("--", 0), axis="columns")
        self.gp["GPU-eff"] = self.gp["GPU-eff-tpl"].apply(lambda tpl: tpl[0])
        self.gp["GPU-eff"] = self.gp["GPU-eff"].apply(lambda x: x if x == "--" else f"{round(x)}%")
        self.gp["hours"] = self.gp["hours"].apply(lambda hrs: round(hrs, 1))
        cols = ["jobid", "user", "cluster", "gpus", "nodes", "cores", "state",
              "partition", "hours", "GPU-eff"]
        self.gp = self.gp[cols]

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.gp.empty:
            return add_dividers(self.create_empty_report(self.gp), title)
        return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
