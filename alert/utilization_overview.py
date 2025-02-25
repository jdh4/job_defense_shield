from base import Alert
from utils import add_dividers


class UtilizationOverview(Alert):

    """Utilization of the clusters by partition."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        def compute_utilization(fields: list, simple: bool=True):
            """Aggregate the data based on fields."""
            d = {"user":lambda series: series.unique().size,
                 "cpu-hours":"sum",
                 "gpu-hours":"sum"}
            gp = self.df.groupby(fields).agg(d)
            gp = gp.rename(columns={"user":"users"})
            gp = gp.reset_index().sort_values(by=["cluster", "cpu-hours"],
                                              ascending=[True, False])
            cols = ["cpu-hours", "gpu-hours"]
            gp[cols] = gp[cols].apply(round).astype("int64")
            if simple:
                for column in ["cpu-hours", "gpu-hours"]:
                    total = gp[column].sum()
                    if total != 0:
                        gp[column] = gp[column].apply(lambda x:
                                                      f"{x} ({round(100 * x / total)}%)")
            else:
                for cluster in gp["cluster"].unique():
                    for field in ["cpu-hours", "gpu-hours"]:
                        total = gp[gp["cluster"] == cluster][field].sum()
                        if total != 0:
                            gp[field] = gp.apply(lambda row:
                                                 f"{row[field]} ({round(100 * row[field] / total)}%)"
                                                 if row["cluster"] == cluster
                                                 else row[field], axis="columns")
            return gp

        self.df = self.df[self.df["elapsedraw"] > 0].copy()
        self.by_cluster = compute_utilization(["cluster"])
        self.by_partition = compute_utilization(["cluster", "partition"], simple=False)
        
        # add max usage for specific partitions (this code to be removed)
        period_hours = 24 * self.days_between_emails
        self.special = self.by_partition.copy()
        if 0:
            self.special["Usage(%)"] = -1
            self.special = self.special.set_index(["cluster", "partition"])
            # gpu usage on della gpu
            gpu = self.special.at[("della", "gpu"), "gpu-hours"]
            gpu = int(gpu.split("(")[0].strip())
            gpu = round(100 * gpu / 364 / period_hours)
            self.special.at[("della", "gpu"), "Usage(%)"] = gpu
            # gpu usage on della mig
            mig = self.special.at[("della", "mig"), "gpu-hours"]
            mig = int(mig.split("(")[0].strip())
            mig = round(100 * mig / 56 / period_hours)
            self.special.at[("della", "mig"), "Usage(%)"] = mig
            # gpu usage on della cli
            pli = self.special.at[("della", "pli"), "gpu-hours"] 
            plic = self.special.at[("della", "pli-c"), "gpu-hours"]
            cli = int(pli.split("(")[0].strip()) + int(plic.split("(")[0].strip())
            cli = round(100 * cli / 296 / period_hours)
            self.special.at[("della", "pli"), "Usage(%)"] = cli
            self.special.at[("della", "pli-c"), "Usage(%)"] = cli
            # gpu usage on stellar gpu
            gpu = self.special.at[("stellar", "gpu"), "gpu-hours"]
            gpu = int(gpu.split("(")[0].strip())
            gpu = round(100 * gpu / 12 / period_hours)
            self.special.at[("stellar", "gpu"), "Usage(%)"] = gpu
            # gpu usage on traverse
            tra = self.special.at[("traverse", "all"), "gpu-hours"]
            tra = int(tra.split("(")[0].strip())
            tra = round(100 * tra / 184 / period_hours)
            self.special.at[("traverse", "all"), "Usage(%)"] = tra

    def create_emails(self):
        """There are no emails for this alert."""
        pass

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        clus = self.by_cluster.to_string(index=keep_index, justify="center")
        part = self.special.to_string(index=False, justify="center")
        return add_dividers(clus, title) + add_dividers(part, f"{title} by Partition")
