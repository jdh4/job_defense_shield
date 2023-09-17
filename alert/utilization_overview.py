from base import Alert
from utils import add_dividers


class UtilizationOverview(Alert):

    """Utilization of the clusters by partition."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    def _filter_and_add_new_fields(self):
        def compute_utilization(fields: list, simple: bool=True):
            """Aggregate the data based on fields."""
            d = {"netid":lambda series: series.unique().size,
                 "cpu-hours":"sum",
                 "gpu-hours":"sum"}
            gp = self.df.groupby(fields).agg(d)
            renamings = {"netid":"users"}
            gp = gp.rename(columns=renamings)
            gp = gp.sort_index().reset_index()
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

    def send_emails_to_users(self):
        """There are no emails for this alert."""
        pass

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        clus = self.by_cluster.to_string(index=keep_index, justify="center")
        part = self.by_partition.to_string(index=keep_index, justify="center")
        return add_dividers(clus, title) + add_dividers(part, f"{title} by Partition")
