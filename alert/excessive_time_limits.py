import pandas as pd
from base import Alert
from utils import SECONDS_PER_MINUTE as spm
from utils import MINUTES_PER_HOUR as mph
from utils import seconds_to_slurm_time_format
from utils import add_dividers
from greeting import GreetingFactory
from email_translator import EmailTranslator


class ExcessiveTimeLimits(Alert):

    """Over-allocating run time."""

    def __init__(self, df, days_between_emails, violation, vpath, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, **kwargs)

    def _add_required_fields(self):
        if not hasattr(self, "email_subject"):
            self.email_subject = "Requesting Too Much Time for Jobs"
        if not hasattr(self, "report_title"):
            self.report_title = "Excessive Time Limits"

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition.isin(self.partitions)) &
                          (self.df.state == "COMPLETED") &
                          (~self.df.user.isin(self.excluded_users)) &
                          (self.df["elapsed-hours"] >= self.min_run_time / mph)].copy()
        self.gp = pd.DataFrame({"User":[]})
        # add new fields
        if not self.df.empty:
            xpu = "cpu"
            self.df["ratio"] = 100 * self.df[f"{xpu}-hours"] / self.df[f"{xpu}-alloc-hours"]
            d = {f"{xpu}-waste-hours":"sum",
                 f"{xpu}-alloc-hours":"sum",
                 f"{xpu}-hours":"sum",
                 "user":"size",
                 "partition":lambda series: ",".join(sorted(set(series))),
                 "ratio":"median"}
            self.gp = self.df.groupby("user").agg(d).rename(columns={"user":"jobs", "ratio":"median(%)"})
            self.gp = self.gp.sort_values(by=f"{xpu}-hours", ascending=False).reset_index(drop=False)
            self.gp["rank"] = self.gp.index + 1
            self.gp = self.gp.sort_values(by=f"{xpu}-waste-hours", ascending=False).reset_index(drop=False)
            self.gp.index += 1
            self.gp[f"{xpu}-hours"] = self.gp[f"{xpu}-hours"].apply(round).astype("int64")
            self.gp[f"{xpu}-waste-hours"] = self.gp[f"{xpu}-waste-hours"].apply(round)
            self.gp[f"{xpu}-alloc-hours"] = self.gp[f"{xpu}-alloc-hours"].apply(round)
            self.gp["mean(%)"] = 100 * self.gp[f"{xpu}-hours"] / self.gp[f"{xpu}-alloc-hours"]
            self.gp["mean(%)"] = self.gp["mean(%)"].apply(round).astype("int64")
            self.gp["median(%)"] = self.gp["median(%)"].apply(round).astype("int64")
            self.gp = self.gp[(self.gp[f"{xpu}-waste-hours"] > self.absolute_thres_hours) &
                              (self.gp["mean(%)"] < self.mean_ratio_threshold) &
                              (self.gp["median(%)"] < self.median_ratio_threshold)]
            # how to deal with this?
            # hasattr()
            if self.num_top_users:
                self.gp = self.gp[self.gp["rank"] <= self.num_top_users]
            cols = ["user",
                    f"{xpu}-waste-hours",
                    f"{xpu}-hours",
                    f"{xpu}-alloc-hours",
                    "mean(%)",
                    "median(%)",
                    "rank",
                    "jobs",
                    "partition"]
            self.gp = self.gp[cols]
            self.gp["cluster"] = self.cluster
            renamings = {"user":"User",
                         f"{xpu}-waste-hours":f"{xpu.upper()}-Hours-Unused"}
            self.gp = self.gp.rename(columns=renamings)

    def create_emails(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.gp.User.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.gp[self.gp.User == user].copy()
                jobs = self.df[self.df.user == user].copy()
                xpu = "cpu"
                num_disp = self.num_jobs_display
                total_jobs = jobs.shape[0]
                case = f"{num_disp} of your {total_jobs} jobs" if total_jobs > num_disp else "your jobs"
                jobs = jobs.sort_values(by=f"{xpu}-waste-hours", ascending=False).head(num_disp)
                jobs["Time-Used"] = jobs["elapsedraw"].apply(seconds_to_slurm_time_format)
                jobs["Time-Allocated"] = jobs["limit-minutes"].apply(lambda x:
                                                                     seconds_to_slurm_time_format(spm * x))
                jobs["Percent-Used"] = jobs["ratio"].apply(lambda x: f"{round(x)}%")
                cols = ["jobid", "user", "Time-Used", "Time-Allocated", "Percent-Used", "cores"]
                jobs = jobs[cols].sort_values(by="jobid")
                renamings = {"jobid":"JobID", "user":"User", "cores":"CPU-Cores"}
                jobs = jobs.rename(columns=renamings)

                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<CASE>"] = case
                tags["<DAYS>"] = str(self.days_between_emails)
                tags["<CLUSTER>"] = self.cluster
                tags["<PARTITIONS>"] = ",".join(sorted(set(usr.partition)))
                tags["<AVERAGE>"] = str(usr["mean(%)"].values[0])
                tags["<NUM-JOBS>"] = str(total_jobs)
                tags["<NUM-JOBS-DISPLAY>"] = str(total_jobs)
                indent = 4 * " "
                table = jobs.to_string(index=False, justify="center").split("\n")
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                tags["<UNUSED-HOURS>"] = str(round(usr[f"{xpu.upper()}-Hours-Unused"].values[0]))
                translator = EmailTranslator(self.email_files_path,
                                             self.email_file,
                                             tags)
                email = translator.replace_tags()
                self.emails.append((user, email, usr))

    def generate_report_for_admins(self, keep_index: bool=False) -> str:
        """Generate a table for system administrators."""
        if self.gp.empty:
            column_names = ["User",
                            "CPU-Hours-Unused",
                            "CPU-Hours",
                            "CPU-Alloc-Hours",
                            "Mean(%)",
                            "Median(%)",
                            "Rank",
                            "Jobs"]
            self.gp = pd.DataFrame(columns=column_names)
            return add_dividers(self.create_empty_report(self.gp), self.report_title)
        self.gp = self.gp.drop(columns=["cluster", "partition"])
        xpu = "cpu"
        renamings = {f"{xpu}-hours":"used",
                     f"{xpu}-alloc-hours":"total"}
        self.gp = self.gp.rename(columns=renamings)
        self.gp["emails"] = self.gp.User.apply(lambda user:
                                 self.get_emails_sent_count(user, self.violation))
        self.gp.emails = self.format_email_counts(self.gp.emails)
        report_str = self.gp.to_string(index=keep_index, justify="center")
        return add_dividers(report_str, self.report_title)
