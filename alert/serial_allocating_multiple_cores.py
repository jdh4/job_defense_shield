import pandas as pd
from base import Alert
from utils import add_dividers
from utils import MINUTES_PER_HOUR as mph
from efficiency import cpu_efficiency
from greeting import GreetingFactory
from email_translator import EmailTranslator


class SerialAllocatingMultipleCores(Alert):

    """Find serial codes that are using multiple CPU-cores."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        self.excluded_users = []
        self.num_top_users = 0
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition.isin(self.partitions)) &
                          (self.df.nodes == 1) &
                          (self.df.cores > 1) &
                          (~self.df.user.isin(self.excluded_users)) &
                          (self.df["elapsed-hours"] >= self.min_run_time / mph)].copy()
        if self.include_running_jobs:
            self.df.admincomment = Alert.get_admincomment_for_running_jobs(self)
        self.df = self.df[self.df.admincomment != {}]
        self.gp = pd.DataFrame({"User":[]})
        if not self.df.empty:
            # add new fields
            self.df["cpu-eff-tpl"] = self.df.apply(lambda row:
                                                   cpu_efficiency(row["admincomment"],
                                                                  row["elapsedraw"],
                                                                  row["jobid"],
                                                                  row["cluster"],
                                                                  single=True,
                                                                  precision=1),
                                                                  axis="columns")
            self.df["error-code"] = self.df["cpu-eff-tpl"].apply(lambda tpl: tpl[1])
            # drop jobs with non-zero error codes
            self.df = self.df[self.df["error-code"] == 0]
            self.df["cpu-eff"] = self.df["cpu-eff-tpl"].apply(lambda tpl: tpl[0])
            # ignore jobs at 0% CPU-eff (also avoids division by zero later)
            self.df = self.df[self.df["cpu-eff"] >= 1]
            # max efficiency if serial is 100% / cores
            self.df["inverse-cores"] = 100 / self.df["cores"]
            self.df["inverse-cores"] = self.df["inverse-cores"].apply(lambda x: round(x, 1))
            self.df["ratio"] = self.df["cpu-eff"] / self.df["inverse-cores"]
            self.df = self.df[(self.df["ratio"] <= 1) &
                              (self.df["ratio"] > self.lower_ratio)]
            renamings = {"elapsed-hours":"Hours",
                         "jobid":"JobID",
                         "user":"User",
                         "partition":"Partition",
                         "cores":"CPU-cores",
                         "cpu-eff":"CPU-Util",
                         "inverse-cores":"100%/CPU-cores"}
            self.df = self.df.rename(columns=renamings)
            self.df = self.df[["JobID",
                               "User",
                               "Partition",
                               "CPU-cores",
                               "CPU-Util",
                               "100%/CPU-cores",
                               "Hours"]]
            self.df = self.df.sort_values(by=["User", "JobID"])
            self.df["100%/CPU-cores"] = self.df["100%/CPU-cores"].apply(lambda x: f"{x}%")
            self.df["CPU-Util"] = self.df["CPU-Util"].apply(lambda x: f"{x}%")
            self.df["cores-minus-1"] = self.df["CPU-cores"] - 1
            self.df["CPU-Hours-Wasted"] = self.df["Hours"] * self.df["cores-minus-1"]
            def jobid_list(series):
                ellipsis = "+" if len(series) > self.max_num_jobid_admin else ""
                return ",".join(series[:self.max_num_jobid_admin]) + ellipsis
            d = {"CPU-Hours-Wasted":"sum",
                 "User":"size",
                 "CPU-cores":"mean",
                 "JobID":jobid_list}
            self.gp = self.df.groupby("User").agg(d).rename(columns={"User":"Jobs"})
            self.gp.reset_index(drop=False, inplace=True)
            self.gp = self.gp[self.gp["CPU-Hours-Wasted"] > self.cpu_hours_threshold]
            if self.num_top_users:
                self.gp = self.gp.head(self.num_top_users)

    def create_emails(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.gp.User.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.User == user].copy()
                cpu_hours_wasted = usr["CPU-Hours-Wasted"].sum()
                #cpu_hours_wasted = self.gp[self.gp.User == user]["CPU-Hours-Wasted"].values[0]
                usr = usr.drop(columns=["User", "cores-minus-1", "CPU-Hours-Wasted"])
                usr["Hours"] = usr["Hours"].apply(lambda hrs: round(hrs, 1))
                num_disp = self.num_jobs_display
                total_jobs = usr.shape[0]
                case = f"{num_disp} of your {total_jobs} jobs" if total_jobs > num_disp else "your jobs"
                # CHANGE NEXT
                hours_per_week = 24 * 7
                num_wasted_nodes = round(cpu_hours_wasted / self.cores_per_node / hours_per_week)
                # create new tag which is two sentences
                # if cores_per_node:
                #Your jobs allocated <CPU-HOURS> CPU-hours that were never used. This is equivalent to
                #making <NUM-NODES> nodes unavailable to all users (including yourself) for 1 week!
                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<CASE>"] = case
                tags["<CLUSTER>"] = self.cluster
                tags["<PARTITIONS>"] = ",".join(sorted(set(usr.Partition)))
                tags["<DAYS>"] = str(self.days_between_emails)
                indent = 4 * " "
                # ADDED num_disp but no sort before this
                table = usr.head(num_disp).to_string(index=False, justify="center").split("\n")
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                tags["<CPU-HOURS>"] = str(cpu_hours_wasted)
                tags["<NUM-NODES>"] = str(num_wasted_nodes)
                translator = EmailTranslator(self.email_file, tags)
                email = translator.replace_tags()
                self.emails.append((user, email, usr))

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.gp.empty:
            column_names = ["User",
                            "CPU-Hours-Wasted",
                            "AvgCores",
                            "Jobs",
                            "JobID",
                            "Emails"]
            self.gp = pd.DataFrame(columns=column_names)
            return add_dividers(self.create_empty_report(self.gp), title)
        self.gp["CPU-Hours-Wasted"] = self.gp["CPU-Hours-Wasted"].apply(round)
        self.gp["CPU-cores"] = self.gp["CPU-cores"].apply(lambda x:
                                                    str(round(x, 1)).replace(".0", ""))
        self.gp = self.gp.rename(columns={"CPU-cores":"AvgCores"})
        self.gp.reset_index(drop=False, inplace=True)
        self.gp["emails"] = self.gp.User.apply(lambda user:
                                 self.get_emails_sent_count(user, self.violation))
        self.gp.emails = self.format_email_counts(self.gp.emails)
        cols = ["User", "CPU-Hours-Wasted", "AvgCores", "Jobs", "JobID", "emails"]
        self.gp = self.gp[cols]
        self.gp = self.gp.sort_values(by="CPU-Hours-Wasted", ascending=False)
        self.gp.reset_index(drop=True, inplace=True)
        self.gp.index += 1
        return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
