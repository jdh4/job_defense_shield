from base import Alert
from utils import send_email
from utils import add_dividers
from utils import MINUTES_PER_HOUR as mph
from greeting import GreetingFactory
from email_translator import EmailTranslator


class TooManyCoresPerGpu(Alert):

    """Find jobs that using too many CPU-cores per GPU."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == self.cluster) &
                          (self.df.partition.isin(self.partitions)) &
                          (self.df.gpus > 0) &
                          (self.df.cores > self.cores_per_gpu_limit * self.df.gpus) &
                          (~self.df.user.isin(self.excluded_users)) &
                          (self.df["elapsed-hours"] >= self.min_run_time / mph)].copy()
        self.df.rename(columns={"user":"User"}, inplace=True)
        # add new fields
        if not self.df.empty:
            self.df["Cores-per-GPU"] = self.df.cores / self.df.gpus
            self.df["Cores-per-GPU"] = self.df["Cores-per-GPU"].apply(lambda x:
                                       str(round(x, 1)).replace(".0", ""))
            self.df["Cores-per-GPU-Target"] = self.cores_per_gpu_target
            cols = ["jobid",
                    "User",
                    "elapsed-hours",
                    "cores",
                    "gpus",
                    "Cores-per-GPU",
                    "Cores-per-GPU-Target"]
            self.df = self.df[cols]
            renamings = {"jobid":"JobID",
                         "cores":"Cores",
                         "gpus":"GPUs",
                         "elapsed-hours":"Hours"}
            self.df = self.df.rename(columns=renamings)

    def send_emails_to_users(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.df.User.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.User == user].copy()
                usr["Hours"] = usr["Hours"].apply(lambda hrs: round(hrs, 1))
                indent = 3 * " "
                table = usr.to_string(index=False, justify="center").split("\n")
                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<CLUSTER>"] = self.cluster
                tags["<NAME>"] = self.cluster_name
                tags["<TARGET>"] = str(self.cores_per_gpu_target)
                tags["<CORES>"] = str(self.cores_per_node)
                tags["<GPUS>"] = str(self.gpus_per_node)
                tags["<DAYS>"] = str(self.days_between_emails)
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                tags["<PARTITIONS>"] = ",".join(self.partitions)
                translator = EmailTranslator(self.email_file, tags)
                s = translator.replace_tags()

                send_email(s, f"{user}@princeton.edu", subject=f"{self.subject}")
                for email in self.admin_emails:
                    send_email(s, f"{email}", subject=f"{self.subject}")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
