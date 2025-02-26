import os
import sys
from datetime import datetime
from abc import abstractmethod
import pandas as pd
from utils import send_email
from utils import SECONDS_PER_HOUR
from utils import HOURS_PER_DAY


class Alert:

    """Base class for all alerts."""

    def __init__(self,
                 df: pd.DataFrame,
                 days_between_emails: int,
                 violation: str,
                 vpath: str,
                 subject: str,
                 **props: dict) -> None:
        self.df = df
        self.days_between_emails = days_between_emails
        self.violation = violation
        self.vpath = vpath
        self.vbase = os.path.join(self.vpath, self.violation)
        self.subject = subject
        self.emails = []
        for key in props:
            setattr(self, key, props[key])
        self._filter_and_add_new_fields()
        # create directory to store user violations
        if self.vbase and not os.path.exists(self.vbase):
            os.mkdir(self.vbase)

    @abstractmethod
    def _filter_and_add_new_fields(self) -> None:
        """Filter the dataframe and add new fields.

        Returns:
            None
        """

    @abstractmethod
    def create_emails(self) -> None:
        """Create emails to the users.

        Returns:
            None
        """

    def send_emails_to_users(self) -> None:
        """Send emails to users."""
        for user, email, usr in self.emails:
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            send_email(email,
                       f"{user}@{self.email_domain}",
                       subject=self.subject,
                       sender=self.sender,
                       reply_to=self.reply_to)
            self.update_violation_log(usr, vfile)
            for admin_email in self.admin_emails:
                send_email(email,
                           admin_email,
                           subject=self.subject,
                           sender=self.sender,
                           reply_to=self.reply_to)
            print(email)

    @abstractmethod
    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        """Generate a report for system administrators.

        Returns:
            A table of data as a string.
        """

    def get_admincomment_for_running_jobs(self) -> pd.Series:
        """Query the Prometheus server for the admincomment of
        jobs in a RUNNING state."""
        sys.path.append(self.jobstats_path)
        from jobstats import Jobstats
        from config import PROM_SERVER
        num_jobs = len(self.df[self.df.state == "RUNNING"])
        print(f"\nQuerying server for admincomment on {num_jobs} running jobs ... ",
              end="",
              flush=True)
        adminc = self.df.apply(lambda row:
                               eval(Jobstats(jobid=row["jobid"],
                                                   cluster=row["cluster"],
                                                   prom_server=PROM_SERVER).report_job_json(False))
                               if row["state"] == "RUNNING"
                               else row["admincomment"], axis="columns")
        print("done.", flush=True)
        return adminc

    def has_sufficient_time_passed_since_last_email(self, vfile: str) -> bool:
        """Return boolean specifying whether sufficient time has passed."""
        last_sent_email_date = datetime(1970, 1, 1)
        if os.path.exists(vfile):
            violation_history = pd.read_csv(vfile,
                                            parse_dates=["email_sent"],
                                            date_format="mixed",
                                            dayfirst=False)
            last_sent_email_date = violation_history["email_sent"].max()
        seconds_since_last_email = datetime.now().timestamp() - last_sent_email_date.timestamp()
        seconds_threshold = self.days_between_emails * HOURS_PER_DAY * SECONDS_PER_HOUR
        return seconds_since_last_email >= seconds_threshold

    def get_emails_sent_count(self, user: str, violation: str) -> str:
        """Return the number of emails sent to a user for a given violation in the
           last N days."""
        root_violations = f"{self.vpath}/{violation}"
        if not os.path.exists(root_violations):
            print(f"Warning: {root_violations} not found in get_emails_sent_count()")
        user_violations = f"{root_violations}/{user}.email.csv"
        if os.path.exists(user_violations):
            d = pd.read_csv(user_violations,
                            parse_dates=["email_sent"],
                            date_format="mixed",
                            dayfirst=False)
            num_emails_sent = d["email_sent"].unique().size
            dt = datetime.now() - d["email_sent"].unique().max()
            days_ago_last_email_sent = round(dt.total_seconds() / 24 / 3600)
            return f"{num_emails_sent} ({days_ago_last_email_sent})"
        return "0 (-)"

    def format_email_counts(self, counts: pd.Series) -> pd.Series:
        """Return the email sent counts with proper alignment of the two
           different quantities."""
        if counts.empty:
            return counts
        if counts.tolist() == ["0 (-)"] * len(counts):
            return pd.Series(["0"] * len(counts))
        max_len = max([len(count.split()[1]) for count in counts.tolist()])
        def fix_spacing(item: str):
            num_sent, days_ago = item.split()
            pair = f"{num_sent}{(max_len - len(days_ago)) * ' '} {days_ago}"
            return pair.replace("(-)", "   ")
        return counts.apply(fix_spacing)

    @staticmethod
    def update_violation_log(usr: pd.DataFrame, vfile: str) -> None:
        """Append the new violations to file."""
        usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        if os.path.exists(vfile):
            curr = pd.read_csv(vfile)
            curr = pd.concat([curr, usr]).drop_duplicates()
            curr.to_csv(vfile, index=False, header=True)
        else:
            usr.to_csv(vfile, index=False, header=True)

    def __len__(self) -> int:
        """Returns the number of rows in the df dataframe."""
        return self.df.shape[0]

    def __str__(self) -> str:
        """Returns the df dataframe as a string."""
        return self.df.to_string()
