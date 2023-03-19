import os
from datetime import datetime
from utils import SECONDS_PER_HOUR
from utils import HOURS_PER_DAY
from abc import abstractmethod
import pandas as pd

class Alert:

    """Base class for all alerts."""

    def __init__(self,
                 df: pd.DataFrame,
                 days_between_emails: int,
                 violation: str,
                 vpath: str,
                 subject: str) -> None:
        self.df = df
        self.days_between_emails = days_between_emails
        self.violation = violation
        self.vpath = vpath
        self.vbase = os.path.join(self.vpath, self.violation)
        self.subject = subject
        self._filter_and_add_new_fields()
        # create directory to store user violations
        if not os.path.exists(self.vbase):
            os.mkdir(self.vbase)

    @abstractmethod
    def _filter_and_add_new_fields(self) -> None:
        """Filter the dataframe and add new fields.

        Returns:
            None
        """

    @abstractmethod
    def send_emails_to_users(self) -> None:
        """Send emails to the users.

        Returns:
            None
        """

    @abstractmethod
    def generate_report_for_admins(self, title: str, keep_index: bool=True) -> str:
        """Generate a report for system administrators.

        Returns:
            A table of data as a string.
        """

    def has_sufficient_time_passed_since_last_email(self, vfile: str) -> bool:
        """Return boolean specifying whether sufficient time has passed."""
        last_write_date = datetime(1970, 1, 1)
        if os.path.exists(vfile):
            last_write_date = datetime.fromtimestamp(os.path.getmtime(vfile))
        seconds_since_last_email = datetime.now().timestamp() - last_write_date.timestamp()
        seconds_threshold = self.days_between_emails * HOURS_PER_DAY * SECONDS_PER_HOUR
        return seconds_since_last_email >= seconds_threshold

    @staticmethod
    def update_violation_log(usr: pd.DataFrame, vfile: str) -> None:
        """Append the new violations to file."""
        usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M")
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
