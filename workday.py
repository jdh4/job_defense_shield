"""Classes to decide whether or not to send emails on holidays."""

from datetime import datetime
from abc import ABC, abstractmethod
from pandas.tseries.holiday import USFederalHolidayCalendar


class Workday(ABC):

    """Abstract base class for deciding on workdays."""

    @abstractmethod
    def is_workday(self) -> bool:
        pass


class UsaWorkday(Workday):

    """Do not send emails on US holidays and local holidays."""

    def is_workday(self) -> bool:
        """Determine if today is a work day."""
        date_today = datetime.now().strftime("%Y-%m-%d")
        cal = USFederalHolidayCalendar()
        us_holiday = bool(date_today in cal.holidays())
        local_holidays = ["2024-12-26", "2024-12-27", "2025-01-20",
                          "2025-06-19", "2025-11-28", "2025-12-26",
                          "2026-01-02", "2026-01-19", "2026-06-19",
                          "2026-11-27", "2026-12-24", "2026-12-31",
                          "2027-01-18", "2027-06-18"]
        local_holiday = bool(date_today in local_holidays)
        day_of_week = datetime.strptime(date_today, "%Y-%m-%d").weekday()
        return (not us_holiday) and (not local_holiday) and (day_of_week < 5)


class CustomWorkday(Workday):

    """Write custom code for your institution."""

    def is_workday(self) -> bool:
        pass


class WorkdayFactory:

    def create_workday(self, method):
        if method == "usa":
            return UsaWorkday()
        elif method == "custom":
            return CustomWorkday()
        else:
            msg = 'Unknown workday. Valid choices are "usa" and "custom".'
            raise ValueError(msg)
