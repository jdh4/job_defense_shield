"""Abstract and concrete classes for cleaning a pandas dataframe
   containing job data."""

from abc import ABC, abstractmethod
from typing import Dict
from datetime import datetime
import pandas as pd


class BaseCleaner(ABC):

    def __init__(self,
                 raw: pd.DataFrame,
                 field_renamings: Dict[str, str],
                 partition_renamings: Dict[str, str]) -> None:
        self.raw = raw
        self.fields = list(raw.columns)
        self.field_renamings = field_renamings
        self.partition_renamings = partition_renamings

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def translate_fields(self):
        """The main code assumes the field names for Slurm. This function
           converts the field names for one workload manager to those
           for Slurm."""
        pass


class SacctCleaner(BaseCleaner):

    """Class for cleaning a pandas dataframe containing Slurm sacct data."""

    def __init__(self, raw, field_renamings, partition_renamings):
        super().__init__(raw, field_renamings, partition_renamings)

    def translate_fields(self):
        pass

    def rename_columns(self) -> pd.DataFrame:
        """Rename certain columns of the dataframe."""
        return self.raw.rename(columns=self.field_renamings)
 
    def clean_partitions(self) -> pd.Series:
        """A job can have a partition value that is a comma-separated list
           of partitions if a system administrator manually changes the
           partition of the job while queued. This method extracts the
           partition that the job actually ran on."""
        num_commas = len(self.raw[self.raw.partition.str.contains(",")])
        if num_commas:
            print(f"Number of rows with comma in partition: {num_commas}")
            series = self.raw.partition[self.raw.partition.str.contains(",")]
            print(series.value_counts())
            return self.raw.partition.apply(lambda p: p.split(",")[0])
        return self.raw.partition

    def rename_partitions(self) -> pd.Series:
        """Rename certain partitions. This will work for comma-separated
           partitions."""
        return self.raw.partition.replace(self.partition_renamings)

    def unify_cancel_state(self) -> pd.Series:
        """There are many different states for cancelled jobs. This method
           causes all cancelled jobs to have the state CANCELLED."""
        return self.raw.state.apply(lambda s:
                                    "CANCELLED" if "CANCEL" in s else s)

    def unlimited_time_limits(self) -> pd.DataFrame:
        """Slurm allows for jobs with unlimited time limits denoted
           by UNLIMITED. This function handles these non-numeric values with the
           exception of pending jobs. Rows with null values for limit-minutes
           have already been dropped. Pending jobs are assigned a limit-minutes
           of -1 so that all values are numerical."""
        num_unlimited = len(self.raw[self.raw["limit-minutes"] == "UNLIMITED"])
        if num_unlimited:
            print(f"Number UNLIMITED: {num_unlimited}")
            now_secs = int(datetime.now().timestamp())
            def fix_limit_minutes(limit_minutes, state, start, end):
                if limit_minutes == "UNLIMITED" and state == "PENDING":
                    return -1
                elif limit_minutes == "UNLIMITED" and \
                    state == "RUNNING" and \
                    str(start).isnumeric():
                    return now_secs - int(start)
                elif limit_minutes == "UNLIMITED" and \
                    str(start).isnumeric() and \
                    str(end).isnumeric():
                    return int(end) - int(start)
                else:
                    return limit_minutes
            self.raw["limit-minutes"] = self.raw.apply(lambda row:
                                        fix_limit_minutes(row["limit-minutes"],
                                                          row["state"],
                                                          row["start"],
                                                          row["end"]), axis="columns")
        return self.raw

    def clean_nodes_and_cores(self) -> pd.DataFrame:
        """Convert the nodes and cores datatype to int64."""
        for col in ["nodes", "cores"]:
            if self.raw[col].dtype == "object":
                num_rows = len(self.raw[~self.raw[col].str.isnumeric()])
                if num_rows:
                    self.raw = self.raw[self.raw[col].str.isnumeric()]
                    print(f"{num_rows:>6} rows dropped with non-numeric {col}")
            self.raw[col] = self.raw[col].astype("int64")
        return self.raw

    def remove_nulls(self) -> pd.DataFrame:
        """Remove pending jobs."""
        self.raw.start = self.raw.apply(lambda row: row["eligible"]
                                        if row["start"] == "Unknown"
                                        else row["start"],
                                        axis="columns")
        self.raw = self.raw[pd.notnull(self.raw.alloctres) &
                            (self.raw.alloctres != "") &
                            pd.notnull(self.raw.start) &
                            (~self.raw.start.isin(["", "None"]))]
        return self.raw

    def clean_time_columns(self) -> pd.DataFrame:
        """Return the dataframe with the numerical columns cleaned."""
        for col in ["elapsedraw", "submit", "eligible"]:
            self.raw[col] = self.raw[col].astype("str")
            num_rows = len(self.raw)
            self.raw = self.raw[pd.notna(self.raw[col])]
            if self.raw[col].dtype == 'object':
                self.raw = self.raw[self.raw[col].str.isnumeric()]
            num_dropped = num_rows - len(self.raw)
            print(f"{num_dropped:>6} rows dropped while cleaning {col}")
            self.raw[col] = self.raw[col].astype("int64")
        self.raw["cpu-seconds"] = self.raw["cpu-seconds"].astype("int64")
        return self.raw

    def clean_start(self) -> pd.DataFrame:
        """Return dataframe with start field cleaned. The value of start
           is Unknown for PENDING jobs."""
        self.raw.start = self.raw.start.astype("str")
        self.raw = self.raw[self.raw.start != "None"]
        self.raw.start = self.raw.apply(lambda row:
                                        "-1" if row["state"] == "PENDING"
                                             else row["start"], axis="columns")
        num_rows = len(self.raw)
        #if self.raw.start.dtype == 'object':
        #    self.raw = self.raw[self.raw.start.str.isnumeric()]
        num_dropped = num_rows - len(self.raw)
        print(f"{num_dropped:>6} rows dropped while cleaning start")
        self.raw.start = self.raw.start.astype("int64")
        return self.raw

    def clean_end(self) -> pd.DataFrame:
        """Return dataframe with end field cleaned. The value of end
           is Unknown for RUNNING and PENDING jobs."""
        self.raw.end = self.raw.end.astype("str")
        now_secs = int(datetime.now().timestamp())
        def fix_end(state, start, end):
            if state == "PENDING":
                return "-1"
            elif str(start).isnumeric() and end == "Unknown":
                return str(now_secs)
            else:
                return end
        self.raw.end = self.raw.apply(lambda row:
                                      fix_end(row["state"],
                                              row["start"],
                                              row["end"]), axis="columns")
        num_rows = len(self.raw)
        #if self.raw.end.dtype == 'object':
        #    self.raw = self.raw[self.raw.end.str.isnumeric()]
        num_dropped = num_rows - len(self.raw)
        print(f"{num_dropped:>6} rows dropped while cleaning end")
        self.raw.end = self.raw.end.astype("int64")
        return self.raw

    def limit_minutes_final(self) -> pd.DataFrame:
        """This method makes sures that all values of limit-minutes are
           numerical. This method exists since there is concern about
           states like RESIZING, REQUEUED and SUSPENDED. Jobs in a state
           of PENDING are handled above when limit-minutes is UNLIMITED."""
        self.raw = self.raw[pd.notna(self.raw["limit-minutes"])]
        if self.raw["limit-minutes"].dtype == 'object':
            num_rows = len(self.raw)
            self.raw = self.raw[self.raw["limit-minutes"].str.isnumeric()]
            num_dropped = len(self.raw) - num_rows
            if num_dropped:
                print(f"{num_dropped} rows dropped for non-numeric limit-minutes")
        self.raw["limit-minutes"] = self.raw["limit-minutes"].astype("int64")
        return self.raw

    def clean(self) -> pd.DataFrame:
        """Return the cleaned dataframe by applying all
           of the cleaning functions."""
        print("\nCleaning sacct data:")
        #print(self.raw.state.value_counts())
        #print(self.raw[self.raw.state == "PENDING"])
        self.raw = self.raw[pd.notna(self.raw.nnodes) &
                            pd.notna(self.raw.ncpus) &
                            pd.notna(self.raw.state) &
                            pd.notna(self.raw.partition) &
                            pd.notna(self.raw.start) &
                            pd.notna(self.raw.end) &
                            pd.notna(self.raw.cputimeraw) &
                            pd.notna(self.raw.timelimitraw)]
        self.raw = self.rename_columns()
        self.raw.partition = self.clean_partitions()
        self.raw.partition = self.rename_partitions()
        self.raw.state = self.unify_cancel_state()
        self.raw = self.unlimited_time_limits()
        self.raw = self.clean_nodes_and_cores()
        #self.raw = self.remove_nulls()
        self.raw = self.clean_time_columns()
        self.raw = self.clean_start()
        self.raw = self.clean_end()
        self.raw = self.limit_minutes_final()
        #_ = self.raw.info()
        return self.raw
