"""Abstract and concrete classes for cleaning a pandas dataframe
   containing job data."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseCleaner(ABC):

    def __init__(self,
                 raw: pd.DataFrame,
                 field_renamings: dict[str, str],
                 partition_renamings: dict[str, str]) -> None:
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
    """Need to be calculating jobs with non-zero elapsed-time."""

    def __init__(self, raw, field_renamings, partition_renamings):
        super().__init__(raw, field_renamings, partition_renamings)

    def translate_fields(self):
        pass

    def rename_columns(self) -> pd.DataFrame:
        """Rename certain columns of the dataframe."""
        return self.raw.rename(columns=self.field_renamings)
 
    def rename_partitions(self) -> pd.Series:
        """Rename certain partitions."""
        return self.raw.partition.replace(self.partition_renamings)

    def clean_partitions(self, series: pd.Series) -> pd.Series:
        """A job can have a partition value that is a comma-separated list
           of partitions if a system administrator manually changes the
           partition of the job while queued. This method extracts the
           partition that the job actually ran on."""
        num_commas = len(series[series.str.contains(",")])
        if num_commas:
            print(f"Number of rows with comma in partition: {num_commas}")
            return series.apply(lambda p: p.split(",")[0])
        return series

    def unify_cancel_state(self) -> pd.Series:
        """There are many different states for cancelled jobs. This method
           causes all cancelled jobs to have the state CANCELLED."""
        return self.raw.state.apply(lambda s:
                                    "CANCELLED" if "CANCEL" in s else s)

    def unlimited_time_limits(self) -> pd.DataFrame:
        """Slurm allows for jobs with unlimited time limits denoted
           by UNLIMITED. One must deal with these non-numeric values."""
        num_unlimited = len(self.raw[self.raw["limit-minutes"] == "UNLIMITED"])
        if num_unlimited:
            print(f"Number UNLIMITED: {num_unlimited}")
            self.raw["limit-minutes"] = self.raw.apply(lambda row: row["end"]
                                        if row["limit-minutes"] == "UNLIMITED" and
                                           row["state"] != "PENDING"
                                        else row["limit-minutes"],
                                        axis="columns")
        return self.raw

    def clean_nodes_and_cores(self):
        """Convert the nodes and cores datatype to int64."""
        self.raw = self.raw[pd.notna(self.raw.state) &
                            pd.notna(self.raw.nodes) &
                            pd.notna(self.raw.cores)]
        num_nodes = len(self.raw[~self.raw.nodes.str.isnumeric()])
        num_cores = len(self.raw[~self.raw.cores.str.isnumeric()])
        if num_nodes or num_cores:
            print(f"Number of rows with non-numeric nodes: {num_nodes}")
            print(f"Number of rows with non-numeric cores: {num_cores}")
            self.raw = self.raw[self.raw.cores.str.isnumeric() &
                                self.raw.nodes.str.isnumeric()]
        self.raw.nodes = self.raw.nodes.astype("int64")
        self.raw.cores = self.raw.cores.astype("int64")
        return self.raw

    def remove_nulls(self):
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

    def clean_time_columns(self):
        for col in ["elapsedraw", "submit", "eligible", "start", "end"]:
            self.raw = self.raw[pd.notna(self.raw[col])]
            self.raw = self.raw[self.raw[col].str.isnumeric()]
            self.raw[col] = self.raw[col].astype("int64")
        self.raw = self.raw[self.raw.elapsedraw > 0]
        self.raw["cpu-seconds"] = self.raw["cpu-seconds"].astype("int64")
        return self.raw

    def limit_minutes_final(self):
        """This method makes sures that all values of limit-minutes are
           numerical. Jobs in a state of PENDING are handled above when
           limit-minutes is UNLIMITED. This method exists since there is
           concern about states like RESIZING, REQUEUED and SUSPENDED."""
        self.raw = self.raw[pd.notna(self.raw["limit-minutes"])]
        self.raw = self.raw[self.raw["limit-minutes"].str.isnumeric()]
        self.raw["limit-minutes"] = self.raw["limit-minutes"].astype("int64")
        return self.raw

    def clean(self):
        print(self.raw.info())
        self.raw = self.rename_columns()
        self.raw = self.raw[pd.notna(self.raw.state) &
                            pd.notna(self.raw.partition)]
        self.raw.partition = self.rename_partitions()
        self.raw.partition = self.clean_partitions(self.raw.partition)
        self.raw.state = self.unify_cancel_state()
        self.raw = self.unlimited_time_limits()
        self.raw = self.clean_nodes_and_cores()
        self.raw = self.remove_nulls()
        self.raw = self.clean_time_columns()
        self.raw = self.limit_minutes_final()
        print(self.raw.info())
        return self.raw
