import os
import subprocess
import pickle
import pandas as pd
from base import Alert
from utils import SECONDS_PER_MINUTE as spm
from utils import SECONDS_PER_HOUR as sph
from utils import MINUTES_PER_HOUR as mph
from efficiency import num_gpus_with_zero_util
from greeting import GreetingFactory
from email_translator import EmailTranslator


class CancelZeroGpuJobs(Alert):

    """Send warnings and automatically cancel jobs with zero GPU utilization.
       Jobs with a limit-minutes that is less than self.cancel_minutes cannot
       be excluded since real limit may be UNLIMITED."""

    def __init__(self, df, days_between_emails, violation, vpath, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, **kwargs)

    def _add_required_fields(self):
        if not hasattr(self, "email_subject"):
            self.email_subject = "Jobs with Zero GPU Utilization"
        if not hasattr(self, "do_not_cancel"):
            self.do_not_cancel = False
        self.jobids_to_cancel = []

    def _filter_and_add_new_fields(self):
        if not hasattr(self, "first_warning_minutes") and \
           not hasattr(self, "second_warning_minutes"):
            lower = self.cancel_minutes * spm
        else:
            lower = self.first_warning_minutes * spm
        upper = (self.cancel_minutes + self.sampling_period_minutes) * spm
        self.df = self.df[(self.df.state == "RUNNING") &
                          (self.df.gpus > 0) &
                          (self.df.cluster == self.cluster) &
                          (self.df.partition.isin(self.partitions)) &
                          (self.df.elapsedraw >= lower) &
                          (self.df.elapsedraw <  upper) &
                          (~self.df.user.isin(self.excluded_users))].copy()
        self.df.rename(columns={"user":"User"}, inplace=True)

        # read cache file containing jobid's that are known to be using the gpus
        pre_approved = []
        if hasattr(self, "jobid_cache_path") and os.path.isdir(self.jobid_cache_path):
            jobid_cache_file = os.path.join(self.jobid_cache_path, ".jobid_cache.pkl")
            if os.path.isfile(jobid_cache_file):
                with open(jobid_cache_file, "rb") as fp:
                    jobs_using_gpus = pickle.load(fp)
                pre_approved = self.df[self.df.jobid.isin(jobs_using_gpus)].jobid.tolist()
                print("pre (overlap)", len(jobs_using_gpus), len(pre_approved), len(set(jobs_using_gpus+pre_approved)))
                self.df = self.df[~self.df.jobid.isin(jobs_using_gpus)]
        if not self.df.empty:
            self.df.admincomment = Alert.get_admincomment_for_running_jobs(self)
            self.df["zero-tuple"] = self.df.apply(lambda row:
                                         num_gpus_with_zero_util(row["admincomment"],
                                                                 row["jobid"],
                                                                 row["cluster"]),
                                                                 axis="columns")
            cols = ["GPUs-Unused", "error_code"]
            self.df[cols] = pd.DataFrame(self.df["zero-tuple"].tolist(), index=self.df.index)
            self.df = self.df[self.df["error_code"] == 0]
            # write cache file of jobid's that are known to be using the gpus
            if hasattr(self, "jobid_cache_path"):
                jobs_using_gpus = self.df[self.df["GPUs-Unused"] == 0].jobid.tolist()
                if jobs_using_gpus:
                    jobid_cache_file = os.path.join(self.jobid_cache_path, ".jobid_cache.pkl")
                    print("post", len(jobs_using_gpus), len(pre_approved), len(set(jobs_using_gpus+pre_approved)))
                    with open(jobid_cache_file, "wb") as fp:
                        pickle.dump(jobs_using_gpus + pre_approved, fp)
            self.df = self.df[self.df["GPUs-Unused"] > 0]
            # filter interactive jobs if such settings are found in config.yaml
            if hasattr(self, "max_interactive_hours") and \
               hasattr(self, "max_interactive_gpus"):
                self.df["interactive"] = self.df["jobname"].apply(lambda x: True
                                                                  if x.startswith("sys/dashboard") or
                                                                     x.startswith("interactive")
                                                                  else False)
                msk = (self.df["interactive"]) & \
                      (self.df.gpus <= self.max_interactive_gpus) & \
                      (self.df["limit-minutes"] <= self.max_interactive_hours * mph)
                self.df = self.df[~msk]
            self.df = self.df[["jobid",
                               "User",
                               "cluster",
                               "partition",
                               "gpus",
                               "GPUs-Unused",
                               "elapsedraw"]]
            renamings = {"gpus":"GPUs-Allocated",
                         "jobid":"JobID",
                         "cluster":"Cluster",
                         "partition":"Partition"}
            self.df.rename(columns=renamings, inplace=True)
            self.df["GPU-Util"] = "0%"
            self.df["Hours"] = self.df.elapsedraw.apply(lambda x: round(x / sph, 1))
            if not self.df.empty:
                print(self.df[["JobID", "GPUs-Unused", "Partition", "elapsedraw"]])

    def create_emails(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.df.User.unique():
            indent = 4 * " "
            #################
            # first warning #
            #################
            if hasattr(self, "first_warning_minutes"):
                upper = (self.first_warning_minutes + self.sampling_period_minutes) * spm
                usr = self.df[(self.df.elapsedraw < upper) &
                              (self.df.User == user)].copy()
                if not usr.empty:
                    usr.drop(columns=["User", "elapsedraw"], inplace=True)
                    table = usr.to_string(index=False, justify="center").split("\n")
                    tags = {}
                    tags["<GREETING>"] = g.greeting(user)
                    tags["<CLUSTER>"] = self.cluster
                    tags["<PARTITIONS>"] = ",".join(sorted(set(usr.Partition)))
                    tags["<MINUTES-1ST>"] = str(self.first_warning_minutes)
                    tags["<SAMPLING>"] = str(self.sampling_period_minutes)
                    tags["<TABLE>"] = "\n".join([indent + row for row in table])
                    tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                    tags["<SCANCEL>"] = f"{indent}$ scancel {usr.JobID.values[0]}"
                    translator = EmailTranslator(self.email_files_path,
                                                 self.email_file_first_warning,
                                                 tags)
                    email = translator.replace_tags()
                    self.emails.append((user, email, None))

            ##################
            # second warning #
            ##################
            if hasattr(self, "first_warning_minutes") and \
               hasattr(self, "second_warning_minutes"):
                lower = self.second_warning_minutes * spm
                upper = (self.second_warning_minutes + self.sampling_period_minutes) * spm
                usr = self.df[(self.df.elapsedraw >= lower) &
                              (self.df.elapsedraw <  upper) &
                              (self.df.User == user)].copy()
                if not usr.empty:
                    usr.drop(columns=["User", "elapsedraw"], inplace=True)
                    table = usr.to_string(index=False, justify="center").split("\n")
                    tags = {}
                    tags["<GREETING>"] = g.greeting(user)
                    tags["<CLUSTER>"] = self.cluster
                    tags["<PARTITIONS>"] = ",".join(sorted(set(usr.Partition)))
                    tags["<MINUTES-2ND>"] = str(self.seond_warning_minutes)
                    tags["<SAMPLING>"] = str(self.sampling_period_minutes)
                    tags["<TABLE>"] = "\n".join([indent + row for row in table])
                    tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                    tags["<SCANCEL>"] = f"{indent}$ scancel {usr.JobID.values[0]}"
                    translator = EmailTranslator(self.email_files_path,
                                                 self.email_file_second_warning,
                                                 tags)
                    email = translator.replace_tags()
                    self.emails.append((user, email, None))

            ################
            # cancellation #
            ################
            lower = self.cancel_minutes * spm
            usr = self.df[(self.df.elapsedraw >= lower) &
                          (self.df.User == user)].copy()
            if not usr.empty:
                usr["State"] = "CANCELLED"
                usr = usr[["JobID",
                           "Cluster",
                           "Partition",
                           "State",
                           "GPUs-Allocated",
                           "GPU-Util",
                           "Hours"]]
                table = usr.to_string(index=False, justify="center").split("\n")
                tags = {}
                tags["<GREETING>"] = g.greeting(user)
                tags["<CLUSTER>"] = self.cluster
                tags["<PARTITIONS>"] = ",".join(sorted(set(usr.Partition)))
                tags["<TABLE>"] = "\n".join([indent + row for row in table])
                tags["<JOBSTATS>"] = f"{indent}$ jobstats {usr.JobID.values[0]}"
                tags["<CANCEL-MIN>"] = str(self.cancel_minutes)
                tags["<CANCEL-HRS>"] = f"{round(self.cancel_minutes / mph)}"
                translator = EmailTranslator(self.email_files_path,
                                             self.email_file_cancel,
                                             tags)
                email = translator.replace_tags()
                self.emails.append((user, email, usr))
                self.jobids_to_cancel.extend(usr.JobID.tolist())

    def cancel_jobs(self) -> None:
        """Call scancel on each jobid. For this to work, it must be ran as
           a user with sufficient privileges."""
        if not self.do_not_cancel:
            for jobid in self.jobids_to_cancel:
                cmd = f"scancel {jobid}"
                _ = subprocess.run(cmd,
                                   stdout=subprocess.PIPE,
                                   shell=True,
                                   timeout=10,
                                   text=True,
                                   check=True)
                print(f"Cancelled job {jobid} due to zero GPU utilization.")
