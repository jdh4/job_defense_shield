import os
import subprocess
import textwrap
import pickle
from time import sleep
from base import Alert
from utils import SECONDS_PER_MINUTE
from utils import SECONDS_PER_HOUR
from utils import MINUTES_PER_HOUR
from utils import get_first_name
from utils import send_email
from efficiency import num_gpus_with_zero_util


class ZeroGpuUtilization(Alert):

    """Send warnings and cancel jobs with zero GPU utilization. Interactive jobs
       with only 1 GPU and a run time limit of less than 8 hours are ignored."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    @staticmethod
    def get_stats_for_running_job(jobid, cluster):
        """Get the job statistics for running jobs by calling jobstats"""
        import importlib.machinery
        import importlib.util
        cluster = cluster.replace("tiger", "tiger2")
        loader = importlib.machinery.SourceFileLoader('jobstats', '/usr/local/bin/jobstats')
        spec = importlib.util.spec_from_loader('jobstats', loader)
        mymodule = importlib.util.module_from_spec(spec)
        loader.exec_module(mymodule)
        stats = mymodule.JobStats(jobid=jobid, cluster=cluster, prom_server="http://vigilant2:8480")
        sleep(0.5)
        return eval(stats.report_job_json(False))

    def _filter_and_add_new_fields(self):
        lower = self.first_warning_minutes * SECONDS_PER_MINUTE
        upper = (self.cancel_minutes + self.sampling_period_minutes) * SECONDS_PER_MINUTE
        self.jb = self.df[(self.df.state == "RUNNING") &
                          (self.df.gpus > 0) &
                          self.df.cluster.isin(self.clusters) &
                          self.df.partition.isin(self.partition) &
                          (self.df.elapsedraw >= lower) &
                          (self.df.elapsedraw <  upper) &
                          (self.df["limit-minutes"] > self.cancel_minutes) &
                          (~self.df.netid.isin(self.excluded_users))].copy()
        # read cache of jobs that are known to be using the gpus
        pre_approved = []
        if os.path.isfile(self.jobids_file):
            with open(self.jobids_file, "rb") as fp:
                jobs_using_gpus = pickle.load(fp)
            pre_approved = self.jb[self.jb.jobid.isin(jobs_using_gpus)].jobid.tolist()
            self.jb = self.jb[~self.jb.jobid.isin(jobs_using_gpus)]
        self.jb.rename(columns={"netid":"NetID"}, inplace=True)
        if not self.jb.empty:
            self.jb["jobstats"] = self.jb.apply(lambda row:
                                                ZeroGpuUtilization.get_stats_for_running_job(row["jobid"],
                                                                                             row["cluster"]),
                                                                                             axis='columns')
            self.jb["GPUs-Unused"] = self.jb.jobstats.apply(num_gpus_with_zero_util)
            # save cache of jobs that are known to be using the gpus
            jobs_using_gpus = self.jb[self.jb["GPUs-Unused"] == 0].jobid.tolist()
            if jobs_using_gpus:
                with open(self.jobids_file, "wb") as fp:
                    pickle.dump(jobs_using_gpus + pre_approved, fp)
            self.jb = self.jb[self.jb["GPUs-Unused"] > 0]
            self.jb["interactive"] = self.jb["jobname"].apply(lambda x: True
                                                                        if x.startswith("sys/dashboard") or
                                                                           x.startswith("interactive")
                                                                        else False)
            self.jb["salloc"] = self.jb["jobname"].apply(lambda x: True if x.startswith("interactive") else False)
            msk = self.jb["interactive"] & (self.jb.gpus == 1) & (self.jb["limit-minutes"] <= self.max_interactive_hours * MINUTES_PER_HOUR)
            self.jb = self.jb[~msk]
            self.jb = self.jb[["jobid", "NetID", "cluster", "partition", "gpus", "GPUs-Unused", "elapsedraw", "salloc"]]
            renamings = {"gpus":"GPUs-Allocated", "jobid":"JobID", "cluster":"Cluster", "partition":"Partition"}
            self.jb.rename(columns=renamings, inplace=True)

    def send_emails_to_users(self):
        for user in self.jb.NetID.unique():
            emails_sent = self.get_emails_sent_count(user, "zero_gpu_utilization", days=10000)
            #################
            # first warning #
            #################
            upper = (self.first_warning_minutes + self.sampling_period_minutes) * SECONDS_PER_MINUTE
            usr = self.jb[(self.jb.elapsedraw < upper) &
                          (self.jb.NetID == user)].copy()
            if not usr.empty:
                s = f"{get_first_name(user)},\n\n"
                text = (
                'You have GPU job(s) that have been running for more than 1 hour but appear to not be using the GPU(s):'
                )
                s += "\n".join(textwrap.wrap(text, width=80))
                s += "\n\n"

                usr["GPU-Util"] = "0%"
                usr["Hours"] = usr.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR, 1))
                usr.drop(columns=["NetID", "elapsedraw", "salloc"], inplace=True)

                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([5 * " " + row for row in usr_str.split("\n")])
                s += "\n"

                if (emails_sent >= self.min_previous_warnings):
                    s += textwrap.dedent("""
                Your jobs will be AUTOMATICALLY CANCELLED if they are found to not be using the
                GPUs for 2 hours. For more information see the <a href="https://researchcomputing.princeton.edu/get-started/utilization-policies">Utilization Policies</a>.
                """)

                s += "\n"
                text = (
                f'Please consider cancelling the job(s) listed above by using the "scancel" command:'
                )
                s += "\n".join(textwrap.wrap(text, width=80))
                s += "\n\n"
                s += f"     $ scancel {usr.JobID.values[0]}"
                s += "\n\n"

                zero = 'Run the "jobstats" command to see the GPU utilization:'
                s += "\n".join(textwrap.wrap(zero, width=80))
                s += "\n\n"
                s += f"     $ jobstats {usr.JobID.values[0]}"
                s += "\n"

                s += textwrap.dedent("""
                See our <a href="https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#zero-util">GPU Computing</a> webpage for three common reasons for encountering zero GPU
                utilization.

                Consider attending an in-person Research Computing <a href="https://researchcomputing.princeton.edu/support/help-sessions">help session</a> for assistance.
                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)

                send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email(s, "alerts-jobs-aaaalegbihhpknikkw2fkdx6gi@princetonrc.slack.com", subject=f"{self.subject}", sender="cses@princeton.edu")
                print(s)

                # append the new violations to the log file
                vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
                usr.drop(columns=["GPU-Util"], inplace=True)
                Alert.update_violation_log(usr, vfile)

            ##################
            # second warning #
            ##################
            lower = self.second_warning_minutes * SECONDS_PER_MINUTE
            upper = (self.second_warning_minutes + self.sampling_period_minutes) * SECONDS_PER_MINUTE
            usr = self.jb[(self.jb.elapsedraw >= lower) &
                          (self.jb.elapsedraw <  upper) &
                          (self.jb.NetID == user)].copy()
            print(usr)
            if not usr.empty and (emails_sent >= self.min_previous_warnings):
                s = f"{get_first_name(user)},\n\n"
                text = (
                'This is a second warning. The jobs below will be cancelled in about 15 minutes unless GPU activity is detected:'
                )
                s += "\n".join(textwrap.wrap(text, width=80))
                s += "\n\n"

                usr["GPU-Util"] = "0%"
                usr["Hours"] = usr.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR, 1))
                usr.drop(columns=["NetID", "elapsedraw", "salloc"], inplace=True)

                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([5 * " " + row for row in usr_str.split("\n")])
                s += "\n\n"
                s += 'See our <a href="https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#zero-util">GPU Computing</a> webpage for three common reasons for encountering zero GPU\n'
                s += "utilization."

                send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}")
                send_email(s, "alerts-jobs-aaaalegbihhpknikkw2fkdx6gi@princetonrc.slack.com", subject=f"{self.subject}")
                print(s)

            ###############
            # cancel jobs #
            ###############
            lower = self.cancel_minutes * SECONDS_PER_MINUTE
            usr = self.jb[(self.jb.elapsedraw >= lower) & (self.jb.NetID == user)].copy()
            print(usr)
            if not usr.empty and (emails_sent >= self.min_previous_warnings):
                s = f"{get_first_name(user)},\n\n"
                text = (
                'The jobs below have been cancelled because they ran for at least 2 hours at 0% GPU utilization:'
                )
                s += "\n".join(textwrap.wrap(text, width=80))
                s += "\n\n"
 
                usr["GPU-Util"] = "0%"
                usr["State"] = "CANCELLED"
                usr["Hours"] = usr.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR, 1))
                usr = usr[["JobID", "Cluster", "Partition", "State", "GPUs-Allocated", "GPU-Util", "Hours"]]

                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([5 * " " + row for row in usr_str.split("\n")])
                s += "\n"
                s += textwrap.dedent("""
                For more information about job cancellations see the <a href="https://researchcomputing.princeton.edu/get-started/utilization-policies">Utilization Policies</a>.

                See our <a href="https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#zero-util">GPU Computing</a> webpage for three common reasons for encountering zero GPU
                utilization.

                Consider attending an in-person Research Computing <a href="https://researchcomputing.princeton.edu/support/help-sessions">help session</a> for assistance.
                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)
                
                send_email(s, f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                for email in admin_self.emails: 
                    send_email(s, f"{email}", subject=f"{self.subject}", sender="cses@princeton.edu")
                print(s)

                for jobid in usr.JobID.tolist():
                    cmd = f"scancel {jobid}"
                    _ = subprocess.run(cmd,
                                       stdout=subprocess.PIPE,
                                       shell=True,
                                       timeout=10,
                                       text=True,
                                       check=True)
                    with open("/var/spool/slurm/job_defense_shield/cancelled.txt", "a") as fp:
                        fp.write(f"{jobid},{user}\n")
