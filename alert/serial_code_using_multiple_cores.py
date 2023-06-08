import textwrap
from base import Alert
from utils import add_dividers
from utils import get_first_name
from utils import send_email
from efficiency import cpu_efficiency
import numpy as np


class SerialCodeUsingMultipleCores(Alert):

    """Find serial codes that are using multiple CPU-cores."""

    cpu_hours_threshold = 50

    def __init__(self, df, days_between_emails, violation, vpath, subject):
        super().__init__(df, days_between_emails, violation, vpath, subject)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == "della") &
                          (self.df.partition == "cpu") &
                          (self.df.nodes == 1) &
                          (self.df.cores > 1) &
                          (self.df.admincomment != {}) &
                          (self.df["elapsed-hours"] >= 1)].copy()
        # add new fields
        self.df["cpu-eff"] = self.df.apply(lambda row:
                                           cpu_efficiency(row["admincomment"],
                                                          row["elapsedraw"],
                                                          row["jobid"],
                                                          row["cluster"],
                                                          single=True,
                                                          precision=1),
                                                          axis="columns")
        # ignore jobs at 0% CPU-eff (also avoids division by zero later)
        self.df = self.df[self.df["cpu-eff"] >= 1]
        # max efficiency if serial is 100% / cores
        self.df["inverse-cores"] = 100 / self.df["cores"]
        self.df["inverse-cores"] = self.df["inverse-cores"].apply(lambda x: round(x, 1))
        self.df["ratio"] = self.df["cpu-eff"] / self.df["inverse-cores"]
        self.df = self.df[(self.df["ratio"] <= 1) & (self.df["ratio"] > 0.85)]
        renamings = {"elapsed-hours":"Hours",
                     "jobid":"JobID",
                     "netid":"NetID",
                     "partition":"Partition",
                     "cores":"CPU-cores",
                     "cpu-eff":"CPU-Util",
                     "inverse-cores":"100%/CPU-cores"}
        self.df = self.df.rename(columns=renamings)
        self.df = self.df[["JobID",
                           "NetID",
                           "Partition",
                           "CPU-cores",
                           "100%/CPU-cores",
                           "CPU-Util",
                           "Hours",
                           "cpu-hours"]]
        self.df = self.df.sort_values(by=["NetID", "JobID"])
        self.df["100%/CPU-cores"] = self.df["100%/CPU-cores"].apply(lambda x: f"{x}%")
        self.df["CPU-Util"] = self.df["CPU-Util"].apply(lambda x: f"{x}%")

    def send_emails_to_users(self):
        for user in self.df.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                if usr["cpu-hours"].sum() >= SerialCodeUsingMultipleCores.cpu_hours_threshold:
                    usr = usr.drop(columns=["cpu-hours"])
                    s =  f"{get_first_name(user)},\n\n"
                    s += f"Below are jobs that ran on Della in the past {self.days_between_emails} days:"
                    s +=  "\n\n"
                    usr_str = usr.to_string(index=False, justify="center").split("\n")
                    s +=  "\n".join([4 * " " + row for row in usr_str])
                    s +=  "\n"
                    s += textwrap.dedent(f"""
                    The CPU utilization (CPU-Util) of each job above is approximately equal to
                    100% divided by the number of allocated CPU-cores (100%/CPU-cores). This
                    suggests that you may be running a code that can only use 1 CPU-core. If this is
                    true then allocating more than 1 CPU-core is wasteful. A good target value for
                    CPU-Util is 90% and above.

                    Please consult the documentation of the software to see if it is parallelized.
                    For a general overview of parallel computing:
        
                        https://researchcomputing.princeton.edu/support/knowledge-base/parallel-code

                    If the code cannot run in parallel then please use the following Slurm
                    directives:

                        #SBATCH --nodes=1
                        #SBATCH --ntasks=1
                        #SBATCH --cpus-per-task=1

                    You will experience shorter queue times by allocating only 1 CPU-core per job.

                    If you believe that the code is capable of using more than 1 CPU-core then
                    consider attending a Research Computing help session for assistance with
                    running parallel jobs:

                        https://researchcomputing.princeton.edu/support/help-sessions

                    You can check the CPU utilization of completed and actively running jobs by using
                    the "jobstats" command. For example:

                        $ jobstats {usr['JobID'].values[0]}

                    Add the following lines to your Slurm scripts to receive an email report with
                    CPU utilization information after each job finishes:

                        #SBATCH --mail-type=end
                        #SBATCH --mail-user={user}@princeton.edu

                    Replying to this automated email will open a support ticket with Research
                    Computing. Let us know if we can be of help.
                    """)
                    send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                    send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                    print(s)

                    # append the new violations to the log file
                    Alert.update_violation_log(usr, vfile)
   
    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            d = {"cpu-hours":np.sum, "NetID":np.size, "CPU-cores":np.mean}
            self.gp = self.df.groupby("NetID").agg(d)
            self.gp = self.gp.rename(columns={"NetID":"Jobs", "cpu-hours":"CPU-hours"})
            self.gp = self.gp[self.gp["CPU-hours"] >= SerialCodeUsingMultipleCores.cpu_hours_threshold]
            self.gp["CPU-hours"] = self.gp["CPU-hours"].apply(round)
            self.gp["CPU-cores"] = self.gp["CPU-cores"].apply(lambda x: round(x, 1))
            self.gp.reset_index(drop=False, inplace=True)
            self.gp = self.gp[["NetID", "CPU-hours", "CPU-cores", "Jobs"]]
            self.gp = self.gp.sort_values(by="CPU-hours", ascending=False)
            self.gp.reset_index(drop=True, inplace=True)
            self.gp.index += 1
            return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
