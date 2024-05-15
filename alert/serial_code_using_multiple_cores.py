import textwrap
from base import Alert
from utils import add_dividers
from utils import get_first_name
from utils import send_email
from efficiency import cpu_efficiency


class SerialCodeUsingMultipleCores(Alert):

    """Find serial codes that are using multiple CPU-cores."""

    cpu_hours_threshold = 100

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df.cluster == "della") &
                          (self.df.partition == "cpu") &
                          (~self.df.netid.isin(["vonholdt"])) &
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
                           "CPU-Util",
                           "100%/CPU-cores",
                           "Hours"]]
        self.df = self.df.sort_values(by=["NetID", "JobID"])
        self.df["100%/CPU-cores"] = self.df["100%/CPU-cores"].apply(lambda x: f"{x}%")
        self.df["CPU-Util"] = self.df["CPU-Util"].apply(lambda x: f"{x}%")
        self.df["cores-minus-1"] = self.df["CPU-cores"] - 1
        self.df["CPU-Hours-Wasted"] = self.df["Hours"] * self.df["cores-minus-1"]

    def send_emails_to_users(self):
        for user in self.df.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                cpu_hours_wasted = usr["CPU-Hours-Wasted"].sum()
                usr = usr.drop(columns=["NetID", "cores-minus-1", "CPU-Hours-Wasted"])
                prev_emails = self.get_emails_sent_count(user, self.violation, days=90)
                num_disp = 15
                total_jobs = usr.shape[0]
                case = f"{num_disp} of your {total_jobs} jobs" if total_jobs > num_disp else "your jobs"
                cores_per_node = 32
                hours_per_week = 24 * 7
                num_wasted_nodes = round(cpu_hours_wasted / cores_per_node / hours_per_week)
                if cpu_hours_wasted >= SerialCodeUsingMultipleCores.cpu_hours_threshold:
                    s =  f"{get_first_name(user)},\n\n"
                    s += f"Below are {case} that ran on Della (cpu) in the past {self.days_between_emails} days:"
                    s +=  "\n\n"
                    usr_str = usr.head(num_disp).to_string(index=False, justify="center").split("\n")
                    s +=  "\n".join([4 * " " + row for row in usr_str])
                    s +=  "\n"
                    s += textwrap.dedent(f"""
                    The CPU utilization (CPU-Util) of each job above is approximately equal to
                    100% divided by the number of allocated CPU-cores (100%/CPU-cores). This
                    suggests that you may be running a code that can only use 1 CPU-core. If this is
                    true then allocating more than 1 CPU-core is wasteful. A good target value for
                    CPU utilization is 90% and above.
                    """)

                    if num_wasted_nodes > 1:
                        s += f"\nYour jobs allocated {cpu_hours_wasted} CPU-hours that were never used. This is equivalent to\n"
                        s += f"making {num_wasted_nodes} nodes unavailable to all users (including yourself) for 1 week!\n"

                    s += textwrap.dedent(f"""
                    Please consult the documentation of the software to see if it is parallelized.
                    For a general overview of parallel computing:
        
                        https://researchcomputing.princeton.edu/support/knowledge-base/parallel-code

                    If the code cannot run in parallel then please use the following Slurm
                    directives:

                        #SBATCH --nodes=1
                        #SBATCH --ntasks=1
                        #SBATCH --cpus-per-task=1

                    You will experience shorter queue times by allocating only 1 CPU-core per job.
                    In some cases this will also allow you run more jobs simultaneously.

                    If you believe that the code is capable of using more than 1 CPU-core then
                    consider attending an in-person Research Computing help session for assistance:

                        https://researchcomputing.princeton.edu/support/help-sessions

                    You can check the CPU utilization of completed and actively running jobs by using
                    the "jobstats" command. For example:

                        $ jobstats {usr['JobID'].values[0]}

                    Replying to this automated email will open a support ticket with Research
                    Computing. Let us know if we can be of help.
                    """)
                    send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                    send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                    send_email(s, "alerts-jobs-aaaalegbihhpknikkw2fkdx6gi@princetonrc.slack.com", subject=f"{self.subject}", sender="cses@princeton.edu")
                    print(s)

                    # append the new violations to the log file
                    Alert.update_violation_log(usr, vfile)
   
    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            d = {"CPU-Hours-Wasted":"sum", "NetID":"size", "CPU-cores":"mean"}
            self.gp = self.df.groupby("NetID").agg(d)
            self.gp = self.gp.rename(columns={"NetID":"Jobs"})
            self.gp = self.gp[self.gp["CPU-Hours-Wasted"] >= SerialCodeUsingMultipleCores.cpu_hours_threshold]
            if self.gp.empty:
                return ""
            self.gp["CPU-Hours-Wasted"] = self.gp["CPU-Hours-Wasted"].apply(round)
            self.gp["CPU-cores"] = self.gp["CPU-cores"].apply(lambda x: round(x, 1))
            self.gp = self.gp.rename(columns={"CPU-cores":"AvgCores"})
            self.gp.reset_index(drop=False, inplace=True)
            self.gp["email90"] = self.gp.NetID.apply(lambda netid:
                                               self.get_emails_sent_count(netid,
                                                                          self.violation,
                                                                          days=90))
            self.gp = self.gp[["NetID", "CPU-Hours-Wasted", "AvgCores", "Jobs", "email90"]]
            self.gp = self.gp.sort_values(by="CPU-Hours-Wasted", ascending=False)
            self.gp.reset_index(drop=True, inplace=True)
            self.gp.index += 1
            return add_dividers(self.gp.to_string(index=keep_index, justify="center"), title)
