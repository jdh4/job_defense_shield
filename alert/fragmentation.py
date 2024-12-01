import math
import textwrap
import pandas as pd
from base import Alert
from efficiency import cpu_memory_usage
from efficiency import cpu_nodes_with_zero_util
from utils import send_email
from utils import add_dividers
from greeting import GreetingFactory


class MultinodeCPUFragmentation(Alert):

    """Find multinode CPU jobs that use too many nodes. Ignore jobs
       with 0% CPU utilization on a node since those will captured
       by a different alert.

       The number of processors per node can vary. One needs to keep
       this in mind when computing the maximum memory used per node.
    """

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    @staticmethod
    def is_fragmented(cluster, partition, cores_per_node, mem_per_node_used):
        # value for della is hard-coded at the moment
        safety = 0.2
        cores_frac = 0.8
        if cluster == "tiger2" and \
           cores_per_node < cores_frac * 40 and \
           mem_per_node_used < (1 - safety) * 192:
            return True
        elif cluster == "della" and \
           partition == "physics" and \
           cores_per_node < cores_frac * 40 and \
           mem_per_node_used < (1 - safety) * 380:
            return True
        elif cluster == "della" and \
           partition != "physics" and \
           cores_per_node < 16 and \
           mem_per_node_used < (1 - safety) * 190:
            return True
        elif cluster == "stellar" and \
           cores_per_node < cores_frac * 96 and \
           partition in ("all", "pppl", "pu") and \
           mem_per_node_used < (1 - safety) * 768:
            return True
        else:
            return False

    @staticmethod
    def min_nodes_needed(cluster, partition, nodes, cores, mem_per_node_used):
        def cores_vs_memory(cores, cores_max, memory, memory_max):
            min_nodes_by_cores = math.ceil(cores / cores_max)
            min_nodes_by_memory = math.ceil(nodes * mem_per_node_used / ((1 - safety) * memory_max))
            return max(min_nodes_by_cores, min_nodes_by_memory)
        safety = 0.2
        if cluster == "della" and partition == "physics":
            return cores_vs_memory(cores, 40, mem_per_node_used, 380)
        elif cluster == "della" and partition != "physics":
            return cores_vs_memory(cores, 32, mem_per_node_used, 190)
        elif cluster == "tiger2":
            return cores_vs_memory(cores, 40, mem_per_node_used, 192)
        elif cluster == "stellar" and partition in ("all", "pppl", "pu"):
            return cores_vs_memory(cores, 96, mem_per_node_used, 768)
        else:
            return nodes

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df["elapsed-hours"] >= 1.1) &
                          (self.df["admincomment"] != {}) &
                          (self.df.nodes > 1) &
                          (self.df["gpu-job"] == 0) &
                          (self.df.state != "OUT_OF_MEMORY") &
                          (self.df.partition.isin(["all", "cpu", "ext", "physics", "pppl", "pu"])) &
                          (~self.df.qos.isin(["stellar-debug"])) &
                          (self.df.cluster.isin(["della", "stellar", "tiger2"]))].copy()
        # add new fields
        if not self.df.empty:
            self.df["nodes-tuple"] = self.df.apply(lambda row:
                                     cpu_nodes_with_zero_util(row["admincomment"],
                                                              row["jobid"],
                                                              row["cluster"]),
                                                              axis="columns")
            cols = ["nodes-unused", "error_code"]
            self.df[cols] = pd.DataFrame(self.df["nodes-tuple"].tolist(), index=self.df.index)
            self.df = self.df[(self.df["error_code"] == 0) & (self.df["nodes-unused"] == 0)]
            self.df["cores-per-node"] = self.df["cores"] / self.df["nodes"]
            self.df["cores-per-node"] = self.df["cores-per-node"].apply(lambda x: round(x, 1))
            self.df["memory-tuple"] = self.df.apply(lambda row:
                                                    cpu_memory_usage(row["admincomment"],
                                                                     row["jobid"],
                                                                     row["cluster"]),
                                                                     axis="columns")
            cols = ["memory-used", "memory-alloc", "error_code"]
            self.df[cols] = pd.DataFrame(self.df["memory-tuple"].tolist(), index=self.df.index)
            self.df = self.df[self.df["error_code"] == 0]
            self.df["memory-per-node-used"] = self.df["memory-used"] / self.df["nodes"]
            self.df["memory-per-node-used"] = self.df["memory-per-node-used"].apply(round)
            self.df = self.df[self.df.apply(lambda row:
                                            self.is_fragmented(row["cluster"],
                                                               row["partition"],
                                                               row["cores-per-node"],
                                                               row["memory-per-node-used"]),
                                                               axis="columns")]
            self.df["min-nodes"] = self.df.apply(lambda row:
                                                 self.min_nodes_needed(row["cluster"],
                                                                       row["partition"],
                                                                       row["nodes"],
                                                                       row["cores"],
                                                                       row["memory-per-node-used"]),
                                                                       axis="columns")
            self.df = self.df.sort_values(["cluster", "user"], ascending=[True, False])
            self.df = self.df[self.df["min-nodes"] < self.df["nodes"]]
            self.df["memory-per-node-used"] = self.df["memory-per-node-used"].apply(lambda x: f"{x} GB")
            self.df["cores-per-node"] = self.df["cores-per-node"].apply(lambda x: str(x).replace(".0", ""))
            self.df = self.df.rename(columns={"elapsed-hours":"hours",
                                              "memory-per-node-used":"mem-per-node-used"})
            cols = ["jobid",
                    "user",
                    "cluster",
                    "partition",
                    "nodes",
                    "cores",
                    "mem-per-node-used",
                    "cores-per-node",
                    "hours",
                    "min-nodes"]
            self.df = self.df[cols]

    def send_emails_to_users(self, method):
        g = GreetingFactory().create_greeting(method)
        for user in self.df.user.unique():


            if user == "martirez": continue  # EXCLUDED USER--HANDLE VIA CONFIG FILE
            if user == "mcreamer": continue  # EXCLUDED USER--HANDLE VIA CONFIG FILE


            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.user == user].copy()
                renamings = {"jobid":"JobID",
                             "cluster":"Cluster",
                             "nodes":"Nodes",
                             "mem-per-node-used":"Mem-per-Node",
                             "cores-per-node":"Cores-per-Node",
                             "hours":"Hours",
                             "min-nodes":"Nodes-Needed"}
                usr = usr.rename(columns=renamings)
                min_nodes = usr["Nodes-Needed"].mode().values[0]
                is_stellar = "stellar" in usr.Cluster.tolist()
                is_tiger = "tiger2" in usr.Cluster.tolist()
                is_della = "della" in usr.Cluster.tolist()
                della = usr[usr.Cluster == "della"].copy()
                all_physics = "physics" in della.partition.tolist() and \
                              bool(della[della.partition == "physics"].shape[0] == della.shape[0])
                all_not_physics = bool(della[della.partition != "physics"].shape[0] == della.shape[0])
                max_cores = usr["cores"].max()
                edays = self.days_between_emails
                s = f"{g.greeting(user)}"
                s += f"Below are your jobs over the past {edays} days which appear to be using more nodes\n"
                s += "than necessary:"
                s += "\n\n"
                usr = usr.drop(columns=["User", "partition", "cores"])
                usr["Hours"] = usr["Hours"].apply(lambda hrs: round(hrs, 1))
                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([4 * " " + row for row in usr_str.split("\n")])
                s += "\n"
                s += textwrap.dedent("""
                The "Nodes" column shows the number of nodes used to run the job. The
                "Nodes-Needed" column shows the minimum number of nodes needed to run the
                job (these values are calculated based on the number of requested CPU-cores
                while taking into account the CPU memory usage of the job). "Mem-per-Node"
                is the mean CPU memory used per node.

                When possible please try to minimize the number of nodes per job by using all
                of the CPU-cores of each node. This will help to maximize the overall job
                throughput of the cluster.
                """)
                if is_della:
                    if all_not_physics:
                        cores_per_node = 32
                        if min_nodes == 1 and max_cores < cores_per_node:
                            cores_per_node = max_cores
                        s += textwrap.dedent(f"""
                        Della (cpu) is composed of nodes with 32 CPU-cores and 190 GB of CPU memory. If
                        your job requires {cores_per_node*min_nodes} CPU-cores (and you do not have high memory demands) then
                        use, for example:

                            #SBATCH --nodes={min_nodes}
                            #SBATCH --ntasks-per-node={cores_per_node}
                            #SBATCH --ntasks={cores_per_node*min_nodes}

                        For more information about the compute nodes on Della:

                            https://researchcomputing.princeton.edu/systems/della
                        """)
                    elif all_physics:
                        s += textwrap.dedent(f"""
                        Della (physics) is composed of nodes with 40 CPU-cores and 380 GB of CPU memory.
                        If your job requires {40*min_nodes} CPU-cores (and you do not have high memory demands)
                        then use, for example:

                            #SBATCH --nodes={min_nodes}
                            #SBATCH --ntasks-per-node=40
                            #SBATCH --ntasks={40*min_nodes}

                        For more information about the compute nodes on Della:

                            https://researchcomputing.princeton.edu/systems/della
                        """)
                    else:
                        s += textwrap.dedent(f"""
                        Della (physics) is composed of nodes with 40 CPU-cores and 380 GB of CPU memory.
                        while Della (cpu) is composed of nodes with 32 CPU-cores and 190 GB of CPU
                        memory. If your job requires {40*min_nodes} CPU-cores (and you do not have high memory
                        demands) then use, for example:

                            #SBATCH --nodes={min_nodes}
                            #SBATCH --ntasks-per-node=40
                            #SBATCH --ntasks={40*min_nodes}

                        For more information about the compute nodes on Della:

                            https://researchcomputing.princeton.edu/systems/della
                        """)
                if is_tiger:
                    s += textwrap.dedent(f"""
                        Tiger is composed of nodes with 40 CPU-cores and either 192 or 768 GB of
                        CPU memory. If your job requires {40*min_nodes} CPU-cores (and you do not have high memory
                        demands) then use, for example:

                            #SBATCH --nodes={min_nodes}
                            #SBATCH --ntasks-per-node=40
                            #SBATCH --ntasks={40*min_nodes}

                        For more information about the compute nodes on Tiger:

                            https://researchcomputing.princeton.edu/systems/tiger
                        """)
                if is_stellar:
                    cores_per_node = 96
                    if min_nodes == 1 and max_cores < cores_per_node:
                        cores_per_node = max_cores
                    s += textwrap.dedent(f"""
                        Stellar (Intel) is composed of nodes with 96 CPU-cores and 768 GB of CPU memory.
                        If your job requires {min_nodes*cores_per_node} CPU-cores (and you do not have high memory demands)
                        then use, for example:

                            #SBATCH --nodes={min_nodes}
                            #SBATCH --ntasks-per-node={cores_per_node}
                            #SBATCH --ntasks={min_nodes*cores_per_node}

                        For more information about the compute nodes on Stellar:

                            https://researchcomputing.princeton.edu/systems/stellar
                        """)
                s += textwrap.dedent(f"""
                If you are unsure about the meanings of --nodes, --ntasks, --ntasks-per-node and
                --cpus-per-task, see our Slurm webpage:

                    https://researchcomputing.princeton.edu/support/knowledge-base/slurm

                Additionally, see this general overview on parallel computing:

                    https://researchcomputing.princeton.edu/support/knowledge-base/parallel-code

                It is very important to conduct a scaling analysis to find the optimal number
                of nodes and CPU-cores to use for a given parallel job. The calculation of
                "Nodes-Needed" above is based on your choice of the total CPU-cores which
                may not be optimal. For information on conducting a scaling analysis:

                    https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis

                See detailed information about each job by running the \"jobstats\" command:

                    $ jobstats {usr['JobID'].values[0]}

                Consider attending an in-person Research Computing help session for assistance:

                    https://researchcomputing.princeton.edu/support/help-sessions
 
                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)
                send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}")
                send_email(s, "alerts-jobs-aaaalegbihhpknikkw2fkdx6gi@princetonrc.slack.com", subject=f"{self.subject}")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            self.df["hours"] = self.df["hours"].apply(lambda hrs: round(hrs, 1))
            return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
