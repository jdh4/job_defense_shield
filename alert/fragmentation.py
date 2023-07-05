import math
import textwrap
from base import Alert
from efficiency import cpu_memory_usage
from efficiency import max_cpu_memory_used_per_node
from efficiency import gpu_efficiency
from efficiency import cpu_nodes_with_zero_util
from utils import get_first_name
from utils import send_email
from utils import add_dividers
from utils import SECONDS_PER_MINUTE
from utils import SECONDS_PER_HOUR
from utils import MINUTES_PER_HOUR
from utils import HOURS_PER_DAY


class MultinodeCPUFragmentation(Alert):

    """Find multinode CPU jobs that use too many nodes. Ignore jobs
       with 0% CPU utilization on a node since those will captured
       by a different alert.

       The number of processors per node can vary. One needs to keep
       this in mind when computing the maximum memory used per node.
    """

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, kwargs)

    @staticmethod
    def is_fragmented(cluster, partition, cores_per_node, mem_per_node_used):
        safety = 0.2
        cores_frac = 0.8
        if cluster == "tiger" and \
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
           cores_per_node < cores_frac * 32 and \
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
            min_nodes_by_memory = min(1, math.ceil(nodes * mem_per_node_used / ((1 - safety) * memory_max)))
            return max(min_nodes_by_cores, min_nodes_by_memory)
        safety = 0.2
        if cluster == "della" and partition == "physics":
            return cores_vs_memory(cores, 40, mem_per_node_used, 380)
        elif cluster == "della" and partition != "physics":
            return cores_vs_memory(cores, 32, mem_per_node_used, 190)
        elif cluster == "tiger":
            return cores_vs_memory(cores, 40, mem_per_node_used, 192)
        elif cluster == "stellar" and partition in ("all", "pppl", "pu"):
            return cores_vs_memory(cores, 96, mem_per_node_used, 768)
        else:
            return nodes

    def _filter_and_add_new_fields(self):
        # filter the dataframe
        self.df = self.df[(self.df["elapsed-hours"] >= 1) &
                          (self.df["admincomment"] != {}) &
                          (self.df.nodes > 1) &
                          (self.df["gpu-job"] == 0) &
                          (self.df.state != "OUT_OF_MEMORY") &
                          (self.df.partition.isin(["all", "cpu", "ext", "physics", "pppl", "pu"])) &
                          (self.df.cluster.isin(["della", "stellar", "tiger"]))].copy()
        # add new fields
        if not self.df.empty:
            self.df["nodes-unused"] = self.df.admincomment.apply(cpu_nodes_with_zero_util)
            self.df = self.df[self.df["nodes-unused"] == 0]
            self.df["cores-per-node"] = self.df["cores"] / self.df["nodes"]
            self.df["cores-per-node"] = self.df["cores-per-node"].apply(lambda x: round(x, 1))
            self.df["memory-tuple"] = self.df.apply(lambda row:
                                                    cpu_memory_usage(row["admincomment"],
                                                                     row["jobid"],
                                                                     row["cluster"]),
                                                                     axis="columns")
            self.df["memory-used"]  = self.df["memory-tuple"].apply(lambda x: x[0])
            self.df["memory-alloc"] = self.df["memory-tuple"].apply(lambda x: x[1])
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
            self.df = self.df.sort_values(["cluster", "netid"], ascending=[True, False])
            self.df = self.df[self.df["min-nodes"] < self.df["nodes"]]
            self.df["memory-per-node-used"] = self.df["memory-per-node-used"].apply(lambda x: f"{x} GB")
            self.df["cores-per-node"] = self.df["cores-per-node"].apply(lambda x: str(x).replace(".0", ""))
            self.df = self.df.rename(columns={"netid":"NetID",
                                              "elapsed-hours":"hours",
                                              "memory-per-node-used":"mem-per-node-used"})
            cols = ["jobid",
                    "NetID",
                    "cluster",
                    "nodes",
                    "min-nodes",
                    "cores-per-node",
                    "mem-per-node-used"]
            self.df = self.df[cols]

    def send_emails_to_users(self):
        for user in self.df.NetID.unique():
            vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
            if self.has_sufficient_time_passed_since_last_email(vfile):
                usr = self.df[self.df.NetID == user].copy()
                usr = usr.drop(columns=["NetID"])
                renamings = {"jobid":"JobID",
                             "netid":"NetID",
                             "cluster":"Cluster",
                             "nodes":"Nodes",
                             "min-nodes":"Min-Nodes-Needed",
                             "cores-per-node":"Cores-per-Node",
                             "mem-per-node-used":"Memory-per-Node-Used",
                             "hours":"Hours"}
                usr = usr.rename(columns=renamings)
                is_della   = "della" in usr.cluster.tolist()
                is_physics = "physics" in usr.partition.tolist()
                is_stellar = "stellar" in usr.cluster.tolist()
                is_tiger   = "tiger" in usr.cluster.tolist()
                edays = self.days_between_emails
                s = f"{get_first_name(user)},\n\n"
                s += "Below are your jobs that ran in the past {edays} using more nodes than needed:"
                s += "\n\n"
                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([2 * " " + row for row in usr_str.split("\n")])
                s += "\n"
                s += textwrap.dedent(f"""
                The "Min-Nodes-Needed" column shows the minimum number of nodes needed to run
                the job. This is based on the number of CPU-cores that you requested as well
                as the CPU memory usage of the job. The value
                of "Min-Nodes-Needed" is less than that of "Nodes" for all jobs above indicating
                job fragmentation. When a job is ran using more nodes than needed
                it prevents other users from running jobs that require full nodes and in
                some cases it introduces data communication inefficiencies.
                For future jobs, please try to use the minimum number of nodes for a given job by
                decreasing the values of the --nodes, --ntasks, --ntasks-per-node Slurm directives.
                This will eliminate job fragmentation which will allow all users to use the
                cluster effectively. When a job is divided over more nodes than it needs to be
                it prevents other users from running jobs that require full nodes.
                """)
                s += "\n"
                if is_della:
                  s += "Della is mostly composed of nodes with 32 CPU-cores and 190 GB of CPU memory.\n"
                  s+= "For more information about the nodes on Della:"
                  s += "\n\n"
                  s += "  https://researchcomputing.princeton.edu/systems/della"
                if is_tiger:
                  s += "TigerCPU is composed of nodes with 40 CPU-cores and either 192 or 768 GB of\n"
                  s += "CPU memory. For more information about the Tiger cluster:"
                  s += "\n\n"
                  s += "  https://researchcomputing.princeton.edu/systems/tiger"
                if is_physics:
                  s += "Della (physics) is composed of nodes with 40 CPU-cores and 380 GB of\n"
                  s += "CPU memory. For more information about the Della cluster:"
                  s += "\n\n"
                  s += "  https://researchcomputing.princeton.edu/systems/della"
                if is_stellar:
                  s += "Stellar (Intel) is composed of nodes with 96 CPU-cores and 768 GB of\n"
                  s += "CPU memory. For more information about the Stellar cluster:"
                  s += "\n\n"
                  s += "  https://researchcomputing.princeton.edu/systems/stellar"
                s += "\n"
                s += textwrap.dedent(f"""
                If you are unsure about the meanings of --nodes, --ntasks, --ntasks-per-node
                and --cpus-per-task, see these webpages:
                  https://researchcomputing.princeton.edu/support/knowledge-base/parallel-code
                  https://researchcomputing.princeton.edu/support/knowledge-base/slurm
                The optimal number nodes and CPU-cores to use for a given parallel code can be
                obtained by conducting a scaling analysis:
                  https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis
                """)
                s += textwrap.dedent(f"""
                Add the following lines to your Slurm scripts to receive an email report with
                node information after each job finishes:

                  #SBATCH --mail-type=end
                  #SBATCH --mail-user={user}@princeton.edu

                Consider attending an in-person Research Computing help session for assistance:

                     https://researchcomputing.princeton.edu/support/help-sessions
                
                Replying to this automated email will open a support ticket with Research
                Computing. Let us know if we can be of help.
                """)
                #send_email(s,   f"{user}@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                send_email(s, "halverson@princeton.edu", subject=f"{self.subject}", sender="cses@princeton.edu")
                print(s)

                # append the new violations to the log file
                Alert.update_violation_log(usr, vfile)

    def generate_report_for_admins(self, title: str, keep_index: bool=False) -> str:
        if self.df.empty:
            return ""
        else:
            return add_dividers(self.df.to_string(index=keep_index, justify="center"), title)
