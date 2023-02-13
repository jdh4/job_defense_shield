import math
from efficiency import cpu_memory_usage
from efficiency import max_cpu_memory_used_per_node
from efficiency import gpu_efficiency
from utils import JOBSTATES
from utils import SECONDS_PER_MINUTE
from utils import SECONDS_PER_HOUR
from utils import MINUTES_PER_HOUR
from utils import HOURS_PER_DAY

def is_fragmented(cluster, partition, cores_per_node, max_mem_used_per_node):
  safety = 0.2
  if cluster == "tiger" and cores_per_node < 40 and max_mem_used_per_node < (1 - safety) * 192:
    return True
  elif cluster == "della" and partition == "physics" and cores_per_node < 40 and max_mem_used_per_node < (1 - safety) * 380:
    return True
  elif cluster == "della" and partition != "physics" and cores_per_node < 28 and max_mem_used_per_node < (1 - safety) * 190:
    return True
  else:
    return False

def min_nodes_needed(cluster, partition, nodes, cores, max_mem_used_per_node):
  safety = 0.2
  if cluster == "della" and partition == "physics":
    min_nodes_by_cores = math.ceil(cores / 40)
    min_nodes_by_memory = min(1, math.ceil(nodes * max_mem_used_per_node / ((1 - safety) * 380)))
    return max(min_nodes_by_cores, min_nodes_by_memory)
  elif cluster == "della" and partition != "physics":
    min_nodes_by_cores = math.ceil(cores / 40)
    min_nodes_by_memory = min(1, math.ceil(nodes * max_mem_used_per_node / ((1 - safety) * 190)))
    return max(min_nodes_by_cores, min_nodes_by_memory)
  elif cluster == "tiger":
    min_nodes_by_cores = math.ceil(cores / 40)
    min_nodes_by_memory = min(1, math.ceil(nodes * max_mem_used_per_node / ((1 - safety) * 192)))
    return max(min_nodes_by_cores, min_nodes_by_memory)
  else:
    return nodes

def multinode_cpu_fragmentation(df):
  cols = ["jobid", "netid", "cluster", "nodes", "cores", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment"]
  fr = df[(df["elapsed-hours"] >= 1) &
          (df["admincomment"] != {}) &
          (df.nodes > 1) &
          (df["gpu-job"] == 0) &
          (df.state != "OUT_OF_MEMORY") &
          (df.partition.isin(["all", "cpu", "ext", "physics", "serial", "pppl", "pu"])) &
          (df.cluster.isin(["della", "stellar", "tiger"]))][cols].copy()
  if fr.empty: return fr
  fr["cores-per-node"] = fr["cores"] / fr["nodes"]
  fr["cores-per-node"] = fr["cores-per-node"].apply(lambda x: round(x, 1))
  fr["memory-tuple"] = fr.apply(lambda row: cpu_memory_usage(row["admincomment"], row["jobid"], row["cluster"]), axis="columns")
  fr["memory-used"]  = fr["memory-tuple"].apply(lambda x: x[0])
  fr["memory-alloc"] = fr["memory-tuple"].apply(lambda x: x[1])
  fr["mean-memory-used-per-node"] = fr["memory-used"] / fr["nodes"]
  fr["mean-memory-per-node"] = fr["mean-memory-used-per-node"].apply(lambda x: round(x, 1))
  fr["max-memory-used-per-node"] = fr.apply(lambda row: max_cpu_memory_used_per_node(row["admincomment"], row["jobid"], row["cluster"]), axis="columns")

  fr = fr[fr.apply(lambda row: is_fragmented(row["cluster"], row["partition"], row["cores-per-node"], row["max-memory-used-per-node"]), axis="columns")]
  fr["min-nodes"] = fr.apply(lambda row: min_nodes_needed(row["cluster"], row["partition"], row["nodes"], row["cores"], row["max-memory-used-per-node"]), axis="columns")
  fr = fr.sort_values(["cluster", "netid"], ascending=[True, False]).rename(columns={"elapsed-hours":"hours"})
  fr = fr[fr["min-nodes"] < fr.nodes]
  fr["max-memory-used-per-node"] = fr["max-memory-used-per-node"].apply(lambda x: f"{x} GB")
  print(fr[["jobid", "netid", "cluster", "min-nodes", "nodes", "cores", "cores-per-node", "mean-memory-used-per-node", "max-memory-used-per-node", "hours"]].to_string(index=False))

  ### EMAIL
  if 0 and args.email:
    for netid in fr.netid:
      vfile = f"{args.files}/fragmentation/{netid}.email.csv"
      last_write_date = datetime(1970, 1, 1)
      if os.path.exists(vfile):
        last_write_date = datetime.fromtimestamp(os.path.getmtime(vfile))
      s = f"{get_first_name(netid)},\n\n"
      if (datetime.now().timestamp() - last_write_date.timestamp() >= 7 * HOURS_PER_DAY * SECONDS_PER_HOUR):
        usr = fr[fr.netid == netid].copy()
        is_della = "della" in usr.cluster.tolist()
        is_tiger = "tiger" in usr.cluster.tolist()
        cols = ["jobid", "netid", "cluster", "cores", "nodes", "min-nodes", "max-memory-used-per-node", "memory-used"]
        renamings = {"jobid":"JobID", "netid":"NetID", "cluster":"Cluster", "min-nodes":"Min-Nodes-Needed", "hours":"Hours", \
                     "cores":"CPU-cores", "nodes":"Nodes", "max-memory-used-per-node":"Max-Memory-Used-per-Node", \
                     "memory-used":"Total-Memory-Usage"}
        usr = usr[cols].rename(columns=renamings)
        # make period clear and number of jobs and rank
        s += "Below are jobs that ran using more nodes than needed in the past 7 days:"
        s += "\n\n"
        s += "\n".join([2 * " " + row for row in usr.to_string(index=False, justify="center").split("\n")])
        s += "\n"
        s += textwrap.dedent(f"""
        The "Min-Nodes-Needed" column shows the minimum number of nodes needed to run
        the job. This is based on the number of CPU-cores that you requested as well
        as the CPU memory usage of the job. The value
        of "Min-Nodes-Needed" is less than that of "Nodes" for all jobs above indicating
        job fragmentation. When a job is ran using more nodes than needed
        it prevents other users from running jobs that require full nodes.
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
          s += "  https://researchcomputing.princeton.edu/systems/della#hardware"
        if is_tiger:
          s += "TigerCPU is composed of nodes with 40 CPU-cores and either 192 or 768 GB of\n"
          s += "CPU memory. For more information about the Tiger cluster:"
          s += "\n\n"
          s += "  https://researchcomputing.princeton.edu/systems/tiger"
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
          #SBATCH --mail-user={netid}@princeton.edu
        
        Replying to this email will open a support ticket with CSES. Let us know if we
        can be of help.
        """)
        #send_email(s, "halverson@princeton.edu", subject=f"Low {xpu.upper()} utilization on {cluster}", sender="cses@princeton.edu")
        #send_email(s,   f"{netid}@princeton.edu", subject="Jobs with zero GPU utilization", sender="cses@princeton.edu")
        print(s)
        usr["email_sent"] = datetime.now().strftime("%m/%d/%Y %H:%M")
        if os.path.exists(vfile):
          curr = pd.read_csv(vfile)
          curr = pd.concat([curr, usr]).drop_duplicates()
          curr.to_csv(vfile, index=False, header=True)
        else:
          usr.to_csv(vfile, index=False, header=True)
      else:
        pass
  print("Exiting fragmentation email routine")
  ### EMAIL

  return fr[["jobid", "netid", "cluster", "min-nodes", "nodes", "cores", "cores-per-node", "mean-memory-used-per-node", "max-memory-used-per-node", "hours"]]


def multinode_gpu_fragmentation(df):
  cols = ["jobid", "netid", "cluster", "nodes", "gpus", "state", "partition", "elapsed-hours", "start-date", "start", "admincomment", "elapsedraw"]
  cond1 = (df["elapsed-hours"] >= 2) & (df.nodes > 1) & (df.gpus > 0) & (df.nodes == df.gpus)
  cond2 = (df["elapsed-hours"] >= 2) & (df.nodes > 1) & (df.gpus > 0) & (df.cluster.isin(["traverse"])) & (df.gpus < 4 * df.nodes)
  m = df[cond1 | cond2][cols].copy()
  m.state = m.state.apply(lambda x: JOBSTATES[x])
  m = m.sort_values(["netid", "start"], ascending=[True, False]).drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  if m.empty: return pd.DataFrame()  # prevents next line from failing
  m.nodes = m.nodes.astype("int64")
  m["eff(%)"] = m.apply(lambda row: gpu_efficiency(row["admincomment"], row["elapsedraw"], row["jobid"], row["cluster"], single=True) if row["admincomment"] != {} else "", axis="columns")
  cols = cols[:7] + ["hours", "eff(%)"]
  return m[cols]
