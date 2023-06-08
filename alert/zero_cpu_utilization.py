from utils import JOBSTATES

# if not sending emails based on actively running jobs with zero CPU then need to send the below
def cpu_jobs_zero_util(df, cluster, partitions):
  zu = df[(df.cluster == cluster) & \
          (df["elapsed-hours"] >= 1) & \
          (df.partition.isin(partitions))].copy()
  zu = zu[zu.admincomment != {}]  # ignore running jobs
  zu["interactive"] = zu["jobname"].apply(lambda x: True if x.startswith("sys/dashboard") or \
                                                            x.startswith("interactive") else False)
  def cpu_nodes_with_zero_util(d):
    ct = 0
    for node in d['nodes']:
      try:
        cpu_time = d['nodes'][node]['total_time']
      except:
        print("total_time not found")
        return 0
      else:
        if float(cpu_time) == 0: ct += 1
    return ct

  zu["nodes-unused"] = zu.admincomment.apply(cpu_nodes_with_zero_util)
  zu = zu[zu["nodes-unused"] > 0].rename(columns={"elapsed-hours":"hours"}).sort_values(by="netid")
  zu.state = zu.state.apply(lambda x: JOBSTATES[x])
  zu.nodes = zu.nodes.astype("int64")
  cols = ["netid", "nodes", "nodes-unused", "jobid", "state", "hours", "interactive", "start-date"]
  return zu[cols]
