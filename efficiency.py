import json
import gzip
import base64
import pandas as pd

def get_stats_dict(x):
  if not x or pd.isna(x) or x == "JS1:Short" or x == "JS1:None":
    return {}
  else:
    return json.loads(gzip.decompress(base64.b64decode(x[4:])))

def cpu_efficiency(d, elapsedraw, jobid, cluster, single=False):
  total = 0
  total_used = 0
  for node in d['nodes']:
    try:
      used  = d['nodes'][node]['total_time']
      cores = d['nodes'][node]['cpus']
    except:
      print("total_time not found for {jobid} on {cluster}")
      return (0, 1)  # dummy values that avoid division by zero later
    else:
      alloc = elapsedraw * cores  # equal to cputimeraw
      total += alloc
      total_used += used
  if total_used > total: print("E: CPU efficiency:", jobid, cluster, total_used, total, flush=True)
  return round(100 * total_used / total, 1) if single else (total_used, total)

def gpu_efficiency(d, elapsedraw, jobid, cluster, single=False):
  total = 0
  total_used = 0
  for node in d['nodes']:
    try:
      gpus = list(d['nodes'][node]['gpu_utilization'].keys())
    except:
      print(f"gpu_utilization not found for {jobid} on {cluster}")
      return 0 if single else (0, 1)  # dummy values that avoid division by zero later
    else:
      for gpu in gpus:
        util = d['nodes'][node]['gpu_utilization'][gpu]
        total      += elapsedraw
        total_used += elapsedraw * (float(util) / 100)
  if total_used > total: print("GPU efficiency:", jobid, cluster, total_used, total, flush=True)
  return round(100 * total_used / total, 1) if single else (total_used, total)

def cpu_memory_usage(d, jobid, cluster):
  total = 0
  total_used = 0
  for node in d['nodes']:
    try:
      used  = d['nodes'][node]['used_memory']
      alloc = d['nodes'][node]['total_memory']
    except:
      print("used_memory not found")
      return (0, 0)
    else:
      total += alloc
      total_used += used
  if total_used > total: print("CPU memory:", jobid, cluster, total_used, total, flush=True)
  return (round(total_used / 1024**3), round(total / 1024**3))

def max_cpu_memory_used_per_node(d, jobid, cluster):
  total = 0
  total_used = 0
  mem_per_node = []
  for node in d['nodes']:
    try:
      used  = d['nodes'][node]['used_memory']
      alloc = d['nodes'][node]['total_memory']
    except:
      print("used_memory not found")
      return (0, 0)
    else:
      mem_per_node.append(used)
    if used > alloc: print("CPU memory:", jobid, cluster, total_used, total, flush=True)
  return round(max(mem_per_node) / 1024**3)

def num_gpus_with_zero_util(d):
  ct = 0
  for node in d['nodes']:
    try:
      gpus = list(d['nodes'][node]['gpu_utilization'].keys())
    except:
      print(f"gpu_utilization not found: node is {node}")
      return 0
    else:
      for gpu in gpus:
        util = d['nodes'][node]['gpu_utilization'][gpu]
        if float(util) == 0: ct += 1
  return ct
