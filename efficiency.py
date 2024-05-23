import json
import gzip
import base64
import pandas as pd


def get_stats_dict(ss64):
    """Convert the base64-encoded summary statistics to JSON."""
    if (not ss64) or pd.isna(ss64) or ss64 == "JS1:Short" or ss64 == "JS1:None":
        return {}
    return json.loads(gzip.decompress(base64.b64decode(ss64[4:])))


def cpu_efficiency(ss, elapsedraw, jobid, cluster, single=False, precision=1, verbose=True):
    """Return a (CPU time used, CPU time allocated, error code)-tuple for a given job.
       If single=True then return a (CPU time used / CPU time allocated, error code)-tuple.
       The error code is needed since the summary statistics (ss) may be malformed."""
    total = 0
    total_used = 0
    error_code = 0
    for node in ss['nodes']:
        try:
            used  = ss['nodes'][node]['total_time']
            cores = ss['nodes'][node]['cpus']
        except:
            if verbose:
                msg = "Warning: JSON probably missing keys in cpu_efficiency:"
                print(msg, jobid, cluster, flush=True)
            error_code = 1
            return (-1, error_code) if single else (-1, -1, error_code)
        else:
            alloc = elapsedraw * cores  # equal to cputimeraw
            total += alloc
            total_used += used
    if total_used > total:
        error_code = 2
        if verbose:
            msg = "Warning: total_used > total in cpu_efficiency:"
            print(msg, jobid, cluster, total_used, total, flush=True)
    if single:
        return (round(100 * total_used / total, precision), error_code)
    return (total_used, total, error_code)


def gpu_efficiency(ss, elapsedraw, jobid, cluster, single=False, precision=1, verbose=True):
    """Return a (GPU time used, GPU time allocated, error code)-tuple for a given job.
       If single=True then return a (GPU time used / GPU time allocated, error code)-tuple.
       The error code is needed since the summary statistics (ss) may be malformed."""
    total = 0
    total_used = 0
    error_code = 0
    for node in ss['nodes']:
        try:
            gpus = list(ss['nodes'][node]['gpu_utilization'].keys())
        except:
            if verbose:
                msg = "Warning: probably missing keys in gpu_efficiency:"
                print(msg, jobid, cluster, flush=True)
            error_code = 1
            return (-1, error_code) if single else (-1, -1, error_code)
        else:
            for gpu in gpus:
                util = ss['nodes'][node]['gpu_utilization'][gpu]
                total      += elapsedraw
                total_used += elapsedraw * (float(util) / 100)
    if total_used > total:
        error_code = 2
        if verbose:
            msg = "Warning: total_used > total in gpu_efficiency:"
            print(msg, jobid, cluster, total_used, total, flush=True)
    if single:
        return (round(100 * total_used / total, precision), error_code)
    return (total_used, total, error_code)


def gpu_memory_usage_eff_tuples(d, jobid, cluster, precision=1):
  all_gpus = []
  for node in d['nodes']:
    try:
      used  = d['nodes'][node]['gpu_used_memory']
      alloc = d['nodes'][node]['gpu_total_memory']
      util  = d['nodes'][node]['gpu_utilization']
    except:
      print("GPU memory trouble:", jobid, cluster, flush=True)
      return [(0, 0, 0)]
    else:
      assert sorted(list(used.keys())) == sorted(list(alloc.keys())), "keys do not match"
      for g in used.keys():
        all_gpus.append((round(used[g] / 1024**3, precision), round(alloc[g] / 1024**3, precision), float(util[g])))
        if used[g] > alloc[g]: print("GPU memory:", jobid, cluster, used[g], alloc[g], flush=True)
        if util[g] > 100 or util[g] < 0: print("GPU util:", jobid, cluster, util[g], flush=True)
  return all_gpus

def cpu_memory_usage(d, jobid, cluster, precision=0):
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
  return (round(total_used / 1024**3, precision), round(total / 1024**3, precision))

def max_cpu_memory_used_per_node(d, jobid, cluster, precision=0):
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
  return round(max(mem_per_node) / 1024**3, precision)

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
