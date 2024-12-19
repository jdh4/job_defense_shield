import os
import sys
import argparse
import subprocess
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import yaml

from utils import SECONDS_PER_MINUTE
from utils import SECONDS_PER_HOUR
from utils import gpus_per_job
from utils import send_email
from utils import show_history_of_emails_sent
from workday import WorkdayFactory
from efficiency import get_stats_dict
from raw_job_data import SlurmSacct
from cleaner import SacctCleaner

from alert.zero_gpu_utilization import ZeroGpuUtilization
from alert.gpu_model_too_powerful import GpuModelTooPowerful
from alert.zero_util_gpu_hours import ZeroUtilGPUHours
from alert.excess_cpu_memory import ExcessCPUMemory
from alert.zero_cpu_utilization import ZeroCPU
from alert.most_gpus import MostGPUs
from alert.most_cores import MostCores
from alert.utilization_overview import UtilizationOverview
from alert.utilization_by_slurm_account import UtilizationBySlurmAccount
from alert.longest_queued import LongestQueuedJobs
from alert.jobs_overview import JobsOverview
from alert.excessive_time_limits import ExcessiveTimeLimits
from alert.serial_allocating_multiple_cores import SerialAllocatingMultipleCores
from alert.multinode_cpu_fragmentation import MultinodeCpuFragmentation
from alert.multinode_gpu_fragmentation import MultinodeGpuFragmentation
from alert.compute_efficiency import LowEfficiencyCPU
from alert.compute_efficiency import LowEfficiencyGPU
from alert.too_many_cores_per_gpu import TooManyCoresPerGpu


def raw_dataframe_from_sacct(flags, start_date, fields, renamings=[], numeric_fields=[], use_cache=False):
    fname = f"cache_sacct_{start_date.strftime('%Y%m%d')}.csv"
    if use_cache and os.path.exists(fname):
        print(f"\n### USING CACHE FILE: {fname} ###\n", flush=True)
        rw = pd.read_csv(fname, low_memory=False)
    else:
        ymd = start_date.strftime('%Y-%m-%d')
        hms = start_date.strftime('%H:%M:%S')
        cmd = f"sacct {flags} -S {ymd}T{hms} -E now -o {fields}"
        if use_cache:
            print("\nCalling sacct (which may require several seconds) ... ", end="", flush=True)
        output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=120, text=True, check=True)
        if use_cache:
            print("done.", flush=True)
        lines = output.stdout.split('\n')
        if lines != [] and lines[-1] == "":
            lines = lines[:-1]
        cols = fields.split(",")
        rw = pd.DataFrame([line.split("|")[:len(cols)] for line in lines])
        rw.columns = cols
        rw.rename(columns=renamings, inplace=True)
        rw = rw[pd.notna(rw.elapsedraw)]
        rw = rw[rw.elapsedraw.str.isnumeric()]
        rw[numeric_fields] = rw[numeric_fields].apply(pd.to_numeric)
        if use_cache:
            rw.to_csv(fname, index=False)
    return rw


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Job Defense Shield')
    parser.add_argument('--zero-cpu-utilization', action='store_true', default=False,
                        help='Identify completed CPU jobs with zero utilization')
    parser.add_argument('--zero-gpu-utilization', action='store_true', default=False,
                        help='Identify running GPU jobs with zero utilization')
    parser.add_argument('--zero-util-gpu-hours', action='store_true', default=False,
                        help='Identify users with the most zero GPU utilization hours')
    parser.add_argument('--low-cpu-efficiency', action='store_true', default=False,
                        help='Identify users with low CPU efficiency')
    parser.add_argument('--low-gpu-efficiency', action='store_true', default=False,
                        help='Identify users with low GPU efficiency')
    parser.add_argument('--excess-cpu-memory', action='store_true', default=False,
                        help='Identify users that are allocating too much CPU memory')
    parser.add_argument('--gpu-model-too-powerful', action='store_true', default=False,
                        help='Identify jobs that should use less powerful GPUs')
    parser.add_argument('--multinode-cpu-fragmentation', action='store_true', default=False,
                        help='Identify CPU jobs that are split across too many nodes')
    parser.add_argument('--multinode-gpu-fragmentation', action='store_true', default=False,
                        help='Identify GPU jobs that are split across too many nodes')
    parser.add_argument('--excessive-time', action='store_true', default=False,
                        help='Identify users with excessive run time limits')
    parser.add_argument('--serial-allocating-multiple', action='store_true', default=False,
                        help='Indentify serial codes allocating multiple CPU-cores')
    parser.add_argument('--too-many-cores-per-gpu', action='store_true', default=False,
                        help='Indentify jobs using too many CPU-cores per GPU')
    parser.add_argument('--utilization-overview', action='store_true', default=False,
                        help='Generate a utilization report by cluster and partition')
    parser.add_argument('--utilization-by-slurm-account', action='store_true', default=False,
                        help='Generate a utilization report by cluster, partition and account')
    parser.add_argument('--longest-queued', action='store_true', default=False,
                        help='List the longest queued jobs')
    parser.add_argument('--most-cores', action='store_true', default=False,
                        help='List the largest jobs by number of allocated CPU-cores')
    parser.add_argument('--most-gpus', action='store_true', default=False,
                        help='List the largest jobs by number of allocated GPUs')
    parser.add_argument('--jobs-overview', action='store_true', default=False,
                        help='List the users with the most jobs')
    parser.add_argument('-d', '--days', type=int, default=7, metavar='N',
                        help='Use job data over N previous days from now (default: 7)')
    parser.add_argument('-S', '--starttime', type=str, default=None,
                        help='Start date/time of window (e.g., 2025-01-01T09:00:00)')
    parser.add_argument('-E', '--endtime', type=str, default=None,
                        help='End date/time of window (e.g., 2025-01-31T22:00:00)')
    parser.add_argument('-M', '--clusters', type=str, default="all",
                        help='Specify cluster(s) (e.g., --clusters=frontier,summit)')
    parser.add_argument('-r', '--partition', type=str, default="",
                        help='Specify partition(s) (e.g., --partition=cpu,bigmem)')
    parser.add_argument('--config-file', type=str, default=None,
                        help='Absolute path to the configuration file')
    parser.add_argument('--email', action='store_true', default=False,
                        help='Send email alerts to users')
    parser.add_argument('--report', action='store_true', default=False,
                        help='Send an email report to administrators')
    parser.add_argument('--check', action='store_true', default=False,
                        help='Show the history of emails sent to users')
    parser.add_argument('-s', '--strict-start', action='store_true', default=False,
                        help='Only include usage during time window and not before')
    args = parser.parse_args()

    # read configuration file
    jds_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cwd_path = os.path.join(os.getcwd(), "config.yaml")
    if args.config_file and os.path.isfile(args.config_file):
        print(f"Configuration file: {args.config_file}")
        with open(args.config_file, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
    elif args.config_file and not os.path.isfile(args.config_file):
        print(f"Configuration file does not exist: {args.config_file}. Exiting ...")
        sys.exit()
    elif args.config_file is None and os.path.isfile(jds_path):
        print(f"Configuration file: {jds_path}")
        with open(jds_path, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
    elif args.config_file is None and os.path.isfile(cwd_path):
        print(f"Configuration file: {cwd_path}")
        with open(cwd_path, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
    else:
        print("Configuration file not found. Exiting ...")
        sys.exit()
    for key in cfg.keys():
        if any([c.isnumeric() for c in key]):
            print(4 * " " + key)

    greeting_method = cfg["greeting-method"]
    violation_logs_path = cfg["violation-logs-path"]
    workday_method = cfg["workday-method"]
    is_workday = WorkdayFactory().create_workday(workday_method).is_workday()
 
    #######################
    ## CHECK EMAILS SENT ##
    #######################
    if args.check:
        if args.days == 7:
            print("\n\nINFO: Checking with --days=60, instead of the default of 7.\n")
            args.days = 60
        if args.zero_gpu_utilization:
            show_history_of_emails_sent(violation_logs_path,
                                        "zero_gpu_utilization",
                                        "ZERO GPU UTILIZATION OF A RUNNING JOB",
                                        args.days)
        if args.zero_cpu_utilization:
            show_history_of_emails_sent(violation_logs_path,
                                        "zero_cpu_utilization",
                                        "ZERO CPU UTILIZATION OF A RUNNING JOB",
                                        args.days)
        if args.gpu_model_too_powerful:
            show_history_of_emails_sent(violation_logs_path,
                                        "gpu_model_too_powerful",
                                        "GPU MODEL TOO POWERFUL",
                                        args.days)
        if args.low_cpu_efficiency:
            show_history_of_emails_sent(violation_logs_path,
                                        "low_cpu_efficiency",
                                        "LOW CPU EFFICIENCY",
                                        args.days)
        if args.low_gpu_efficiency:
            show_history_of_emails_sent(violation_logs_path,
                                        "low_gpu_efficiency",
                                        "LOW GPU EFFICIENCY",
                                        args.days)
        if args.zero_util_gpu_hours:
            show_history_of_emails_sent(violation_logs_path,
                                        "zero_util_gpu_hours",
                                        "ZERO UTILIZATION GPU-HOURS",
                                        args.days)
        if args.multinode_cpu_fragmentation:
            show_history_of_emails_sent(violation_logs_path,
                                        "multinode_cpu_fragmentation",
                                        "MULTINODE CPU FRAGMENTATION",
                                        args.days)
        if args.multinode_gpu_fragmentation:
            show_history_of_emails_sent(violation_logs_path,
                                        "multinode_gpu_fragmentation",
                                        "MULTINODE GPU FRAGMENTATION",
                                        args.days)
        if args.excess_cpu_memory:
            show_history_of_emails_sent(violation_logs_path,
                                        "excess_cpu_memory",
                                        "EXCESS CPU MEMORY",
                                        args.days)
        if args.serial_allocating_multiple:
            show_history_of_emails_sent(violation_logs_path,
                                        "serial_allocating_multiple",
                                        "SERIAL CODE ALLOCATING MULTIPLE CPU-CORES",
                                        args.days)
        if args.too_many_cores_per_gpu:
            show_history_of_emails_sent(violation_logs_path,
                                        "too_many_cores_per_gpu",
                                        "TOO MANY CPU-CORES PER GPU",
                                        args.days)
        if args.excessive_time:
            show_history_of_emails_sent(violation_logs_path,
                                        "excessive_time_limits",
                                        "EXCESSIVE TIME LIMITS",
                                        args.days)
        if args.most_gpus or args.most_cores or args.longest_queued:
            print("Nothing to check for --most-gpus, --most-cores or --longest-queued.")
        sys.exit()

    # pandas display settings
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    start_date = datetime.now() - timedelta(days=args.days)
    fields = ["jobid",
              "user",
              "cluster",
              "account",
              "partition",
              "cputimeraw",
              "elapsedraw",
              "timelimitraw",
              "nnodes",
              "ncpus",
              "alloctres",
              "submit",
              "eligible",
              "start",
              "end",
              "qos",
              "state",
              "admincomment",
              "jobname"]
    # jobname must be last in list below to catch "|" characters in jobname
    assert fields[-1] == "jobname"
    fields = ",".join(fields)

    use_cache = False if (args.email or args.report) else True
    raw = SlurmSacct(args.days, args.starttime, args.endtime, fields, args.clusters, args.partition)
    raw = raw.get_job_data()

    # clean the raw data
    field_renamings = {"cputimeraw":"cpu-seconds",
                       "nnodes":"nodes",
                       "ncpus":"cores",
                       "timelimitraw":"limit-minutes"}
    partition_renamings = cfg["partition_renamings"]
    raw = SacctCleaner(raw, field_renamings, partition_renamings).clean()
    df = raw.copy()
    num_nulls = df.isnull().sum().sum()
    if num_nulls:
        print(f"Number of null values in df: {num_nulls}")

    if args.strict_start:
        # remove usage before the start of the time window
        df["secs-from-start"] = df["start"] - start_date.timestamp()
        df["secs-from-start"] = df["secs-from-start"].apply(lambda x: x if x < 0 else 0)
        df["elapsedraw"] = df["elapsedraw"] + df["secs-from-start"]

    def add_new_and_derived_fields(df):
        df["cpu-seconds"] = df["elapsedraw"] * df["cores"]
        df["gpus"] = df.alloctres.apply(gpus_per_job)
        df["gpu-seconds"] = df["elapsedraw"] * df["gpus"]
        df["gpu-job"] = np.where((df["alloctres"].str.contains("gres/gpu=")) & (~df["alloctres"].str.contains("gres/gpu=0")), 1, 0)
        df["cpu-only-seconds"] = np.where(df["gpus"] == 0, df["cpu-seconds"], 0)
        df["elapsed-hours"] = df["elapsedraw"] / SECONDS_PER_HOUR
        df.loc[df["start"] != "Unknown", "start-date"] = pd.to_datetime(df["start"].astype(int), unit='s').dt.strftime("%a %-m/%d")
        df["cpu-waste-hours"] = np.round((df["limit-minutes"] * SECONDS_PER_MINUTE - df["elapsedraw"]) * df["cores"] / SECONDS_PER_HOUR)
        df["gpu-waste-hours"] = np.round((df["limit-minutes"] * SECONDS_PER_MINUTE - df["elapsedraw"]) * df["gpus"] / SECONDS_PER_HOUR)
        df["cpu-alloc-hours"] = np.round(df["limit-minutes"] * SECONDS_PER_MINUTE * df["cores"] / SECONDS_PER_HOUR)
        df["gpu-alloc-hours"] = np.round(df["limit-minutes"] * SECONDS_PER_MINUTE * df["gpus"] / SECONDS_PER_HOUR)
        df["cpu-hours"] = df["cpu-seconds"] / SECONDS_PER_HOUR
        df["gpu-hours"] = df["gpu-seconds"] / SECONDS_PER_HOUR
        df["admincomment"] = df["admincomment"].apply(get_stats_dict)
        return df

    df = add_new_and_derived_fields(df)
    df.reset_index(drop=True, inplace=True)

    fmt = "%a %b %-d"
    s = f"{start_date.strftime(fmt)} - {datetime.now().strftime(fmt)}"

    ############################################
    ## RUNNING JOBS WITH ZERO GPU UTILIZATION ##
    ############################################
    if args.zero_gpu_utilization:
        alerts = [alert for alert in cfg.keys() if "zero-gpu-utilization" in alert]
        for alert in alerts:
            zero_gpu = ZeroGpuUtilization(df,
                                          days_between_emails=args.days,
                                          violation="zero_gpu_utilization",
                                          vpath=violation_logs_path,
                                          subject="Jobs with Zero GPU Utilization",
                                          **cfg[alert])
            if args.email:
                zero_gpu.send_emails_to_users(greeting_method)

    ################################
    ## ZERO UTILIZATION GPU-HOURS ##
    ################################
    if args.zero_util_gpu_hours:
        alerts = [alert for alert in cfg.keys() if "zero-util-gpu-hours" in alert]
        for alert in alerts:
            zero_gpu_hours = ZeroUtilGPUHours(df,
                                   days_between_emails=args.days,
                                   violation="zero_util_gpu_hours",
                                   vpath=violation_logs_path,
                                   subject="GPU-hours at 0% Utilization",
                                   **cfg[alert])
            if args.email and is_workday:
                zero_gpu_hours.send_emails_to_users(greeting_method)
            title="Zero Utilization GPU-Hours"
            s += zero_gpu_hours.generate_report_for_admins(title,
                                                           start_date,
                                                           keep_index=True)


    #################################
    ## MULTINODE CPU FRAGMENTATION ##
    #################################
    if args.multinode_cpu_fragmentation:
        alerts = [alert for alert in cfg.keys() if "multinode-cpu-fragmentation" in alert]
        for alert in alerts:
            cpu_frag = MultinodeCpuFragmentation(df,
                                   days_between_emails=args.days,
                                   violation="multinode_cpu_fragmentation",
                                   vpath=violation_logs_path,
                                   subject="Multinode CPU Jobs with Fragmentation",
                                   **cfg[alert])
            if args.email and is_workday:
                cpu_frag.send_emails_to_users(greeting_method)
            title = "Multinode CPU Jobs with Fragmentation"
            s += cpu_frag.generate_report_for_admins(title, keep_index=False)


    #################################
    ## MULTINODE GPU FRAGMENTATION ##
    #################################
    if args.multinode_gpu_fragmentation:
        alerts = [alert for alert in cfg.keys() if "multinode-gpu-fragmentation" in alert]
        for alert in alerts:
            gpu_frag = MultinodeGpuFragmentation(df,
                                   days_between_emails=args.days,
                                   violation="multinode_gpu_fragmentation",
                                   vpath=violation_logs_path,
                                   subject="Multinode GPU Jobs with Fragmentation",
                                   **cfg[alert])
            if args.email and is_workday:
                gpu_frag.send_emails_to_users(greeting_method)
            title = "Multinode GPU Jobs with Fragmentation"
            s += gpu_frag.generate_report_for_admins(title)


    ########################
    ## LOW CPU EFFICIENCY ##
    ########################
    if args.low_cpu_efficiency:
        alerts = [alert for alert in cfg.keys() if "low-cpu-efficiency" in alert]
        for alert in alerts:
            low_cpu = LowEfficiencyCPU(df,
                                       days_between_emails=args.days,
                                       violation="low_cpu_efficiency",
                                       vpath=violation_logs_path,
                                       subject="Jobs with Low CPU Efficiency",
                                       **cfg[alert])
            if args.email and is_workday:
                low_cpu.send_emails_to_users(greeting_method)
            title = "Low CPU Efficiencies"
            s += low_cpu.generate_report_for_admins(title, keep_index=True)

    ########################
    ## LOW GPU EFFICIENCY ##
    ########################
    if args.low_gpu_efficiency:
        alerts = [alert for alert in cfg.keys() if "low-gpu-efficiency" in alert]
        for alert in alerts:
            low_gpu = LowEfficiencyGPU(df,
                                       days_between_emails=args.days,
                                       violation="low_gpu_efficiency",
                                       vpath=violation_logs_path,
                                       subject="Jobs with Low GPU Efficiency",
                                       **cfg[alert])
            if args.email and is_workday:
                low_gpu.send_emails_to_users(greeting_method)
            title = "Low GPU Efficiencies"
            s += low_gpu.generate_report_for_admins(title, keep_index=True)


    #######################
    ## EXCESS CPU MEMORY ##
    #######################
    if args.excess_cpu_memory:
        alerts = [alert for alert in cfg.keys() if "excess-cpu-memory" in alert]
        for alert in alerts: 
            mem_hours = ExcessCPUMemory(df,
                                        days_between_emails=args.days,
                                        violation="excess_cpu_memory",
                                        vpath=violation_logs_path,
                                        subject="Jobs Requesting Too Much CPU Memory",
                                        **cfg[alert])
            if args.email and is_workday:
                mem_hours.send_emails_to_users(greeting_method)
            title = "TB-Hours (1+ hour jobs, ignoring approximately full node jobs)"
            s += mem_hours.generate_report_for_admins(title, keep_index=True)


    ###########################################
    ## SERIAL CODE ALLOCATING MULTIPLE CORES ##
    ###########################################
    if args.serial_allocating_multiple:
        alerts = [alert for alert in cfg.keys() if "serial-allocating-multiple" in alert]
        for alert in alerts:
            serial = SerialAllocatingMultipleCores(df,
                                     days_between_emails=args.days,
                                     violation="serial_allocating_multiple",
                                     vpath=violation_logs_path,
                                     subject="Serial Jobs Allocating Multiple CPU-cores",
                                     **cfg[alert])
            if args.email and is_workday:
                serial.send_emails_to_users(greeting_method)
            title = "Serial Jobs Allocating Multiple CPU-cores"
            s += serial.generate_report_for_admins(title, keep_index=True)


    ##########################
    ## ZERO CPU UTILIZATION ##
    ##########################
    if args.zero_cpu_utilization:
        alerts = [alert for alert in cfg.keys() if "zero-cpu-utilization" in alert]
        for alert in alerts:
            zero_cpu = ZeroCPU(df,
                               days_between_emails=args.days,
                               violation="zero_cpu_utilization",
                               vpath=violation_logs_path,
                               subject="Jobs with Zero CPU Utilization",
                               **cfg[alert])
            if args.email and is_workday:
                zero_cpu.send_emails_to_users(greeting_method)
            title = "Jobs with Zero CPU Utilization"
            s += zero_cpu.generate_report_for_admins(title, keep_index=False)


    ############################
    ## TOO MANY CORES PER GPU ##
    ############################
    if args.too_many_cores_per_gpu:
        alerts = [alert for alert in cfg.keys() if "too-many-cores-per-gpu" in alert]
        for alert in alerts:
            cpg = TooManyCoresPerGpu(df,
                                     days_between_emails=args.days,
                                     violation="too_many_cores_per_gpu",
                                     vpath=violation_logs_path,
                                     subject="Consider Using Fewer CPU-Cores per GPU",
                                     **cfg[alert])
            if args.email and is_workday:
                cpg.send_emails_to_users(greeting_method)
            title = "Too Many Cores Per GPU"
            s += cpg.generate_report_for_admins(title)


    ###########################
    ## EXCESSIVE TIME LIMITS ##
    ###########################
    if args.excessive_time:
        alerts = [alert for alert in cfg.keys() if "excessive-time" in alert]
        for alert in alerts:
            time_limits = ExcessiveTimeLimits(df,
                                              days_between_emails=args.days,
                                              violation="excessive_time_limits",
                                              vpath=violation_logs_path,
                                              subject="Requesting Too Much Time for Jobs",
                                              **cfg[alert])
            if args.email and is_workday:
                time_limits.send_emails_to_users(greeting_method)
            title = "Excessive Time Limits"
            s += time_limits.generate_report_for_admins(title)


    ############################
    ## GPU MODEL TOO POWERFUL ##
    ############################
    if args.gpu_model_too_powerful:
        alerts = [alert for alert in cfg.keys() if "gpu-model-too-powerful" in alert]
        for alert in alerts:
            too_power = GpuModelTooPowerful(df,
                                            days_between_emails=args.days,
                                            violation="gpu_model_too_powerful",
                                            vpath=violation_logs_path,
                                            subject="Jobs with GPU Model Too Powerful",
                                            **cfg[alert])
            if args.email and is_workday:
                too_power.send_emails_to_users(greeting_method)
            s += too_power.generate_report_for_admins("GPU Model Too Powerful")


    ################################
    ## INFO: UTILIZATION OVERVIEW ##
    ################################
    if args.utilization_overview:
        util = UtilizationOverview(df,
                                   days_between_emails=args.days,
                                   violation="null",
                                   vpath=violation_logs_path,
                                   subject="")
        title = "Utilization Overview"
        s += util.generate_report_for_admins(title)

    ##################################
    ## UTILIZATION BY SLURM ACCOUNT ##
    ##################################
    if args.utilization_by_slurm_account:
        util = UtilizationBySlurmAccount(df,
                                         days_between_emails=args.days,
                                         violation="null",
                                         vpath=violation_logs_path,
                                         subject="")
        title = "Utilization by Slurm Account"
        s += util.generate_report_for_admins(title)

    #########################
    ## LONGEST QUEUED JOBS ##
    #########################
    if args.longest_queued:
        queued = LongestQueuedJobs(raw,
                             days_between_emails=args.days,
                             violation="null",
                             vpath=violation_logs_path,
                             subject="")
        title = "Longest queue times (1 job per user, ignoring job arrays, 4+ days)"
        s += queued.generate_report_for_admins(title)

    ###################
    ## JOBS OVERVIEW ##
    ###################
    if args.jobs_overview:
        jobs = JobsOverview(df,
                            days_between_emails=args.days,
                            violation="null",
                            vpath=violation_logs_path,
                            subject="")
        title = "Most jobs (1 second or longer -- ignoring running and pending)"
        s += jobs.generate_report_for_admins(title)

    ################
    ## MOST CORES ##
    ################
    if args.most_cores:
        most_cores = MostCores(df,
                               days_between_emails=args.days,
                               violation="null",
                               vpath=violation_logs_path,
                               subject="")
        title = "Jobs with the most CPU-cores (1 job per user)"
        s += most_cores.generate_report_for_admins(title)

    ###############
    ## MOST GPUS ##
    ###############
    if args.most_gpus:
        most_gpus = MostGPUs(df,
                             days_between_emails=args.days,
                             violation="null",
                             vpath=violation_logs_path,
                             subject="")
        title = "Jobs with the most GPUs (1 job per user, ignoring cryoem)"
        s += most_gpus.generate_report_for_admins(title)

    ########################## 
    ## SEND EMAIL TO ADMINS ##
    ########################## 
    if args.report:
        send_email(s, "halverson@princeton.edu", subject="Cluster utilization report", sender="halverson@princeton.edu")

    print(s, end="\n\n")
    print(datetime.now())
