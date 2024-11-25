# MIG Alert

The alert find jobs that could have used a MIG GPU instead of a full H100 or A100 GPU.

NVIDIA H100 and A100 GPUs can be configured as [Multi-Instance GPU](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) MIG instances. This
alert can be used to find jobs that run on H100 or A100 GPUs but could have used MIG.

The example alert provided in `alert/mig.py` 

```python
self.df = self.df[(self.df.cluster == self.cluster) &
                  (self.df.partition == self.partition) &
                  (self.df.cores == 1) &
                  (self.df.gpus == 1) &
                  (self.df.admincomment != {}) &
                  (~self.df.username.isin(self.excluded_users)) &
                  (self.df.state != "OUT_OF_MEMORY") &
                  (self.df["elapsed-hours"] >= self)].copy()
...
self.df = self.df[(self.df["GPU-Util"] <= self.gpu_util_threshold) &
                  (self.df["GPU-Util"] != 0) &
                  (self.df["GPU-Mem-Used"] < self.gpu_mem_threshold) &
                  (self.df["CPU-Mem-Used"] < self.cpu_mem_threshold)]
```

The example alert finds jobs that allocated 1 CPU-core, 1 GPU and used less than 32 GB of CPU memory, 10 GB of GPU memory, and
a GPU utilization of less than 15%. The job also must have ran for more than 60 minutes.

Here is an example configuration file (`config.yaml`) entry for the MIG alert:

```
could-use-mig:
  file: alert/mig.py
  clusters:
    - della
  partitions:
    - gpu
  min_run_time: 60        # minutes
  num_cores_threshold: 1  # count
  num_gpus_threshold: 1   # count
  gpu_util_threshold: 15  # percent
  gpu_mem_threshold: 10   # GB
  cpu_mem_threshold: 32   # GB
  excluded_users:
    - aturing
    - einstein
```

`min_run_time` is the minimum run time of the job for it to be considered. Jobs that did not run longer
than this limit will be ignored. The default is 60 minutes.

`num_cores_threshold` is the number of CPU-cores. For instance, if a job requires a large number of
of CPU-cores than it is exempt from MIG. The default value is 1 CPU-core.

`num_gpus_threshold`: The number of allocated GPUs to be considered by the alert.

`gpu_util_threshold` is the GPU utilization as available from `nvidia-smi`. Jobs with a GPU utilization
of less or equal to this value will be included. The default value is 15%.

Some institutions provide a range of MIG instances (e.g., not all H100 or A100 GPUs are converted to 7 MIG instances). In this case you will
need to modify the example to find your case. Note that you can make multiple MIG alerts to handle your situation.

## Main

```python
if args.mig:
    alerts = [alert for alert in cfg.keys() if "should-be-using-mig" in alert]
    for alert in alerts:
        mig = MultiInstanceGPU(df,
                               days_between_emails=args.days,
                               violation="should_be_using_mig",
                               vpath=args.files,
                               subject="Consider Using the MIG GPUs on Della",
                               **cfg[alert])
        if args.email and is_today_a_work_day():
            mig.send_emails_to_users()
        s += mig.generate_report_for_admins("Could Have Been MIG Jobs")
```

## Usage

Send emails to users with jobs that could have used the MIG GPUs instead of full GPUs:

```
$ job_defense_shield --could-use-mig --clusters=della --partition=gpu --days=10 --email

```
