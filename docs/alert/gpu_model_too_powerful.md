# GPU Model Too Powerful

This alert is used to identify jobs that could have ran on less powerful GPUs.
For example, it can find jobs that ran on NVIDIA H100 GPUs but could have
used the less powerful L40S GPUs or [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/).
The GPU utilization, CPU/GPU memory usage, and number of allocated CPU-cores
is taken into account when identifying jobs.


## Report

```bash
$ python job_defense_shield.py --gpu-model-too-powerful

          GPU Model Too Powerful         
-----------------------------------------
 User   GPU-Hours  Jobs  JobID    email90
-----------------------------------------
yw6760    1.1       1   61122477    2
```

## Email Message

```
Hello Alan (u12345),

Below are jobs that ran on an A100 GPU on Della in the past 10 days:

   JobID    User  GPU-Util GPU-Mem-Used CPU-Mem-Used  Hours
  60984405 aturing   9%        2 GB         3 GB      3.4  
  60984542 aturing   8%        2 GB         3 GB      3.0  
  60989559 aturing   8%        2 GB         3 GB      2.8  

The jobs above have a low GPU utilization and they use less than 10 GB of GPU
memory and less than 32 GB of CPU memory. Such jobs could be run on the MIG
GPUs. A MIG GPU has 1/7th the performance and memory of an A100. To run on a
MIG GPU, add the "partition" directive to your Slurm script:

  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=1
  #SBATCH --gres=gpu:1
  #SBATCH --partition=mig

For interactive sessions use, for example:

  $ salloc --nodes=1 --ntasks=1 --time=1:00:00 --gres=gpu:1 --partition=mig

If you are using Jupyter OnDemand then set the "Node type" to "mig" when
creating the session.

By running jobs on the MIG GPUs you will experience shorter queue times and
you will help keep A100 GPUs free for jobs that need them. For more info:

  https://researchcomputing.princeton.edu/systems/della#gpus

As an alternative to MIG, you may consider trying to improve the GPU
utilization of your code. A good target value is greater than 50%. Consider
writing to the mailing list of the software that you are using or attend
an in-person Research Computing help session:

  https://researchcomputing.princeton.edu/support/help-sessions

For general information about GPU computing at Princeton:

  https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing

Replying to this automated email will open a support ticket with Research
Computing.
```

The example alert provided in `alert/gpu_model_too_powerful.py` 

## Configuration File

Here is an example configuration file (`config.yaml`) entry for the MIG alert:

```yaml
gpu-model-too-powerful:
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
  email_file: "alert/mig.py"
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
$ python job_defense_shield.py --gpu-model-too-powerful --clusters=della --partition=gpu --days=7 --email
```

Exactly the same as above:

```
$ python job_defense_shield.py --gpu-model-too-powerful -M della -r gpu --email
```
