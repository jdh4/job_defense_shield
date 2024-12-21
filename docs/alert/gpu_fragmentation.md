# Multinode GPU Fragmentation

Consider a cluster with 4 GPUs per node. A user can run a job
with 8 GPUs by either (1) allocating 4 GPUs on 2 nodes
or (2) allocating 1 GPU on 8 nodes. The former is in general
strongly preferred. This alert catches jobs doing the latter,
i.e., multinode jobs that allocate less than the
number of available GPUs per node (e.g., 1 GPU on 8 nodes).

Jobs with a GPU at 0% utilization are ignored since they will
be captured by the `--zero-gpu-utilization` alert.


## Report

Here is example output of running the alert:

```
$ python job_defense_shield.py --multinode-gpu-fragmentation
table
```

## Email

```
Hello Alan (u12345),

Below are jobs that ran on Della in the past 7 days that used 1 GPU per node
over multiple nodes:

     JobID     User   GPUs  Nodes GPUs-per-Node  Hours State GPU-eff
    60969293 aturing   4     2          2          24    TO     0%  

The GPU nodes on Della have either 8 GPUs per node. For future jobs,
please try to use as few nodes as possible by allocating more GPUs per node. This
is done by modifying the --gres Slurm directive as explained here:

    https://researchcomputing.princeton.edu/support/knowledge-base/slurm#gpus

In any of your jobs have a low GPU utilization then please consider using only
a single GPU per job to improve efficiency.

For more information about the Della GPU nodes:

    https://researchcomputing.princeton.edu/systems/della#gpus

Consider attending an in-person Research Computing help session for assistance:

    https://researchcomputing.princeton.edu/support/help-sessions

Replying to this automated email will open a support ticket with Research
Computing.
```


## Configuration File

```yaml
multinode-gpu-fragmentation-2:
  cluster: della
  partitions:
    - gpu
  gpus_per_node: 2  # count
  min_run_time: 61  # minutes
  email_file: "email/multinode_gpu_fragmentation_2.txt"
  admin_emails:
    - alerts-jobs-aaaalegbihhpknikkw2fkdx6gi@princetonrc.slack.com
    - halverson@princeton.edu
```

## Usage

```
$ python job_defense_shield.py --multinode-gpu-fragmentation
```

## cron


