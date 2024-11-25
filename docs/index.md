# Overview

Job Defense Shield is a Python code for sending automated email alerts to users and for creating reports for system administrators. As discussed above, summary statistics for each completed job are stored in a compressed format in the `AdminComment` field in the Slurm database. The software described here works by calling the Slurm `sacct` command while requesting several fields including `AdminComment`. The `sacct` output is stored in a `pandas` dataframe for processing.

Automated email alerts to users are available for these cases:

- CPU or GPU jobs with 0% utilization (see email below)
- Heavy users with low mean CPU or GPU efficiency
- Jobs that allocate excess CPU memory (see email below)
- Jobs that allocate too many CPU-cores per GPU
- Jobs that allocate too much CPU memory per GPU
- Serial jobs that allocate multiple CPU-cores
- Users that routinely run with excessive time limits
- Jobs that could have used a smaller number of nodes
- Jobs that could have used less powerful GPUs
- Jobs thar ran on specialized nodes but did not need to

All of the instances in the list above can be formulated as a report
for system administrators. The most popular reports for system
administrators are:

- A list of users (and their jobs) with the most GPU-hours at 0% utilization
- A list of the heaviest users with low CPU/GPU utilization
- A list of users that are over-allocating the most CPU memory
- A list of users that are over-allocating the most time

The Python code is written using object-oriented programming techniques which makes it easy to create new alerts and reports.

## Example Emails

Below is an example email for the automatic cancellation of a GPU job with 0% utilization:

```
Hi Alan,

The jobs below have been cancelled because they ran for nearly 2 hours at 0% GPU
utilization:

     JobID    Cluster  Partition    State    GPUs-Allocated GPU-Util  Hours
    60131148   della      llm     CANCELLED         4          0%      2.0  
    60131741   della      llm     CANCELLED         4          0%      1.9  

See our GPU Computing webpage for three common reasons for encountering zero GPU
utilization:

    https://<your-institution>.edu/knowledge-base/gpu-computing

Replying to this automated email will open a support ticket with Research
Computing.
```

Below is an example email to a user that is requesting too much CPU memory:

```
Hi Alan,

Below are your jobs that ran on BioCluster in the past 7 days:

     JobID   Memory-Used  Memory-Allocated  Percent-Used  Cores  Hours
    5761066      2 GB          100 GB            2%         1      48
    5761091      4 GB          100 GB            4%         1      48
    5761092      3 GB          100 GB            3%         1      48

It appears that you are requesting too much CPU memory for your jobs since
you are only using on average 3% of the allocated memory. For help on
allocating CPU memory with Slurm, please see:

    https://<your-institution>.edu/knowledge-base/memory

Replying to this automated email will open a support ticket with Research
Computing. 
```

## Usage

The software has a `check` mode that shows on which days a given user received an alert of a given type. Users that appear to be ignoring the email alerts can be contacted directly. Emails to users are most effective when sent sparingly. For this reason, there is a command-line parameter to specify the amount of time that must pass before the user can receive another email of the same nature.

The example below shows how the script is called to notify users in the top N by usage with low CPU or GPU efficiencies over the last week:

```
$ job_defense_shield --low-xpu-efficiencies --days=7 --email
```

The default thresholds are 60% and 15% for CPU and GPU utilization, respectively, and N=15.

There is a corresponding entry in the configuration file:

```
low-xpu-efficiencies:
  cluster:
    - della
  partitions:
    - all
  threshold: 10
```
