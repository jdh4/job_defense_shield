# Automatically Cancel GPU Jobs at 0% Utilization

This alert is different than all of the others. It requires Jobstats and must
be ran as a privileged user to call `scancel`. It also applies to actively
running jobs.

This is one of the most powerful features of the Job Defense Shield as it
can save your institution expensive resources.

Below is an example email:

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

## Configuration File

```
zero-gpu-utilization-della-gpu:
  clusters:
    - della
  partition:
    - gpu
  first_warning_minutes: 60
  second_warning_minutes: 105
  cancel_minutes: 120
  sampling_period_minutes: 15
  min_previous_warnings: 1
  max_interactive_hours: 8
  jobids_file: "/var/spool/slurm/job_defense_shield/jobids.txt"
  excluded_users:
    - aturing
    - einstein
  admin_emails:
    - jdh4@princeton.edu
```

## Usage

```
PY=/var/spool/slurm/cancel_zero_gpu_jobs/envs/jds-env/bin
JDS=/var/spool/slurm/job_defense_shield
MYLOG=/var/spool/slurm/cancel_zero_gpu_jobs/log
VIOLATION=/var/spool/slurm/job_defense_shield/violations
MAILTO=jdh4@princeton.edu

*/15 * * * * ${PY}/python -uB ${JDS}/job_defense_shield.py --zero-gpu-utilization --days=1 --email --files=${VIOLATION} -M della -r gpu > ${MYLOG}/zero_gpu_utilization.log 2>&1
```

