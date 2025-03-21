# GPU-Hours at 0% Utilization

This alert sends emails to users that have consumed GPU-hours
at 0% utilization. It can also be used to generate a report of these users for
system administrators.

## Report

Here is an example report:

```
$ job_defense_shield --zero-util-gpu-hours

                           Zero Utilization GPU-Hours
-------------------------------------------------------------------------------
       User   0%-GPU-Hours  Jobs                     JobID                    
-------------------------------------------------------------------------------
1     u20461      397        16           60458831,60460188,60478799,60479839+
2     u99704      196         8           60552976,60552983,60552984,60552985+
3     u04204       62        39           60457297,60457395,60457408,60460181+
4     u39983       32        40           60419086,60419088,60419089,60419090+
5     u93550       22         6           60423037,60423668,60424743,60425344+
6     u92847       17         5           60516409,60516469,60516554,60516718+
7     u18225       17        17           60461780,60467419,60467445,60487739+
8     u99455        9         4            60475110,60496234,60496390,60554903
9     u30193        8         2                            60424873,60444734_0
10    u62696        7        13    60422906,60540828_18,60545878_8,60545878_9+
-------------------------------------------------------------------------------
   Cluster: della
Partitions: gpu, llm
     Start: Thu Sept 1, 2025 at 08:00 AM
       End: Thu Sept 8, 2025 at 12:37 PM
```

The table above shows that user `u20461` consumed 397 GPU-hours at 0% utilization.
Four of the sixteen JobID's are shown.

## Configuration File

Below is an example entry for `config.yaml`:

```yaml
zero-util-gpu-hours-1:
  cluster: della
  partitions:
    - gpu
  min_run_time: 0               # minutes
  gpu_hours_threshold_user: 24  # hours
  gpu_hours_threshold_admin: 0  # hours
  max_num_jobid: 4              # count
  admin_emails:
    - admin@institution.edu
```

The parameters are explained below:

- `cluster`: Specify the cluster name as it appears in the Slurm database. One cluster name
per alert. Use multiple `zero-util-gpu-hours` alerts for multiple clusters.

- `partitions`: Specify one or more Slurm partitions. The number of GPU-hours is summed over all partitions.

- `min_run_time`: Minimum run time in minutes for a job to be included in the calculation. For example, if `min_run_time: 30` is used then jobs that ran for less than 30 minutes are ignored. Default: 0

- `gpu_hours_threshold_user`: Only users with greater than or equal to this number of GPU-hours at 0% utilization will receive an email.

- `gpu_hours_threshold_admin`: Only users with greater than or equal to this number of GPU-hours at 0% utilization will appear in the report for administrators.

- `max_num_jobid`: (Optional) Maximum number of JobID's to show for a given user. If the number of
jobs per user is greater than this value then a "+" character is appended to the end of the list. Default: 4

- `include_running_jobs`: (Optional) If `True` then jobs in a state of `RUNNING` will be included in the calculation. The Prometheus server must be queried for each running job, which can be an expensive operation. Default: False

- `nodelist`: (Optional) Only apply this alert to jobs that ran on the specified nodes. See [example](../nodelist.md).

- `excluded_users`: (Optional) List of users to exclude from receiving emails. These users will still appear
in reports for system administrators when `--report` is used.

- `admin_emails`: (Optional) The emails sent to users will also be sent to these administator emails. This applies
when the `--email` option is used.

- `report_emails`: (Optional) Reports will be sent to these administator emails. This applies
when the `--report` option is used.

!!! info "Multi-GPU Jobs"
    For jobs that allocate multiple GPUs, only the GPU-hours for the GPUs at 0% utilization are included.

Below is second example entry for `config.yaml`:

```yaml
zero-util-gpu-hours:
  cluster: stellar
  partitions:
    - h100
  min_run_time: 30               # minutes
  gpu_hours_threshold_user: 24   # hours
  gpu_hours_threshold_admin: 12  # hours
  max_num_jobid: 3               # count
```

For the configuration above, only jobs that ran for 30 minutes or more are considered. Users will receive
an email (when `--email-users` is used) if they consumed 24 GPU-hours or more at 0% utilization. System
administrators will see users in the report (using `--email-admins`) that consumed 12 GPU-hours or more.
The JobID will be shown for up to three jobs per user. Notice that the optional settings
(`excluded_users`, `user_emails_bcc`, `report_emails`) are omitted in the YAML entry.


## How to Write Your Email File

You have these quanities available to you:

```
Dear u20461:

Over the past 7 days you have ran 16 jobs on Della that have burnt
97 GPU-hours at 0% utilization. Here are the jobid's:

    60458831,60460188,60478799,60479839+

Please investigate the reason for the GPUs not being used.
```

### Tags

These tags can be used to generate custom emails:

- `<GREETING>`: The greeting generated by `greeting-method`.
- `<CLUSTER>`: The cluster specified for the alert (i.e., `cluster`).
- `<PARTITIONS>`: The partitions listed for the alert (i.e., `partitions`).
- `<GPU-HOURS>`: Total number of GPU-hours at 0% utilization.
- `<NUM-JOBS>`: Total number of jobs with at least one idle GPU.
- `<TABLE>`: Table of job data.
- `<JOBSTATS>`: `jobstats` command for the first JobID (`$ jobstats 12345678`).

## Usage

Email users about GPU-hours at 0% utilization: 

```
$ python job_defense_shield.py --zero-util-gpu-hours --email
```

Send a report to system administrators by email:

```
$ job_defense_shield --zero-util-gpu-hours --days=7 --report
```

## Related Alerts

If you are looking to automatically jobs running a 0% GPU utilization then
see this section.

If you are looking for finding users with low but non-zero GPU utilization then
see the [low GPU utilization](low_gpu_util.md) alert.
