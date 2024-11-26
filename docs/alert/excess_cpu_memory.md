# Excess CPU Memory

This alert sends emails to users that are allocating excess CPU memory.
For example, the job used 1 GB but the user allocated 100 GB.

## Configuration File


```
excess-cpu-memory:
  clusters:
    - della
  partition:
    - cpu
  cores_per_node: 28
  tb_hours_per_day: 10
  ratio_threshold: 0.35
  mean_ratio_threshold: 0.35
  median_ratio_threshold: 0.35
  num_top_users: 10
  combine_partitions: False
  excluded_users:
    - aturing
    - einstein
    - vonholdt
  admin_emails:
    - alerts-jobs-aaaalegbihhpknikkw2fkdx6gi@princetonrc.slack.com
    - halverson@princeton.edu
```

Each configuration parameter is explained below:

- `cores_per_node`: Number of CPU-cores per node.

- `tb_hours_per_day`: The threshold for TB-hours per day. This value
is multiplied by `--days` to determine the threshold of TB-hours for
the user to receive an email message.

- `ratio_threshold`: This quantity is the sum of CPU memory used divded
by the total memory allocated for all jobs of a given user in the specified
time window. This quantity varies between 0 and 1.

- `median_ratio_threshold`: This is median value of memory used divided by
memory allocated for the individual jobs of the user.

- `mean_ratio_threshold`: The mean value of the memory used divided by the
memory allocated per job for a given user. This quantity varies between 0
and 1.

## Usage

