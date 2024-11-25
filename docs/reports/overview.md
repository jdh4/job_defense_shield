# Reports for System Administrators

The general idea to generate a report is to simply add the `--report` option to an alert.
Here is an example:

```
$ job_defense_shield --zero-gpu-util --days=7 --report
```

With the appropriate entry in `config.yaml`:

```
zero-gpu-utilization:
  file: alert/zero_util_gpu_hours.py
  clusters:
    - della
  partitions:
    - gpu
  excluded_users:
    - aturing
    - einstein
```
