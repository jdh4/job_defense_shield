# Overview

The general idea is to examine all of the job data and send emails to user that are
underutilizing the systems.

Why is it --partition instead of --partitiions. We use the same name as sacct
which is partition.

Look at the `holidays` Python module for writing a custom workday.

## Email Greeting

## Grace Period

Email alerts are most effective when sent sparingly. For this reason, the software is designed to
ensure that a certain amount of time has passed before the user can receive the notice again. Use
the `--days` option to set the minimum number of days between receiving an email.

## Violation History

When a user receives an email report, their file is updated with the jobs.

## `email` mode

Use the `--email` option to send emails to the users for a given alert.

## `report` mode

Use the `--report` option to send a report for the given alert to system administrators.

## `check` mode

Use the `--check` to see which users have recevied alerts and when they received them.

## Performance Tip

If you alerts only needs data for specific clusters and partitions then:

```
$ job_defense_shield --zero-util-gpu-hours --days=7 --clusters=della --partition=gpu,llm --email
```

The `--clusters` and `--partition` options are passed through to `sacct`. This means
that less data is queried saving time. If you do not specify these options then
everything in the Slurm database is retrieved.

You can run multiple alerts at once:

```
$ job_defense_shield --zero-util-gpu-hours --excess-cpu-memory --could-use-mig --days=7 --email
```

Here is an entire report:

```
#!/bin/bash
PY="/home/jdh4/bin/jds-env/bin"
BASE="/tigress/jdh4/utilities/job_defense_shield"
CFG="${BASE}/config.yaml"
${PY}/python -uB ${BASE}/job_defense_shield.py --days=7 \
                                               --config-file=${CFG} \
                                               --report \
                                               --zero-util-gpu-hours \
                                               --mig \
                                               --low-xpu-efficiency \
                                               --zero-cpu-utilization \
                                               --cpu-fragmentation \
                                               --gpu-fragmentation \
                                               --excessive-time \
                                               --excess-cpu-memory \
                                               --serial-using-multiple \
                                               --longest-queued \
                                               --jobs-overview \
                                               --utilization-overview \
                                               --most-gpus \
                                               --most-cores > ${BASE}/log/report.log 2>&1
```

## Configuration File

Show a file with multiple alerts and global settings are the top. Explain that the --<alert> flag determines which alerts described in the configuration file run.

## Next Steps

At this point you are ready to setup the first alert that actually emails users. The simplest and most broadly applicable alert is  excess run time.

They key takeaways are to create your config file. You can have multiple entries for the same alert type.

## Configuring Crontab

Setup crontab to automatically call the code.
