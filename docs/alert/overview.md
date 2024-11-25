# Overview

- The general idea is to examine all of the job data and send emails to user that are
underutilizing the systems.
- option to include actively running jobs each alert
- netid -> username
- flag to send all user emails to admins (--dry-run)
- greeting mechanism that is external (one choice is Hello username)
- may need to relate user to an external email by writing code
- cleaning mechansim
- what excluded-users is empty or not in config.yaml
- move subject out of base.py and other attributes
- consider logger with clean ouput
- specify location of file for alert
- how to keep email message external (like notes in config.py)
- what is comma in partition like cpu,physics

TODO: return email as list of strings; add new method to send the list; add tests


## Custom Greeting

Specify the full path to the greeting code:

```
greeting: "/sys/admin/job_defense_shield/greeting/greeting.py"
```


```
$ mkdir -p jds/greeting
# then write code
```

```python
def greeting(username: str) -> str:
    return f"Hello {username}"
```

```python
def greeting(username: str) -> str:
    if time < noon:
        return "Good morning,"
    elif time < 5 pm:
        return "Good afternoon:"
    return f"Hello {username}"
```

```python
def get_first_name(netid: str, formal: bool=False) -> str:
    """Get the first name of the user by calling ldapsearch."""
    cmd = f"ldapsearch -x uid={netid} displayname"
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=5, text=True, check=True)
    lines = output.stdout.split('\n')
    for line in lines:
        if line.startswith("displayname:"):
            full_name = line.replace("displayname:", "").strip()
            if ": " in full_name:
                full_name = b64decode(full_name).decode("utf-8")
            if full_name.replace(".", "").replace(",", "").replace(" ", "").replace("-", "").isalpha():
                return f"Dear {full_name.split()[0]}" if formal else f"Hi {full_name.split()[0]}"
    return "Hello"
```


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
