# Overview


The general idea is to examine all of the job data and send emails to user that are
underutilizing the systems.

Short-term work:

- flag to send all user emails to admins (--dry-run)
- data cleaning mechansim
- what is comma in partition like cpu,physics
- move subject out of base.py and other attributes
- need to strictly end jobs in window if --end is specified
- add cluster and partition to has_enough_time_passed_since_last (store cluster and partition with violation email)
- need ability to wite csv or json for —report for sys admins
- custom emails and routine to store emails in a list and then send in a separate method
- set cluster, partitions, handle start and end dates, fields, extra args, get_data, set start, set end


Long-term work:

- active finding excess memory of running jobs (need to import jobstats module)
- consider logger with clean ouput
- may need to relate user to an external email by writing code
- how to keep email message external (like notes in config.py)
- option to include actively running jobs each alert

TODO: return email as list of strings; add new method to send the list; add tests

Why is it --partition instead of --partitiions. We use the same name as sacct
which is partition.

Look at the `holidays` Python module for writing a custom workday.

## Custom Greeting

Specify the full path to the greeting code:

```
greeting: "/sys/admin/job_defense_shield/greeting/greeting.py"
```


```
$ mkdir -p jds/greeting
# then write code
```

Here is simple example that will work at any institution:

```python
class Greeting:

    def __init__(self, user):
        self.user = user

    def greeting(self):
        """Return the greeting or first line for user emails."""
        return f"Hello {self.user},\n\n"
```

Below is another example for `greeting.py` that uses the built-in `pwd` module
to interact with the password database to get the first name of the user:

```python
import pwd

class Greeting:

    def __init__(self, user):
        self.user = user

  def greeting(self):
      """Return the greeting or first line for user emails."""
      try:
          user_info = pwd.getpwnam(self.user)
      except KeyError:
          return f"Hello {self.user},\n\n"
      full_name = user_info.pw_gecos
      first_name = full_name.split()[0]
      return f"Hello {first_name} ({self.user}),\n\n"
```

Here is a more advanced example that calls `ldapsearch` to find the first name of the user:

```python
import string
import subprocess
from base64 import b64decode

class Greeting:

    def __init__(self, user):
        self.user = user

    def greeting(self):
        """Return the greeting or first line for user emails."""
        cmd = f"ldapsearch -x uid={self.user} displayname"
        output = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                shell=True,
                                timeout=5,
                                text=True,
                                check=True)
        lines = output.stdout.split('\n')
        trans_table = str.maketrans('', '', string.punctuation)
        for line in lines:
            if line.startswith("displayname:"):
                full_name = line.replace("displayname:", "").strip()
                if ": " in full_name:
                    full_name = b64decode(full_name).decode("utf-8")
                if full_name.translate(trans_table).replace(" ", "").isalpha():
                    return f"Hi {full_name.split()[0]},\n\n"
        return f"Hello {self.user},\n\n"
```

One could also write a `greeting` method based on `getent passwd`.


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
