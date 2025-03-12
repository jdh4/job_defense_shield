# Emails

## Custom Emails to Users

Each alert requires a text file for the `email_file`:

```yaml
##################################
## ZERO CPU UTILIZATION (ALERT) ##
##################################
zero-cpu-utilization-1:
  cluster: stellar
  partitions:
    - cpu
  email_file: "zero_cpu_utilization.txt"
  admin_emails:
    - admin@institution.edu
```

The location of the `email_file` is set in `config.yaml` by:

```yaml
email-files-path: /path/to/email/
```

Here is an example `email_file`:

```
$ cat /path/to/email/zero_cpu_utilization.txt
<GREETING>

Below are your recent jobs that did not use all of the allocated nodes:

<TABLE>

The CPU utilization was found to be 0% on each of the unused nodes. You can see
this by running the "jobstats" command, for example:

<JOBSTATS>

Please investigate the reason(s) that the code is not using all of the allocated
nodes before running additional jobs.

Replying to this automated email will open a support ticket with Research
Computing.
```

There are three "tags" in the text file above: `<GREETING>`, `<TABLE>` and `<JOBSTATS>`.
Each tag will be replaced by the corresponding value in Python when creating the email. The resulting email will appear as:

```
Hello Alan (u12345),

Below are your recent jobs that did not use all of the allocated nodes:

      JobID    Cluster  Nodes  Nodes-Unused CPU-Util-Unused  Cores  Hours
    62734245    della     4          3             0%          12    2.3 
    62734246    della     6          5             0%          12    2.4 

The CPU utilization was found to be 0% on each of the unused nodes. You can see
this by running the "jobstats" command, for example:

    $ jobstats 62734245

Please investigate the reason(s) that the code is not using all of the allocated
nodes before running additional jobs.

Replying to this automated email will open a support ticket with Research
Computing.
```

Tags can be placed anywhere in your `email_file`. For example, one can include a tag in the middle of a sentence:

```
Below are your jobs on <CLUSTER> that did not use all of the allocated nodes:
```

Each alert has a finite set of tags that may be used to generate custom emails. There are
a set of example email files in the `email` directory of the [GitHub repository](https://github.com/jdh4/job_defense_shield). It is
recommended that you copy these and modify them as you see fit. It might also be a good
idea to put them under version control along with `config.yaml` and `holidays.txt`.

## Testing the Sending of Emails to Users

If `config.yaml` exists, an administrator can see the output of an alert by running it:

```
$ python job_defense_shield.py --low-gpu-efficiency
```

One adds the `--email` flag to send emails to users:

```
$ python job_defense_shield.py --low-gpu-efficiency --email

```

For testing, one can add a second flag that will only send the emails to `admin_emails` and not the users:

```
$ python job_defense_shield.py --low-gpu-efficiency --email --no-emails-to-users
```

The `--no-emails-to-users` will also prevent violation log files from being updated. This allows administrators to test and modify the email messages in safety.

There is one alert that requires one extra step, which is [Cancel 0% GPU Jobs](alert/cancel_gpu_jobs.md). In this case, one should add the following to the alert definition:

```yaml
  do_not_cancel: True
```
