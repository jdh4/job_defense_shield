# Installation

We assume that the [Jobstats platform](https://github.com/PrincetonUniversity/jobstats) is available and working.

!!! info "Cancelling Jobs at 0% GPU Utilization"
    To automatically cancel actively running jobs, the software must be ran as a user with sufficient privileges to call `scancel`. This may inform your decision of where to install the software. All of the other alerts can be ran as a regular user. It may make sense to have one installation on a secure machine which cancels GPU jobs and a second installation on a login node which is ran as a regular user for the other alerts.

The installation requirements for Job Defense Shield are `pandas` and `pyyaml`. The `requests` module is needed only if one wants to address the underutilization of actively running jobs. In this case, the Prometheus server must be queried.

Here are some ways the requirements can be installed:

=== "Ubuntu"

    ```
    $ apt-get install python3-pandas python3-yaml python3-requests
    ```

=== "conda"

    ```bash
    $ conda create --name jds-env pandas pyarrow pyyaml requests -c conda-forge
    $ conda activate jds-env
    ```

=== "pip"

    ```bash
    $ python3 -m venv jds-env
    $ source jds-env/bin/activate
    (jds-env) $ pip3 install pandas pyarrow pyyaml requests
    ``` 

Next, pull down the repository:

```
$ git clone https://github.com/jdh4/job_defense_shield
$ cd job_defense_shield
```

## Testing the Installation

The simplest test is to run the help menu:

```
$ python job_defense_shield.py --help
```

Try running a simple informational alert. First, make a trivial configuration file called `config.yaml` in the same directory as `job_defense_shield.py`:

```
$ cat config.yaml
```
```yaml
%YAML 1.1
---
#####################
## GLOBAL SETTINGS ##
#####################
jobstats-module-path: /tmp
violation-logs-path: /tmp
email-files-path: /tmp
email-domain-name: "@institution.edu"
sender: support@institution.edu
reply-to: support@institution.edu
report-emails:
  - admin@institution.edu
```

!!! tip
    If the path that you use for `violation-logs-path` does not exist then the software will try to make it. If `/tmp` does not exist then use something else like `/scratch`.

Be sure to replace `email-domain-name`, `sender`, `reply-to` and `report-emails` with your values.

To test the software, run this command (which does not send any emails):

```
$ python job_defense_shield.py --utilization-overview
```

The command above will show an overview of the number of CPU-hours and GPU-hours
across all clusters and partitions in the Slurm database over the past 7 days.

Here is an example output:

```
$ python job_defense_shield.py --utilization-overview


           Utilization Overview           
------------------------------------------
cluster   users   cpu-hours    gpu-hours  
------------------------------------------
   della  491   1664010 (16%) 107435 (76%)
 stellar  148   6290093 (61%)    3193 (2%)
   tiger   43   2276967 (22%)       0 (0%)
traverse    1     126183 (1%)  31546 (22%)


            Utilization Overview by Partition            
---------------------------------------------------------
cluster    partition    users   cpu-hours    gpu-hours  
---------------------------------------------------------
   della           cpu  296    963959 (58%)       0 (0%)
   della         pli-c   27    217984 (13%)  31254 (29%)
   della    gpu-shared  116     158050 (9%)  42930 (40%)
   della       datasci   45      95059 (6%)       0 (0%)
   della       physics   16      66573 (4%)       0 (0%)
   della           gpu   35      63406 (4%)  11736 (11%)
   della        cryoem   20      24510 (1%)    4524 (4%)
   della         donia    6      19055 (1%)       0 (0%)
   della        gpu-ee    3      18779 (1%)     255 (0%)
   della           pli   23      16976 (1%)    7806 (7%)
   della       gputest  115      11698 (1%)    1485 (1%)
   della           mig   47       7445 (0%)    7445 (7%)
   della         malik    1        515 (0%)       0 (0%)
   della     gpu-wentz    1          3 (0%)       1 (0%)
 stellar            pu   48   2998473 (48%)       0 (0%)
 stellar          pppl   38   1747524 (28%)       0 (0%)
 stellar         cimes   23   1423071 (23%)       0 (0%)
 stellar        serial   44      60886 (1%)       0 (0%)
 stellar           all   55      45176 (1%)       0 (0%)
 stellar           gpu   22      14691 (0%)  3193 (100%)
 stellar        bigmem    3        270 (0%)       0 (0%)
   tiger           cpu   31   1792892 (79%)            0
   tiger           ext    5    476715 (21%)            0
   tiger        serial   15       7360 (0%)            0
traverse           all    1   126183 (100%) 31546 (100%)
```

You can go further back in time by using the `--days` option:

```
$ python job_defense_shield.py --utilization-overview --days=14
```

!!! info
    Using a large value for the `--days` option can cause the Slurm database to fail to produce the data. The default is 7 days.

One can only include data from specific clusters or partitions using the `-M` and `-r` options from `sacct`:

```
$ python job_defense_shield.py --utilization-overview -M della
```

The `-M` and `-r` options are very useful for alerts that only apply to a particular cluster or particular partitions.

## Email Test

By having your email address in `report-emails`, the `--report` flag can be used to send the output to administrators by email:

```
$ python job_defense_shield.py --utilization-overview --report
```

## Troubleshooting the Installation

Make sure you are using the right `python`. All three commands below should run successfully:

```
$ python -c "import sys; print(sys.version)"
$ python -c "import pandas; print(pandas.__version__)"
$ python -c "import pyyaml; print(pyyaml.__version__)"
```

If the configuration file is not found then try specifying the full path:

```
$ python job_defense_shield.py --config-file=/path/to/config.yaml --utilization-overview --report
```
 
## Creating a Configuration File for Production

See the [next section](configuration.md) to learn how to write a proper configuration file.
