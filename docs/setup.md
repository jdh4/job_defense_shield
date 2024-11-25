# Installation

We assume that the [Jobstats platform](https://github.com/PrincetonUniversity/jobstats) is available and working. While there are some alerts
that do not require Jobstats (e.g., time, utilization-overview), to take full advantage of the software it should be present.

The installation requirements for Job Defense Shield are Python 3.6+ and version 1.2+ of the Python `pandas` package. The `jobstats` command is also required if one wants to examine actively running jobs such as when looking for jobs with zero GPU utilization. The Python code, example alerts and emails, and instructions are available in the <a href="https://github.com/PrincetonUniversity/job_defense_shield" target="_blank">GitHub repository</a>.

```
$ conda create --name jds-env pandas pyarrow pytest blessed requests pyyaml -c conda-forge -y
```

$ apt-get install python3-pandas python3-requests python3-yaml python3-blessed

You only need `requests` and `blessed` and `jobstats.py` to exmaine actively
running jobs.

## Testing the Installation

To test the software, run this simple command which does not send any emails:

```
$ job_defense_shield --utilization-overview --days=7
```

The command above will show an overview of the number of CPU-hours and GPU-hours
across all clusters and partitions in the Slurm database.

Here is an example:

```
$ job_defense_shield.py --utilization-overview --days=7


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

## Troubleshooting the Installation

Finding the right `python`

```
$ python -c "import pandas"
$ python -c "import pyyaml"
```

Finding the configuration file. The code will look for the configuration file in multiple
locations. You can always specify the full path with:

```
--cfg=<path/to>/job_defense_shield/config.yaml
```
 
