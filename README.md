[![Tests](https://github.com/jdh4/job_defense_shield/actions/workflows/tests.yml/badge.svg)](https://github.com/jdh4/job_defense_shield/actions?workflow=Tests)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

# Job Defense Shield

The software in this repo creates a report of problem users and problem jobs on the large Research Computing clusters. The software identifies the following:

+ actively running jobs where a GPU has zero utilization  
+ the heaviest users with low CPU or GPU utilization  
+ jobs that use the datascience nodes but do not need them  
+ jobs that could have been run on MIG GPUs instead of full A100 GPUs  
+ multinode CPU jobs where one or more nodes have zero utilization  
+ users with excessive run time limits  
+ jobs with CPU or GPU fragmentation (e.g., 1 GPU per node over 4 nodes)  
+ jobs with the most CPU-cores and jobs with the most GPUs  
+ pending jobs with the longest queue times  
+ jobs that request more than the default memory but do not use it  

The script does not identify:
+ abuses of file storage or I/O  
+ problems with jobs or users on Adroit

## How to run

```
$ ./job_defense_shield.py --email \
                          --days=3 \
                          --zero-gpu-utilization \
                          --files /tigress/jdh4/utilities/job_defense_shield/violations
                          
$ ./job_defense_shield.py --email \
                          --watch \
                          --zero-gpu-utilization \
                          --low-xpu-efficiencies \ 
                          --datascience \
                          --gpu-fragmentation                          
```

The data science nodes are checked once per day:

```
$ ./job_defense_shield.py --email \
                          --datascience \
                          --days=7 \
                          --files /tigress/jdh4/utilities/job_defense_shield/violations                          
```

## Which users are ignoring the automated emails?

```
$ ./job_defense_shield.py --check --zero-gpu-utilization --days=30
```

### Notes for developers

- As Slurm partitions are added and removed the script should be updated  
- For jdh4, the git repo is /tigress/jdh4/utilities/job_defense_shield

To run the unit tests:

```
$ module load anaconda3/2023.3
$ pytest  --cov=. --capture=tee-sys tests
```

### How to use

Run the commands below on a login node (e.g., tigergpu) to execute the script:

```bash
$ git clone https://github.com/jdh4/job_defense_shield.git
$ module load anaconda3/2022.10
$ cd job_defense_shield
$ ./job_defense_shield.py -h
```

###  Gotchas

1. Some Traverse jobs are CPU only
2. Pandas:

```
>>> import pandas as pd
>>> df = pd.DataFrame({"A":[1, 2, 3], "B":[4, 5, 6]})
>>> df = df[df.A > 10]
>>> df.empty
True
>>> df["C"] = df.apply(lambda row: row["A"] * row["B"], axis="columns")
# ValueError: Wrong number of items passed 2, placement implies 1

df["C"] = df.A.apply(round)  # this is okay
>>>
```

## Installation

The requirements are:

- Python 3.7 or above  
- Pandas  
- jobstats (if looking to send emails about actively running jobs)  

You can run the script on `tigergpu` using the `jds-bin` in `/home/jdh4/bin`. The Conda environment was create in this way:

```
[jdh4@tigergpu ~]$ cat .condarc
envs_dirs:
- /home/jdh4/bin
```

```
$ module load anaconda3/2022.10
$ conda create --name jds-env numpy pandas blessed requests pyyaml -c conda-forge -y
```

The above leads to the shebang line as:

```
#!/home/jdh4/bin/jds-bin/python -u -B
```

If you do not need to inspect actively running jobs then you do not need `requests` or `blessed`.

## cron

```
[jdh4@tigergpu ~]$ crontab -l
30 8 * * 1,4 /home/jdh4/bin/jds-env/bin/python -u -B /tigress/jdh4/utilities/job_defense_shield/job_defense_shield.py --email > /dev/null 2>&1
```

## convert CSV to JSON

```
import glob
import shutil

files = glob.glob("*.csv")

for f in files:
  user = f.split(".")[0]
  print(user, f)
  shutil.copy(f, f"{user}.email.csv")

if 0:
  for f in files:
    with open(f) as j:
      data = j.readlines()
    print(data[0])
```
