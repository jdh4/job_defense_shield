# Job Defense Shield

The software in this repo creates a report of problem users and problem jobs on the large Research Computing clusters. The software identifies the following:

+ the heaviest users with low CPU or GPU utilization  
+ multinode CPU jobs where one or more nodes have zero utilization  
+ multi-GPU jobs where one or more GPUs have zero utilization  
+ users that have been over-allocating CPU or GPU time  
+ jobs with CPU or GPU fragmentation (e.g., 1 GPU per node over 4 nodes)  
+ jobs with the most CPU-cores and jobs with the most GPUs  
+ jobs with the longest queue times  
+ jobs that request more than the default memory but do not use it  
+ users (G1 and G2) with all failed jobs on the last day they submitted  

The script does not identify:
+ abuses of file storage or I/O  
+ problems with jobs or users on Adroit

## How to run

```
$ /home/jdh4/bin/jds-bin/python job_defense_shield.py --zero-gpu-utilization --files /tigress/jdh4/utilities/job_defense_shield/violations --email
$ /home/jdh4/bin/jds-bin/python job_defense_shield.py --zero-gpu-utilization --low-gpu-efficiencies --datascience --gpu-fragmentation --email
$ /home/jdh4/bin/jds-bin/python job_defense_shield.py --all --files /tigress/jdh4/utilities/job_defense_shield/violations --email
```

The data science nodes are checked once per day:

```
$ /home/jdh4/bin/jds-bin/python job_defense_shield.py --files /tigress/jdh4/utilities/job_defense_shield/violations --datascience --email
```

## Which users are ignoring the automated emails?

```
$ /home/jdh4/bin/jds-bin/python job_defense_shield.py --zero-gpu-utilization --check
$ /home/jdh4/bin/jds-bin/python job_defense_shield.py --check-all
```

## What to do after writing to a user?

The event must be logged so that when `--check` it does not suggest writing again to the user. The next line will write to the dataframe indicating that a new email is not needed:

```
$ /home/jdh4/bin/jds-bin/python job_defense_shield.py --resolved ab1234 --violation zero-gpu
```

### Notes for developers

- As Slurm partitions are added and removed the script should be updated  
- dossier.py comes from [here](https://github.com/jdh4/tigergpu_visualization)
- For jdh4, the git repo is /tigress/jdh4/utilities/job_defense_shield

### How to use

Run the commands below on a login node (e.g., tigergpu) to execute the script:

```bash
$ wget https://raw.githubusercontent.com/jdh4/job_defense_shield/main/job_defense_shield.py
$ wget https://raw.githubusercontent.com/jdh4/tigergpu_visualization/master/dossier.py
$ module load anaconda3/2021.11
$ python job_defense_shield.py
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

You can run the script on `tigergpu` using the `jds-bin` in `/home/jdh4/bin`. The Conda environment was create in this way:

```
[jdh4@tigergpu ~]$ cat .condarc
envs_dirs:
- /home/jdh4/bin
```

```
$ module load anaconda3/2021.5
$ conda create --name jds-env numpy pandas blessed requests -c conda-forge -y
```

The above leads to the shebang line as:

```
#!/home/jdh4/bin/jds-bin/python -u -B
```

## cron

```
[jdh4@tigergpu ~]$ crontab -l
30 8 * * 1,4 /home/jdh4/bin/jds-env/bin/python -u -B /tigress/jdh4/utilities/job_defense_shield/job_defense_shield.py --email > /dev/null 2>&1
```
