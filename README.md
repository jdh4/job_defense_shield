# Job Defense Shield

The software in this repo creates a report of problem users and problem jobs on the large Research Computing clusters.

It looks for the following:
+ users that have been over-allocating CPU or GPU time
+ CPU and GPU job fragmentation (e.g., 1 GPU per node over N nodes)
+ the largest CPU and GPU jobs
+ the jobs that have been queued for the longest
+ cases where a user encountered all failed jobs on the last day they submitted

The script does not protect against:
+ abuses of file storage or I/O


`dossier.py` is used to obtain the position and department of a user. It is maintained [here](https://github.com/jdh4/tigergpu_visualization).


## Be Aware

- As partitions are added and removed the script should be updated.  
