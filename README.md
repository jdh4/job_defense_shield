# Job Defense Shield

The software in this repo creates a report of problem users and problem jobs on the large Research Computing clusters.

It looks for the following:
+ users that have been over-allocating CPU or GPU time
+ CPU and GPU job fragmentation (e.g., 1 GPU per node over N nodes)
+ the largest CPU and GPU jobs
+ the jobs that have been queued for the longest

The script does not protect against:
+ abuses of file storage or I/O
