# Job Defense Shield

The software in this repo creates a report of problem users and problem jobs on the large Research Computing clusters. The software finds the following:

+ finds the heaviest users with low CPU or GPU utilization  
+ finds multinode CPU jobs where one or more nodes have zero utilization  
+ finds multi-GPU jobs where one or more GPUs have zero utilization  
+ users that have been over-allocating CPU or GPU time  
+ CPU and GPU job fragmentation (e.g., 1 GPU per node over 4 nodes)  
+ jobs with the most CPU-cores and jobs with the most GPUs  
+ jobs with the longest queue times  
+ jobs that request more than the default memory but do not use it  
+ cases where a user encountered all failed jobs on the last day they submitted  

The script does not protect against:
+ abuses of file storage or I/O  

### Notes for developers

- As partitions are added and removed the script should be updated.  
