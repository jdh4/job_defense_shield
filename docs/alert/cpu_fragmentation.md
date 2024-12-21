# Multinode CPU Fragmentation

Consider a cluster with 64 CPU-cores per node. A user can run a job
that requires 128 CPU-cores by (1) allocating 64 CPU-cores on 2 nodes
or (2) allocating 4 CPU-cores on 32 nodes. The former is in general
strongly preferred. This alert catches jobs doing the latter,
i.e., multinode jobs that allocate less than the
number of available CPU-cores per node (e.g., 4 CPU-cores on 32 nodes).
The memory usage of each job is taken into account when looking
for fragmentation.

Jobs with 0% CPU utilization on a node are ignored since those will captured
by a different alert.

## Report for System Administrators

```
$ python job_defense_shield --multinode-cpu-fragmentation
```

## Email

Below is an example:

## Configuration File

Below is the minimal settings for this alert:


## Usage

Use it
