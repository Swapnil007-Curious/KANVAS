# Alibaba PAI GPU cluster trace 2020 (cluster-trace-gpu-v2020)

This dataset is a copy of Alibaba's public GPU cluster trace from 2020, plus one derived table used by the KANVAS
notebooks. The trace covers about two months (July and August 2020) of a production machine learning cluster in
Alibaba PAI with 1,897 machines and 6,742 GPUs. It records every job, task and instance, the resources each task
asked for, per-instance resource sensors and per-machine telemetry.

Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

This upload is not made, reviewed or endorsed by Alibaba.

## Licence and attribution

The trace is released by Alibaba under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).
The full licence text is in `LICENSE.txt`, copied from the trace's own LICENSE file. The derived table in this
dataset is released under the same licence. The trace's repository README also asks that the data be used for
research or study, and encourages users to cite the paper below.

Please cite the trace paper when you use this data:

> Qizhen Weng, Wencong Xiao, Yinghao Yu, Wei Wang, Cheng Wang, Jian He, Yong Li, Liping Zhang, Wei Lin and Yu Ding.
> MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters.
> 19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22), 2022.

```
@inproceedings{weng2022mlaas,
  title     = {{MLaaS} in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous {GPU} Clusters},
  author    = {Weng, Qizhen and Xiao, Wencong and Yu, Yinghao and Wang, Wei and Wang, Cheng and He, Jian and
               Li, Yong and Zhang, Liping and Lin, Wei and Ding, Yu},
  booktitle = {19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22)},
  year      = {2022}
}
```

## Changes from the official release

- The official CSV files have no header row; the column names ship as separate `.header` files. Each CSV here has
  its header line prepended and is otherwise byte-identical to the official file. Removing the first line gives a
  file whose sha256 matches the official README; both checksums are in `SHA256SUMS.txt`.
- `kanvas_tasks_1h.parquet` is new (see below). It was built from an older copy of the trace whose
  `pai_machine_metric.csv` printed some floats with fewer digits. Parsed, those values differ from the official ones
  by at most one unit in the last place (relative difference below 4e-16, in under 1% of rows), so a rebuild from
  the files here gives the same features to about 1e-15 relative.

## Files

| File | Rows | What one row is |
|---|---:|---|
| `pai_job_table.csv` | 1,055,501 | a job: submitting user, status, submission and completion time |
| `pai_task_table.csv` | 1,261,050 | a task of a job: status, launch and end time, requested CPU, memory, GPU and instance count, GPU type |
| `pai_instance_table.csv` | 7,522,002 | an instance (one running copy of a task): status, launch and end time, machine |
| `pai_sensor_table.csv` | 3,033,232 | per-instance resource use averaged over its life: CPU, GPU utilisation, memory, GPU memory, I/O |
| `pai_group_tag_table.csv` | 1,055,032 | per-job tags: requested GPU type spec, a group tag for repeated jobs, workload name (set for about 10%) |
| `pai_machine_spec.csv` | 1,897 | a machine: GPU type, CPU cores, memory, GPU count |
| `pai_machine_metric.csv` | 2,009,423 | machine-level telemetry averaged over one instance's life on that machine |
| `kanvas_tasks_1h.parquet` | 539,816 | a labelled GPU task with the ten KANVAS features (derived) |

Units, from the official README: `plan_cpu` is CPU in percent of a core (600 = 6 cores), `plan_mem` is GB,
`plan_gpu` and `gpu_wrk_util` are GPU in percent of one GPU (50 = half a GPU), `cap_gpu` is the number of GPUs in a
machine. Times are seconds, shifted by a constant to hide the real dates; time of day and day of week are kept.
`gpu_type` MISC means older GPUs such as K40m, K80 and M60.

## The derived table `kanvas_tasks_1h.parquet`

One row per task whose status is Failed (label 1) or Terminated (label 0) and for which every feature exists.
It was built with `kanvas/fulltrace.py` from the KANVAS repository (https://github.com/Swapnil007-Curious/KANVAS,
commit 9fbe4c0) and is the table the KANVAS paper calls the full trace. Failure rate 27.39%.

| Column | Meaning |
|---|---|
| `job_name`, `task_name` | keys into the raw tables |
| `plan_cpu`, `plan_mem`, `plan_gpu`, `inst_num` | the task's requests, from `pai_task_table` |
| `cap_gpu` | GPU count of the machine that ran the task's first instance, from `pai_machine_spec` |
| `avg_cpu_usr`, `avg_gpu_util`, `avg_cpu_kernel`, `avg_load_1`, `avg_net_receive` | means of `machine_cpu_usr`, `machine_gpu`, `machine_cpu_kernel`, `machine_load_1` and `machine_net_receive` over `pai_machine_metric` rows of other jobs that ended on that machine in the hour before the task's first instance started |
| `label` | 1 = Failed, 0 = Terminated |
| `start_time` | launch time of the task's first instance (trace seconds) |
| `user_rank` | the submitting user as an integer: 1 is the user with the most tasks in this table, 2 the next, and so on |
| `machine` | machine of the first instance |
| `n_metric_rows` | number of telemetry rows averaged |

The telemetry features look only at other jobs and only at the hour before the task started, so they carry no
information from the task's own run. Tasks with no telemetry row in that hour, CPU-only tasks and tasks with a
missing value were dropped.

A warning for anyone modelling this table: one user (`user_rank` 1) submits 31% of the tasks and 55% of the
failures, and random task-level splits overstate how well a model does on users it has not seen. Evaluate with
users held out or with a time-ordered split.

## Notebooks that use this dataset

A five-part series built from the KANVAS project: a trace EDA, an analysis of idle GPUs, a LightGBM failure
baseline with SHAP, a from-scratch Kolmogorov-Arnold additive model, and a comparison of its spline curves with
SHAP. Code: https://github.com/Swapnil007-Curious/KANVAS
