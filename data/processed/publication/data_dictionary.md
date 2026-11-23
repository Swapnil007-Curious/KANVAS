# KANVAS publication data dictionary (v2)

Built from the Alibaba cluster-trace-gpu-v2020 PAI trace (CC BY 4.0; Weng et al., NSDI 2022) by
`scripts/build_full_trace.py`. Every labelled task of the trace (1,141,835 tasks with status Failed or
Terminated) goes through the same rules as the first version; 539,816 tasks survive, with a failure rate of
27.39%. The attrition is in `results/data/attrition_full_1h.csv`.

A task is joined to its first instance, and that instance's machine supplies the telemetry. The five window
features average the metric rows of **other jobs** that **ended** on the assigned machine in the **hour before**
the task started, so no feature can see the task's own execution or its own job.

No identifiers are exported: no user, job, task, worker or machine name. The submitting user was used only as a
split key and cannot be recovered from these files, so the user-grouped protocol of the paper cannot be rerun
from them alone; it needs the raw trace.

## kanvas_feature_matrix.csv.gz

Raw values. Unlike the first version, the features are **not clipped**: clipping at the 1st and 99th percentiles
and min-max scaling are fitted on each training split (`kanvas/preprocess.py`).

| column | source | meaning | units |
|---|---|---|---|
| `task_id` | (generated) | anonymous row number in build order; carries no trace identifier | integer |
| `plan_cpu` | `pai_task_table.plan_cpu` | CPU the task requested | 100 = 1 core |
| `plan_mem` | `pai_task_table.plan_mem` | memory the task requested | GB |
| `plan_gpu` | `pai_task_table.plan_gpu` | GPU the task requested | 100 = 1 GPU |
| `inst_num` | `pai_task_table.inst_num` | instances the task requested | count |
| `avg_cpu_usr` | `pai_machine_metric.machine_cpu_usr` | mean user-space CPU on the machine in the 1 h window | percent |
| `avg_gpu_util` | `pai_machine_metric.machine_gpu` | mean GPU utilisation on the machine in the window | percent (summed over GPUs) |
| `avg_cpu_kernel` | `pai_machine_metric.machine_cpu_kernel` | mean kernel-space CPU on the machine in the window | percent |
| `avg_load_1` | `pai_machine_metric.machine_load_1` | mean 1-minute load average on the machine in the window | load units |
| `cap_gpu` | `pai_machine_spec.cap_gpu` | GPUs installed on the machine | count |
| `avg_net_receive` | `pai_machine_metric.machine_net_receive` | mean network receive rate on the machine in the window | bytes/s |
| `label` | `pai_task_table.status` | 1 = Failed, 0 = Terminated (Running and Waiting dropped) | binary |
| `start_time` | `pai_instance_table.start_time` | start of the first instance, seconds from the trace origin | seconds |
| `original_sample` | (generated) | 1 for the 18,890 tasks of the first version's matrix | binary |

`start_time` reproduces the time-ordered protocol (earliest 70% train, next 15% validate, last 15% test).

## kanvas_phi_curves.csv

The canonical KAAM (full trace, random split 42, seed 0): each feature's learned contribution on 200 points
between the training 5th and 95th percentiles, centred on its mean over those points.

| column | meaning |
|---|---|
| `feature_name` | one of the ten features |
| `x_value` | feature value in raw units |
| `phi_value` | contribution to the failure log-odds |

## kanvas_benchmark_results.csv

One row per dataset (full trace or original sample), protocol, test subset and model: test ROC-AUC (mean over
test sets of the mean over seeds) with its 95% bootstrap interval, PR-AUC, Brier score, expected calibration
error (15 bins), the numbers of test sets, seeds and test tasks, and the test failure rate.

## kanvas_comparisons.csv

KAAM minus each other model per dataset, protocol and subset: the AUC difference, its paired-bootstrap 95%
interval (2,000 resamples; users resampled under the user-grouped protocol), the bootstrap p-value and its Holm
adjustment within the protocol.

## v1/

The three files and the dictionary of the first version (18,890 tasks, clipped on all rows), unchanged.
