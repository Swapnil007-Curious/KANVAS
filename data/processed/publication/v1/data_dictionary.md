# KANVAS publication data dictionary

Built from the Alibaba Cluster-Trace-GPU-2020 PAI trace (`pai_task_table`, `pai_instance_table`,
`pai_machine_metric`, `pai_machine_spec`). 18,890 tasks survive the full pipeline;
the failure rate is 27.50%.

A task is joined to its first instance, and that instance's machine supplies the telemetry.
Machine features average **other jobs'** workers that ended on the assigned machine in the
**1 hour before** the task started, so no feature can see the task's own execution.

No personal data is involved. Alibaba's internal `job_name` / `task_name` identifiers are
deliberately not exported; `task_id` is an anonymous row number.

## kanvas_feature_matrix.csv

| column | source | meaning | units |
|---|---|---|---|
| `task_id` | (generated) | anonymous row number; carries no trace identifier | integer |
| `plan_cpu` | `pai_task_table.plan_cpu` | CPU the task requested | 100 = 1 core |
| `plan_mem` | `pai_task_table.plan_mem` | memory the task requested | GB |
| `plan_gpu` | `pai_task_table.plan_gpu` | GPU the task requested | 100 = 1 GPU |
| `inst_num` | `pai_task_table.inst_num` | instances the task requested | count |
| `avg_cpu_usr` | `pai_machine_metric.machine_cpu_usr` | mean user-space CPU on the assigned machine in the 1h lookback | percent |
| `avg_gpu_util` | `pai_machine_metric.machine_gpu` | mean GPU utilization on the assigned machine in the 1h lookback | percent (sums across GPUs, so >100 is normal) |
| `avg_cpu_kernel` | `pai_machine_metric.machine_cpu_kernel` | mean kernel-space CPU on the assigned machine in the 1h lookback | percent |
| `avg_load_1` | `pai_machine_metric.machine_load_1` | mean 1-minute load average on the assigned machine in the 1h lookback | load units |
| `cap_gpu` | `pai_machine_spec.cap_gpu` | GPUs physically installed on the assigned machine | count |
| `avg_net_receive` | `pai_machine_metric.machine_net_receive` | mean network receive rate on the assigned machine in the 1h lookback | bytes/s |
| `label` | `pai_task_table.status` | 1 = Failed, 0 = Terminated; Running/Waiting dropped as unresolved | binary |

All ten features are clipped to their 1st/99th percentile, and rows with any NaN are dropped,
both inside `build_kanvas_dataset`.

## kanvas_phi_curves.csv

The canonical KAAM model's learned per-feature curves, long format and directly plottable.

| column | meaning |
|---|---|
| `feature_name` | one of the ten features above |
| `x_value` | feature value in raw units, 200 points spanning that feature's 5th-95th training percentile |
| `phi_value` | that feature's additive contribution to the failure log-odds, centered on the curve's own mean |

KAAM's prediction is `sigmoid(bias + sum_i phi_i(x_i))`, so these curves *are* the model's
explanation exactly, not an approximation of it.

## kanvas_benchmark_results.csv

Table 1 in flat form: one row per model (KAAM, logistic regression, capacity-matched MLP) with
accuracy, AUC and its bootstrap interval, precision, recall, F1, Brier score, learnable parameter
count and training wall-clock seconds; plus one row per model pair carrying the paired-bootstrap
95% interval on the AUC difference (2,000 resamples of the test rows).

Canonical run: split `random_state=42`, initialization seed
0, both fixed before any metric was computed.
