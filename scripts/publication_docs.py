"""Text of the v2 data dictionary and the draft Hugging Face dataset card, filled from the result files.

Called by scripts/export_publication.py. The card is a local draft; nothing is uploaded.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "processed" / "publication"

DICTIONARY = """# KANVAS publication data dictionary (v2)

Built from the Alibaba cluster-trace-gpu-v2020 PAI trace (CC BY 4.0; Weng et al., NSDI 2022) by
`scripts/build_full_trace.py`. Every labelled task of the trace ({labelled:,} tasks with status Failed or
Terminated) goes through the same rules as the first version; {tasks:,} tasks survive, with a failure rate of
{rate:.2%}. The attrition is in `results/data/attrition_full_1h.csv`.

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
| `original_sample` | (generated) | 1 for the {orig:,} tasks of the first version's matrix | binary |

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
"""

CARD = """---
license: cc-by-4.0
pretty_name: KANVAS GPU job failure features (Alibaba PAI 2020)
task_categories:
- tabular-classification
tags:
- gpu-cluster
- failure-prediction
- interpretability
- shortcut-learning
size_categories:
- 100K<n<1M
configs:
- config_name: full_trace
  data_files: kanvas_feature_matrix.csv.gz
- config_name: benchmark
  data_files: kanvas_benchmark_results.csv
---

# KANVAS GPU job failure features

Ten leak-free features and a failure label for {tasks:,} tasks of the Alibaba PAI GPU cluster trace of 2020,
with the benchmark results of the KANVAS study: an additive Kolmogorov-Arnold model against logistic regression,
an MLP, LightGBM and two Explainable Boosting Machines under four evaluation protocols.

**Draft card, not uploaded.** Prepared locally with the v2 study.

## Source and licence

Derived from Alibaba's cluster-trace-gpu-v2020 (https://github.com/alibaba/clusterdata), released under
CC BY 4.0. Please cite the trace paper: Q. Weng et al., "MLaaS in the Wild: Workload Analysis and Scheduling in
Large-Scale Heterogeneous GPU Clusters", NSDI 2022. These derived files are released under the same licence.

## Contents

See `data_dictionary.md`. `kanvas_feature_matrix.csv.gz` has one row per task: the four resource requests, the
GPU count of the assigned machine, five averages of other jobs' telemetry on that machine in the hour before the
task started, the label (1 = Failed), the start time and a flag for the first version's {orig:,}-task sample.
Failure rate: {rate:.2%}.

## Intended use and an important limitation

The features predict failures for **users already seen in training** and transfer poorly to **new users**.
{limitation}
One user submits {u1t:.1%} of the tasks and {u1f:.1%} of the failures, so random task-level splits of this data
overstate how well a model would do on a new user's jobs. Use grouped or time-ordered evaluation, and report
known and new users separately.

## Privacy

No user, job, task, worker or machine identifier is included. The user was used only as a split key.
"""


def limitation_sentence():
    """One sentence with the user-grouped and new-user AUC ranges, when the analysis exists."""
    import pandas as pd

    p = ROOT / "results" / "analysis" / "full" / "model_ci.csv"
    if not p.exists():
        return "(Numbers are filled in after the analysis.)"
    ci = pd.read_csv(p)
    g = ci[(ci.protocol == "grouped") & (ci.subset == "all")]
    n = ci[(ci.protocol == "time") & (ci.subset == "new users")]
    k = ci[(ci.protocol == "time") & (ci.subset == "known users")]
    return (f"On the full trace, test ROC-AUC under a user-grouped evaluation ranges from {g.auc.min():.3f} to "
            f"{g.auc.max():.3f} over the {g.model.nunique()} models fitted on the full trace; under a time-ordered split it ranges from {k.auc.min():.3f} "
            f"to {k.auc.max():.3f} on known users and from {n.auc.min():.3f} to {n.auc.max():.3f} on new users.")


def write_docs():
    facts = json.loads((ROOT / "results" / "data" / "data_facts.json").read_text(encoding="utf-8"))
    sc = facts["status_counts"]
    u = facts["users_full"]
    fmt = {"labelled": sc["Failed"] + sc["Terminated"], "tasks": u["tasks"], "rate": u["failures"] / u["tasks"],
           "orig": facts["users_original"]["tasks"], "u1t": u["u1_task_share"], "u1f": u["u1_failure_share"],
           "limitation": limitation_sentence()}
    (PUB / "data_dictionary.md").write_text(DICTIONARY.format(**fmt), encoding="utf-8")
    (PUB / "README.md").write_text(CARD.format(**fmt), encoding="utf-8")
    print("wrote data_dictionary.md and README.md (dataset card draft)")
