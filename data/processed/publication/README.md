---
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

Ten leak-free features and a failure label for 539,816 tasks of the Alibaba PAI GPU cluster trace of 2020,
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
task started, the label (1 = Failed), the start time and a flag for the first version's 18,890-task sample.
Failure rate: 27.39%.

## Intended use and an important limitation

The features predict failures for **users already seen in training** and transfer poorly to **new users**.
On the full trace, test ROC-AUC under a user-grouped evaluation ranges from 0.566 to 0.641 over the 4 models fitted on the full trace; under a time-ordered split it ranges from 0.580 to 0.599 on known users and from 0.462 to 0.534 on new users.
One user submits 31.0% of the tasks and 54.6% of the failures, so random task-level splits of this data
overstate how well a model would do on a new user's jobs. Use grouped or time-ordered evaluation, and report
known and new users separately.

## Privacy

No user, job, task, worker or machine identifier is included. The user was used only as a split key.
