# KANVAS v2 evaluation protocol

Written on 2026-11-23, after the v2 datasets were built and described and before any model was
trained on them or any test label was read. The experiment runner records the SHA-256 of this
file in every entry of the test-access log (`results/test_access_log.jsonl`), so a later edit
would be visible. Deviations, if any, are appended in the last section with their reason; the
text above that section is not edited after the first test access.

## 1. Datasets

| name | definition | tasks |
|---|---|---|
| full | every labelled task of the trace that survives the frozen v1 rules at a 1 h lookback (`scripts/build_full_trace.py`, `data/processed/v2/full_1h.parquet`) | 539,816 |
| original | the 18,890 tasks of the v1 paper, taken from the full build (identical values before clipping; checked in `results/data/build_checks.json`) | 18,890 |
| lookback | the v1 40,000-task stratified sample, rebuilt at 0.5, 1, 2, 4 and 6 h | 17,751 to 20,067 |

The build does not clip or scale. Every protocol fits clipping (1st and 99th percentiles) and
min-max scaling on its training split alone and applies the fitted constants to the validation
and test splits (`kanvas/preprocess.py`). All models receive the same transformed inputs.

The submitting user (`pai_job_table.user`) and the first instance's start time are split keys
only. Neither is a model input, and neither appears in any exported file.

## 2. Protocols

Every protocol is run on both `full` and `original`.

- **(a) random**: stratified 70/15/15 split (test first, then validation from the rest), split
  seeds 42, 43, 44, 45 and 46 (the v1 seeds; split 42 is the canonical split).
- **(b) time-ordered**: tasks sorted by the first instance's start time (stable sort, ties in
  dataset order). The earliest 70% train, the next 15% validate, the last 15% test. All tasks
  have telemetry, so the split covers the telemetry span only. A test task is *known-user* when
  its user has at least one training task and *new-user* otherwise; test metrics are reported
  for all test tasks and for each subset.
- **(c) user-grouped**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` with the
  user as the group. For outer fold k, the remaining users are split by
  `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=k)` and the first inner fold is
  the validation set. No user appears in two of train, validation and test. Results are reported
  as mean and standard deviation over the five folds, and per fold.
- **(d) leave-dominant-user-out**: User 1 is the user with the most tasks; User 4 is the user
  ranked fourth by tasks in the original sample (second by failures). Variant d1 removes User 1,
  variant d14 removes Users 1 and 4; each then uses the random split of (a) with split seed 42.

## 3. Models and selection

All models are trained on the training split. Each stochastic model is trained with
initialisation seeds 0, 1, 2, 3 and 4. Hyperparameters are chosen per split on the validation
split, using seed 0, by validation ROC-AUC (ties within 1e-4 go to the more regularised setting);
the chosen setting is then used for all five seeds. The test split is never used for a choice.

| model | fixed settings | validation grid |
|---|---|---|
| KAAM | 10 interior knots, cubic, SiLU base, minibatch Adam (lr 0.01, batch 2,048), at most 200 epochs, early stopping after 10 epochs without a lower validation log loss, best epoch restored | smoothness weight lambda in {0, 1e-4, 1e-3, 1e-2} |
| MLP | ReLU, same loop, batch, epoch cap and patience as KAAM | hidden (8, 6) or (12,) (149 and 145 parameters, KAAM has 151); lr in {0.003, 0.01} |
| logistic regression | L2, lbfgs, max_iter 10,000 | C in {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100} |
| LightGBM | lr 0.1, at most 2,000 rounds, early stopping after 50 rounds on validation log loss, bagging 0.8 every round, feature fraction 0.8 | num_leaves in {31, 127} x min_child_samples in {20, 200} |
| EBM (additive) | `ExplainableBoostingClassifier(interactions=0)`, other settings at the interpret 0.7.8 defaults, fitted on the training split (its own early stopping uses an internal holdout of that split) | none |
| EBM (interactions) | the same with the default `interactions` | none |

KAAM's B-spline basis depends only on the fixed knots and the inputs, so it is computed once per
split and the coefficients are fitted on it; the fitted parameters are loaded into the standard
`KAAM` module, whose `forward` gives the same output (checked by a test). Logistic regression is
deterministic: its five seeds give identical predictions, so it is fitted once per split.

Compute rule, fixed now: on `full`, the EBM with interactions is fitted with seed 0 only (about
ten minutes per fit on this laptop); on `original` it uses all five seeds, which measures its
seed variation. A neural additive model is not run.

## 4. Metrics

ROC-AUC is the primary metric. PR-AUC (average precision), the Brier score and the expected
calibration error (15 equal-width bins) are reported with it, and reliability diagrams use the
same bins. No decision threshold is tuned; the case audits use 0.5.

A model's score on a protocol is the mean over its test sets (splits or folds) of the mean over
seeds.

## 5. Statistics

- **Paired bootstrap**: 2,000 resamples (seed 0). For protocols (a), (b) and (d) the resampled
  unit is the task; for (c) it is the user, since the user is the unit of generalisation there.
  Each test set is resampled independently and the protocol score is recomputed as above; the same
  resamples are used for every model, so differences are paired. Intervals are 95% percentile
  intervals; the two-sided p-value is twice the smaller tail share with the +1 correction.
- **Families and correction**: in each results table, the comparisons of KAAM with each other
  model (logistic regression, MLP, LightGBM, both EBMs) within one protocol and dataset form one
  family; p-values are Holm-adjusted within it. A comparison is called *not significant* when the
  adjusted p-value is at least 0.05.
- **Equivalence margin**: KAAM is said to *match* a model on a protocol when the 95% interval of
  the difference lies inside [-0.01, +0.01] AUC.
- **Grouped AUCs against chance**: for each model, the user-level bootstrap of the fold-mean AUC
  gives an interval and a two-sided p-value for AUC = 0.5; the six p-values are Holm-adjusted.
  Per-fold AUCs are reported with user-level intervals.
- **Mechanism check for (c)**: for every fold, the failure rate of one-core tasks
  (`plan_cpu` = 100) against all other tasks among training users and among test users, and the
  AUC of `-plan_cpu` alone on each side.
- **Lookback ablation**: on the lookback datasets, every task keeps the role it has in the
  canonical 1 h split of the original sample; tasks that exist only at wider windows get their own
  stratified 70/15/15 split (seed 42). KAAM (seed 0, lambda chosen on the 1 h validation split) is
  compared with the 1 h model on the test tasks both score; the four p-values are Holm-adjusted.

## 6. Interpretability analyses

- **Canonical model**: KAAM on `full`, random split 42, seed 0, at its selected lambda.
- **Exact SHAP check**: Kernel SHAP on the canonical model's logit, background of 100 training
  tasks, 40 test tasks, every coalition enumerated; compared with the centred KAAM contributions.
- **KAAM against EBM shapes**: for each feature, the canonical KAAM curve and the additive EBM
  term (split 42, seed 0) on 200 points over the training 5th-95th percentile in raw units, each
  centred on its mean over the grid; Pearson correlation per feature.
- **Case audits**: on the canonical model's test set at threshold 0.5, the most confident true
  positive, the most confident true negative, the false negative with the highest risk and the
  false positive with the highest risk; contributions centred on the mean training task.
- **User concentration**: share of failures held by the top-k users, on both datasets.
- **plan_cpu across protocols**: the seed-0 KAAM `plan_cpu` curve from every protocol, centred on
  the same raw-unit window.

## 7. Test access

Only `kanvas/testaccess.py` returns test labels, and every call appends one line to
`results/test_access_log.jsonl` (time, dataset, protocol, split, model, purpose, SHA-256 of this
file). Predictions on test tasks are written to `results/predictions/` when a model is fitted;
labels are read only through the logged function. The appendix of the full paper lists the log.

## 8. Deviations

None at the time of writing.

**Deviation 1 (2026-11-23, 16:10 local time, before any test access; the test-access log was empty).**
Early stopping for KAAM, the MLP and LightGBM now monitors validation ROC-AUC instead of validation
log loss: training stops after 10 epochs (50 boosting rounds for LightGBM) in which the best
validation AUC has not risen by more than 1e-4, and the best epoch is restored. The epoch cap of the
two networks is raised from 200 to 500. Reason, found on validation data only: under the
time-ordered split the validation failure rate is 0.68 against 0.21 in training, so a barely trained
network that predicts about 0.5 has a lower validation log loss than a trained one. On the original
sample, all five MLP seeds stopped at epoch 1 (validation AUC between 0.11 and 0.78) and one KAAM
seed did the same. On the full random split, two of five MLP seeds reached the 200-epoch cap
without stopping. ROC-AUC is already the selection metric, so early stopping and selection now use
the same quantity under every protocol. Every KAAM, MLP and LightGBM fit made under the old rule
(original: random s42-s46 and time; full: random s42) was moved to `results/superseded/` and refitted.
Logistic regression and both EBMs do not use this early stopping and were kept.

**Deviation 2 (2026-11-23, 16:30 local time, before any test access).** On the full trace the
additive EBM is fitted with seed 0 only, as the EBM with interactions already was. Reason: with
four runner processes sharing the laptop's two physical cores, one additive-EBM fit on a full
training split took more than 20 minutes, so five seeds over the 13 full-trace splits would have
added about 20 hours of compute. EBM seed variation is measured on the original sample, where
both EBMs keep five seeds. The full-trace processes were restarted with this rule; no EBM fit on
the full trace had finished.

**Deviation 3 (2026-11-23, 16:45 local time, before any test access).** On the original sample the
EBM with interactions keeps five seeds on the five random splits, where its seed variation is
measured, and uses seed 0 under the time-ordered, user-grouped and leave-user-out protocols. Reason:
under the shared load one fit took 8 to 14 minutes, so five seeds on the seven remaining
original-sample splits would have added about seven hours. The time-ordered split of the original
sample, whose five seeds were already running, keeps them.

**Deviation 4 (2026-11-23, 21:00 local time, before any test access).** On the full trace neither
EBM is fitted under the protocols, except the additive EBM on the canonical split (random split 42,
seed 0), which is kept for the comparison of shape functions. Reason: one additive EBM fit with the
package defaults on a full training split (about 378,000 tasks) had used more than two hours of CPU
time, under four hours of wall time, without finishing; at that rate the 26 planned full-trace EBM
fits would have needed about 60 hours. The EBM comparisons under every protocol are made on the
original sample, where both EBMs were fitted on every split. The three unfinished full-trace EBM fits
(time-ordered, user-grouped fold 1, random split 43) were stopped.
