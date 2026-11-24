# Independent review of the IEEE version

A fresh reviewer agent, with no part in writing the paper, reviewed `paper/ieee/main_ieee.pdf` (the 5-page build of
2026-11-23, 22:19) as an IEEE conference reviewer. It consulted the full version, the protocol and the result
files to spot-check claims. The review is reproduced below unchanged, followed by the response: what was fixed,
what was declined and why, and what remains open because it needs new experiments.

**Disclosure.** For its spot checks the reviewer read test labels directly from
`data/processed/v2/full_1h.parquet`, outside the logged function `kanvas/testaccess.py`. That read happened after
every model had been fitted and every analysis of the protocol had run. It is recorded as a manual entry in
`results/test_access_log.jsonl` (protocol `review`). No model, setting or protocol entry was changed because of
it. The post hoc analyses added in response (`scripts/posthoc_review.py`) read test labels only through the logged
function.

---

## The review

### Review: "KANVAS: Auditing Shortcut Learning in GPU Job Failure Prediction with Additive Kolmogorov-Arnold Models" (IEEE conference version, 5 pages)

I re-extracted the text of the rebuilt `paper/ieee/main_ieee.pdf` (22:19 build); all quotes below are from that version. For spot checks I read the LaTeX sources, `paper/PROTOCOL.md`, `results/analysis/{full,original}`, `results/interpret`, the test-access log, `scripts/analyze.py` / `kanvas/metrics.py`, and ran read-only computations on `data/processed/v2/full_1h.parquet` and the saved test predictions (nothing was written to disk; note these computations read labels directly from the parquet file rather than through `kanvas/testaccess.py`, so they are not in the access log).

#### 1. Summary of the paper

The paper takes the Alibaba PAI 2020 GPU trace and builds ten features per labelled task: four resource requests, the machine's GPU count, and five telemetry averages computed from *other* jobs' workers on the same machine that ended in the hour before the task started, a window built to avoid a label leak (own-task telemetry covers 42.4% of terminated vs 0.09% of failed instances). 539,816 of 1,141,835 labelled tasks survive. Under four protocols said to be fixed in advance (random splits, a single time-ordered 70/15/15 split with known-/new-user test subsets, 5-fold user-grouped CV, and random splits with User 1, or Users 1 and 4, removed) it compares an additive one-layer KAN ("KAAM", a P-spline GAM with 151 parameters) with logistic regression, a ~150-parameter MLP, LightGBM, and two EBMs (fitted only on an earlier 18,890-task sample). Headline results: LightGBM leads on random splits (0.900 vs 0.766); under user-grouped CV KAAM has the highest point estimate (0.641 vs 0.566 for LightGBM, not significant); on the time-ordered split every model scores 0.576–0.599. KAAM's `plan_cpu` curve exposes a "one-core" rule (+1.73 log-odds) attributed to User 1 (54.6% of failures), which shrinks to +0.11 when User 1 is removed. The paper concludes that random splits mostly measure recognition of known users and recommends grouped and time-ordered evaluation.

#### 2. Scores (1 to 5)

| Criterion | Score | Justification |
|---|---|---|
| Soundness | 2 | Descriptive numbers are internally consistent, but the comparative claims are fragile: "KAAM scores highest with users held out" rests on a fold-mean driven by one 3-user fold; LightGBM's grouped collapse comes from fits that stopped after 2–8 boosting rounds; EBMs were validated on a random holdout; the "loses AUC without User 1" comparison mixes test populations; the bootstrap ignores training variance and clustering. |
| Significance | 3 | A useful warning for systems-ML work on cluster traces (telemetry-coverage leak, known-user inflation), but the general lesson (group by the unit of generalization) is well established. |
| Novelty | 2 | The paper itself says a one-layer KAN with one edge per input is a P-spline GAM; nothing in the audit requires KAN-specific properties, and EBM-A or any GAM gives the same exact additive contributions. The novelty is an empirical case study of one trace. |
| Clarity | 3 | Compact and precise, with every number generated from result files. But several key terms are undefined (canonical split, User 1/4 numbering, "the rule fixed in the first version", grid vs data correlation), figures are dense, and important context sits in an unreferenced "full version". |
| Reproducibility | 3 | Public data and an exceptionally complete repository (protocol, log, macros). As submitted, though, the paper gives no link to code or the full version and omits several settings (LightGBM/MLP/LR grids, KAAM learning rate and epoch cap, ECE bins, bootstrap design, CV seeds). |
| Overall recommendation | 2 (weak reject / major revision) | The audit framing and leak analysis are worth publishing, but the headline comparative claims need re-analysis, and several need new experiments. |
| Reviewer confidence | 4 | Claims were checked against the released result files and several quantities recomputed from saved predictions. |

#### 3. Strengths

- **Identifies and quantifies a label-revealing leak.** Telemetry presence covers 42.4% of terminated vs 0.09% of failed instances. The paper replaces own-task telemetry with a clearly specified pre-start neighbour window (Fig. 1) that excludes the task's own job and any row ending at or after t0. This is a valuable, concrete warning for anyone using this trace.
- **Self-correction.** The authors find and fix a leak in their own earlier version (global percentile clipping) and say so plainly.
- **Sound protocol design** separating known from new users. It includes a time-ordered split with known-/new-user subsets and a leave-dominant-user-out variant, and hyperparameters are selected on validation data only.
- **Process transparency.** There is a protocol file whose SHA-256 is written into every test-access log entry, and four timestamped pre-access deviations are disclosed with reasons. In the repository, all 32 log entries carry the hash of the current `PROTOCOL.md`, and the first test access (16:20 UTC) comes after the last deviation (21:00 IST = 15:30 UTC).
- **Negative results are reported honestly:** KAAM matches none of the baselines on random splits, all models are weak on the time split, and the change from the first version is acknowledged.
- **Internally consistent numbers.** Every number is a macro generated from result files; about 35 values checked against the CSV/JSON results all matched. The full-trace build reproduces the v1 matrix (0 missing rows, identical labels, max relative difference 3e-16).
- **The case audit is concrete and persuasive as an illustration.** The most confident true positive's request appears in 44,191 training tasks, 87.5% of which failed and 99.5% of which come from User 1.
- **Candid limitations section** on selection (47.3% of tasks kept) and the proxy nature of the telemetry.

#### 4. Major issues

**M1. "KAAM scores highest when users are held out" depends on one fold with 3 test users.** (Fixable without new experiments: re-analysis.)
*Where:* Abstract ("With users held out the lead vanishes: LightGBM falls to 0.566 and KAAM, at 0.641, scores highest"), contribution bullet 3, §VI "User-grouped folds", Table I "Grouped", Fig. 2, Conclusion.
*What is wrong:*
- Fold sizes are very unequal. In the repository (`fold_auc.csv`, `split_facts.csv`), full-trace fold f4 has **3 test users and 192,747 test tasks** (35.7% of the data); the other folds have 277 to 313 users and about 86,700 tasks each.
- Fold f4 is essentially a leave-User-1-out fold. Its three users are User 1 (167,362 tasks, 48.3% failure rate) and two users with 1.2% and 6.5% failure rates.
- Much of KAAM's f4 score is between-user discrimination: 0.754 on f4, but only 0.697 within User 1's tasks.
- On the four folds with ~280 held-out users each, the ranking flips: MLP 0.624, KAAM 0.613, LR 0.588, LightGBM 0.553. So the MLP, not KAAM, is highest on the folds that actually test new users.
- The ranking depends on how folds are combined. Pooling the five folds' seed-averaged predictions gives KAAM 0.702, MLP 0.655, LR 0.587, LightGBM 0.543.
- The fold standard deviation is 0.106 over 5 folds.
- The original sample has the same structure. Its fold f2 has 3 test users and 6,693 tasks; KAAM scores 0.728 and EBM-A 0.330 there. Without that fold KAAM still leads (0.593 vs 0.563 for LR), but each of the remaining folds has only about 3,000 test tasks.

*Why it matters:* This is the main comparative claim, and it appears in the abstract and the conclusion.
*Fix:* Report the 3-user fold separately (as the leave-User-1-out experiment it effectively is), report the four many-user folds as the new-user estimate, show the fold SD or CI in Table I, and replace "scores highest" with "no model is distinguishable". Ideally, use a grouping that balances tasks across folds, or always hold User 1 out separately (that last step needs reruns).

**M2. LightGBM's grouped "collapse" is a degenerate-training artifact.** (Needs new experiments.)
*Where:* §VI "User-grouped folds"; Discussion ("their extra accuracy consists largely of user-specific request combinations that an additive model cannot form").
*What is wrong:*
- The fits stopped almost immediately. `selection.csv` shows LightGBM's mean best iteration under grouped CV was **1.8 rounds (f3), 3.4 (f4) and 7.8 (f0)**.
- Seed variance is huge. On f4 the five seeds score 0.445, 0.780, 0.554, 0.756 and 0.547; the grouped seed SD is 0.044 for LightGBM vs 0.003 for KAAM.
- Two folds drive LightGBM's grouped mean of 0.566: it scores 0.490 on f3, where LR, KAAM and the MLP score 0.76 to 0.80, and 0.616 on f4.
- The LightGBM grid ({31, 127} leaves × {20, 200} minimum child samples) has no strongly regularized setting. KAAM's smoothing penalty, by contrast, was chosen at the grid's upper bound (1e-2) in 4 of 5 grouped folds and on the time split.
- No evidence shows that *interactions* are what fail to transfer.

*Fix:* Use a regularized GBDT grid (depth 2 to 4, 4 to 15 leaves, minimum child samples of at least 1,000, L2, a lower learning rate, a minimum number of rounds). Add an **interaction-constrained (additive) LightGBM**, which tests "interactions vs additivity" directly. Average predictions over seeds or report seed spread.

**M3. The EBM comparisons are unfair and partly misreported.** ((a) and (b) need new experiments; (c) is fixable.)
*Where:* §IV ("The EBMs are fitted on the original sample only, since one fit on a full training split takes hours of CPU time"), §V, Table I note, Limitations.
- **(a) Early stopping was neither group- nor time-aware.** Per `PROTOCOL.md` §3 the EBMs are untuned package defaults (interpret 0.7.8). Their early stopping uses **an internal random holdout of the training split**, not the protocol's user-grouped or time-ordered validation set that every other model used. Under the grouped and time protocols this rewards memorizing users and periods, and plausibly explains EBM-A at 0.489 and EBM-P at 0.504 on the original-sample grouped folds (EBM-A scores 0.330 on the 3-user fold).
- **(b) The closest additive peer is missing** from the full-trace panel, which carries all the headline comparisons.
- **(c) The statement that the EBMs were fitted only on the original sample is inaccurate.** An additive EBM *was* fitted on the full-trace canonical split; its shapes are used in Fig. 4, and it scored **0.783 vs 0.766 for KAAM** (`summary.csv`; reported in the full version, omitted here).

*Fix:* Report the full-trace EBM-A result, state the validation asymmetry, and refit the EBMs using the protocol's validation set for early stopping (or tuned outer bags). Also fit EBM-A on the full-trace splits using fewer outer bags or subsampling; a 4.3-hour fit (15,646 s) on a shared 4-thread laptop is not a real barrier for a conference paper.

**M4. "Without User 1 every model loses AUC" confounds the model with the test population.** (Fixable.)
*Where:* §VI "Leaving out the dominant users" ("Without User 1 every model loses AUC (KAAM -0.062, LightGBM -0.055)").
*What is wrong:* The two AUCs are measured on different populations. User 1's tasks fail at 48.3% vs about 18% for other users, and that gap makes ranking easier when User 1 is in the test set.
*Like-for-like check:* Scoring the split-42 canonical models only on non-User-1 test tasks, from saved predictions, gives:

| Model | Trained with User 1, scored on non-User-1 tasks | Trained without User 1 (-U1 split), scored on its test set |
|---|---|---|
| KAAM | 0.671 | 0.703 |
| LR | 0.619 | 0.656 |
| MLP | 0.724 | 0.721 |
| LightGBM | 0.836 | 0.845 |
| EBM-A | 0.694 | (not fitted on full) |

Both columns cover the same population, drawn by different random splits. Removing User 1 from training *improves* the additive models on other users by about 0.03 to 0.04 and leaves the MLP and LightGBM unchanged. That is the opposite of the paper's framing, and it is actually better evidence for the shortcut claim.
*Fix:* Report like-for-like comparisons.

**M5. Random-split inflation is largely exact duplicates, not only "known users".** (Reporting is fixable; a corrected split needs new experiments.)
*Where:* Discussion ("A random split of this trace tests recognition of known users"), the deployment advice, and Related Work, which cites dependent-sample leakage [19] but never checks for it.
*What I found:*
- **29.3% of split-42 test tasks (23,683) have an exact 10-feature duplicate in training.** For those tasks, the duplicate's mean training label alone ranks failures at **AUC 0.927**.
- 62.9% of these tasks belong to User 1.
- The time-ordered test set has **0%** exact duplicates in training.
- Same-job leakage is negligible: there are 533,544 jobs for 539,816 tasks, and only 1.6% of test tasks have a sibling task in training. So the duplicates are concurrent submissions with the same request on the same machine and window.

*Why it matters:* The mechanism behind LightGBM's 0.900 may be memorization of near-simultaneous twins whose outcomes are not available at prediction time. That is a different conclusion from "recognizing known users' recurring pipelines".
*Fix:* Report the duplicate rate, add a de-duplicated or burst/time-blocked random split, and add a nearest-neighbour or request-lookup baseline.

**M6. The time-ordered protocol has a pathological validation period and only one split.** (Needs new experiments.)
*Where:* §V, §VI "Time-ordered split", Abstract ("every model scores 0.576 to 0.599, even on known users"), Discussion ("with frequent retraining").
*What is wrong:*
- The validation period is **73.2% User 1 tasks**, against 22.2% in training and 29.7% in test.
- Its 67.9% failure rate drops to **12.7% once User 1 is excluded**; the paper reports the rate without explaining it.
- Validation AUCs are 0.83 to 0.95 against test AUCs of 0.58 to 0.60. The networks early-stopped at epochs 5 to 6 and LightGBM at 27 rounds.
- Model selection, and Deviation 1, were driven by this burst.
- Test data start about 2.9 days after training ends, and this gap is not stated.
- Retraining, which the Discussion recommends, is never evaluated.

*Fix:* Describe the validation period's composition. Use rolling-origin evaluation with several cutoffs and short horizons (for example, retrain daily and test on the next day). Take validation from the end of the training window, or make it user-stratified. Report known-user AUC as a function of the gap.

**M7. Statistical inference problems.** (Mostly fixable.)
*Where:* §V and all ° marks in Table I.
- **(a) Training variance is left out.** The bootstrap resamples test units with the models fixed (`bootstrap_models`). This ignores training and fold variability, which dominates: the grouped fold SD is 0.106 against KAAM's seed SD of 0.003. CIs like 0.766 [0.764, 0.767] say nothing about generalization.
- **(b) Clustering is ignored.** Task-level resampling under the random, time and leave-user-out protocols ignores clustering by user (one user has 31% of tasks) and by duplicates. The only significant new-user result, LightGBM 0.534 vs KAAM 0.466 (Holm p = 0.009), treats 748 tasks from **79 users** as independent.
- **(c) The 3-user fold's user bootstrap is degenerate.** It has only 10 distinct multisets of users. The asymmetric percentile intervals are a symptom: 0.641 in [0.527, 0.670], and the difference +0.076 in [-0.031, 0.107].
- **(d) Multiple comparisons are under-corrected.** Holm is applied only within each protocol×subset family. On the full trace each family has **three** comparisons, not the "five comparisons per protocol" the paper states. Roughly 56 comparisons plus chance tests across both datasets get no cross-family correction, and the known/new subsets are post-hoc strata.
- **(e) Non-significance is used as support.** "Scores highest, though not significantly" leans on a non-result. The protocol defines a ±0.01 equivalence margin that the IEEE text never uses; under it, KAAM vs MLP in grouped CV (difference CI [-0.032, 0.023]) is inconclusive.

*Fix:* Use a user-cluster bootstrap everywhere, report the fold SD (or corrected resampled t-tests in the style of Nadeau-Bengio), describe the families accurately, and drop rankings the intervals do not support.

**M8. The new-user narrative contradicts Table I.** (Fixable.)
*Where:* Contribution bullet 3 ("the interaction models' random-split lead disappears on new users, where KAAM scores highest"), Abstract.
*What is wrong:*
- On the full trace's time-split new-user subset, LightGBM (0.534) is best and significantly above KAAM (0.466, below chance). Table I shows LightGBM in bold with no °, yet the text gives only the range "0.462 to 0.534".
- On the original sample, LR is best on new users (0.565 vs KAAM 0.515, on 123 tasks).

So in two of the paper's own new-user views, KAAM is not highest.
*Fix:* State this and narrow the claim.

**M9. The abstract and conclusion over-claim.** (Fixable.)
- **(a)** "The features serve known users' recurring jobs" rests on random-split evidence the paper argues is inflated (M5 makes this worse). On known users a few days later, AUC is 0.580 to 0.599.
- **(b)** "The readable model's curves and exact contributions showed the reason, a one-user shortcut" is offered as explaining why *every* model loses accuracy. But LightGBM's random-split lead persists without Users 1 and 4 (0.805 vs 0.672), so the User-1 rule cannot explain LightGBM. "Its random-split lead is spread over the many users..." is a hypothesis stated as fact.
- **(c)** The mechanism paragraph reports one-core failure rates of "15.0% to 36.5% among the held-out users" without the comparison rates for other tasks. In f0 and f1, held-out one-core tasks fail at 27.8% and 36.5% vs 17.4% and 17.3% for other tasks, so the rule partially transfers (1.6 to 2.1 times the base rate). The effect reverses only in f2 and f3, and in f3 -`plan_cpu` alone gives 0.717.
- **(d)** "The rule vanishes when that user is removed" holds only for the leave-user-out curves (+0.11 for -U1, -0.21 for -U1,4). In grouped fold f4, whose training set also excludes User 1, the one-core value is still **+0.82** (`plan_cpu_protocols.csv`).

**M10. The interpretability claims rest on unstable, non-identifiable curves.** (Fixable without new experiments; curve bands need minor reruns.)
*Where:* Abstract ("ten B-spline curves are its complete explanation"), §VI "Interpretability", Fig. 4, Discussion.
*What is wrong:*
- The curves are a complete account *of the model*, not a stable account *of the data*. The canonical `plan_cpu` curve correlates **0.22** with the time-split curve and **0.25 to 0.49** with the grouped-fold curves (`plan_cpu_summary.csv`). KAAM and EBM-A fitted on the same split correlate **0.04** on `plan_mem` and `plan_gpu` on the pre-specified grid.
- The text reports the data-weighted median (0.85, described as exploratory in the full version) instead of the protocol's grid median (0.75), and does not label it exploratory.
- Fig. 4 shows only one telemetry panel, although the text's claim of agreement concerns the telemetry curves.
- The exact-SHAP check (1.6e-7) holds by construction for any additive model, including EBM-A, so it does not argue for KAAM over a standard GAM.

*Fix:* Report the grid statistic as primary, discuss concurvity among the discrete request features (3,044 distinct request tuples), show seed and split bands, and justify KAAM over EBM-A, pyGAM or mgcv.

**M11. "Pre-registered" overstates the safeguards, and the original sample is not confirmatory.** (Fixable.)
*Where:* Contribution bullet 2; §V.
- The protocol was written on the same day as all the fits (2026-11-23), after the datasets were built and described.
- The author had already seen v1 results on the original sample with the same split seeds (42 to 46), and the User-1 hypothesis came from v1.
- The protocol is not in any external registry, and in the repository `paper/` is untracked by git. Its integrity rests on a hash in a self-maintained log, which is consistent but not independent evidence.

*Fix:* Say "a protocol fixed before test access (self-registered; hash recorded in the access log)", deposit it with a timestamp (OSF or Zenodo), and label the original-sample analyses as non-confirmatory.

**M12. The motivating premise is uncited, and simple baselines are missing.** (Partly needs new experiments.)
- The opening claim, "Failure predictors for shared GPU clusters are usually evaluated on random task-level splits", cites no failure-prediction study on PAI or similar traces that uses such splits.
- Simple baselines would quantify memorization directly: a request-tuple lookup or kNN baseline (the 44,191-task tuple suggests it would be strong), and a standard P-spline GAM or NAM to show what the KAN parameterization adds, if anything.

#### 5. Minor issues

1. §V says Holm is applied "over the five comparisons per protocol"; on the full trace there are three (no EBMs).
2. Abstract: "Ten leak-free features are built for every labelled task (539,816 survive)". Fewer than half survive (47.3%); rephrase.
3. The attrition in §III does not add up: 1,141,835 - 539,816 = 602,019, but 306,674 + 251,776 = 558,450. The 43,537 tasks with no resolvable machine telemetry are not mentioned.
4. Several terms are undefined in the IEEE version: "canonical split" (random split 42, seed 0); the User 1 / User 4 numbering (rank by task count); "the rule fixed in the first version" for case selection; "grid" vs "data" correlations in Fig. 4; "first seed" in Fig. 3 (seed 0); "mean training task" (mean logit or logit of the mean task?).
5. "a ReLU MLP with 149 and 145 parameters" should say there are two architectures, (8,6) and (12).
6. §V promises PR-AUC and the Brier score, which are never reported. The ECE bin count (15) is not stated. ECE outside random splits is not reported, although it is large: 0.105 to 0.133 under grouped CV and about 0.37 for time-split new users.
7. The one-core `plan_cpu` value is +1.73 in the text and Fig. 3 but +1.68 in the case audit, on the same canonical model. The centring differs (raw-unit window vs training mean); explain this.
8. Table I:
   - It shows no dispersion (the grouped fold SD is 0.106).
   - Bold marks "best" on a 123-task subset (original sample, New).
   - Known and New are subsets of Time and need a spanning header.
   - LightGBM 0.566 carries a ° while LR 0.600 is significant, which will puzzle readers unless the variance is explained.
   - The table note's EBM statement is inaccurate (M3c).
9. Fig. 2 has no fold labels, sizes or user counts, and one fold's "user-level 95% interval" rests on 3 users.
10. Fig. 3: the legend entries ("user-grouped folds", "without User 1 / Users 1, 4") each cover several curves, the centring reference for "log-odds" is not given, and the x-axis is in cores while Fig. 4 uses "100 = 1 core". Make the units consistent.
11. Fig. 5: define "p12" (training percentile) in the caption and state what the -1.23 baseline is.
12. The Discussion's deployment sentence ("could support operators ... with frequent retraining") recommends something never evaluated.
13. §V says "Four changes ... are documented in the full version" but describes two, and the EBM one repeats §IV.
14. "Our earlier version" / "the first version" and "the full version" are never cited or linked, so reviewers cannot check v1, the deviations, the code or the data. "Code, data and every result file are released with the full version" needs a URL (anonymized if required).
15. Missing implementation details:
    - LightGBM grid values, learning rate, and bagging and feature fractions;
    - the MLP learning-rate grid and the LR C grid;
    - the EBM version and defaults;
    - the KAAM learning rate (0.01) and epoch cap (500);
    - the StratifiedGroupKFold seeds;
    - the time gap between the training and test windows.
16. Clarify what `Terminated` means as the negative class. Does it include user-killed jobs? (Fig. 5 calls it "finished".)
17. The abstract says "two Explainable Boosting Machines" without noting they were fitted only on the 18,890-task sample.
18. "the models with interactions use that best" (Discussion) is vague.
19. References: [15] "21th" should be "21st"; [6] "english translation" should be "English translation"; cite scikit-learn for StratifiedGroupKFold. IEEE style prefers an en dash in "Kolmogorov–Arnold".
20. Voice: the paper uses "we" but the Acknowledgment uses "I" (single author). Pick one.
21. Fig. 4 shows six of the ten features, chosen without explanation, and two of them have grid r = 0.04, which contradicts the text's "agree closely" framing.

#### 6. Claims checked against the repository

| # | Claim (IEEE PDF) | Source checked | Result |
|---|---|---|---|
| 1 | 539,816 tasks survive, 27.4% failures, 1,159 users; 47.3% kept | parquet (539,816 rows), `build_checks.json` | Holds |
| 2 | 306,674 lost (no window) + 251,776 (missing value) | `numbers.tex` (`data.tele.nowindow`, `data.missing.tasks`) | Holds, but omits 43,537 no-machine tasks (attrition does not sum) |
| 3 | Build reproduces v1 matrix value for value; 2.7 min | `build_checks.json` (0 missing, identical labels, max rel. diff 3e-16; 162.5 s) | Holds |
| 4 | User 1: 31.0% of tasks, 54.6% of failures; User 4: 3.5%, 84.5%; together 65.5% | parquet (167,362 / 19,089 tasks), `user_concentration.csv` | Holds |
| 5 | Table I values (full and original) | `summary.csv`, `comparisons.csv` | Hold |
| 6 | Random: KAAM +0.033 vs LR, -0.038 vs MLP, -0.134 vs LightGBM; all significant | `comparisons.csv` | Holds (task-level bootstrap; see M7) |
| 7 | ECE at most 0.034 for LightGBM, MLP, KAAM (random) | `summary.csv` (0.017, 0.018, 0.034) | Holds |
| 8 | Time: 0.576–0.599; known 0.580–0.599; new 0.462–0.534; 99.1% known; 748 new tasks; rates 21.1/67.9/16.1%; final 7.0 days | `summary.csv`, `split_facts.csv` | Holds; validation is 73.2% User 1 (unstated); new-user tasks come from 79 users |
| 9 | Grouped: KAAM 0.641 [0.527, 0.670]; not significant vs MLP and LightGBM; LR significant | `model_ci.csv`, `comparisons.csv` (Holm p = 0.93, 0.56, 0.021) | Numbers hold; "scores highest" depends on 3-user fold f4 (M1) |
| 10 | Fold AUCs 0.486 to 0.799 | `fold_auc.csv` | Holds |
| 11 | Mechanism: 77.1–82.6% vs 15.0–36.5%; "mostly User 1's"; -plan_cpu 0.431–0.717 | `mechanism.csv`; recomputation (User 1 = 85.6–92.5% of one-core training tasks) | Numbers hold; omitting the other-task base rates misleads (M9c) |
| 12 | Without User 1: KAAM -0.062, LightGBM -0.055; LightGBM leads by 0.133 without Users 1 and 4 | `summary.csv` | Arithmetic holds; interpretation fails the like-for-like check (M4) |
| 13 | At most 19 of 80,973 random-split test tasks from unseen users | `split_facts.csv` (max 19, s45) | Holds |
| 14 | Kernel SHAP agreement within 1.6e-7 | `shap_check.json` (1.586e-7; 40 tasks, background 100) | Holds |
| 15 | plan_cpu +1.73 at one core (canonical), +0.11 without User 1 | `plan_cpu_protocols.csv` | Holds; grouped f4 (no User 1 in training) still gives +0.82 |
| 16 | λ = 0 on all random splits; strongest on time split and four of five grouped folds | `selection.csv` | Holds in the current PDF (an earlier build said "five") |
| 17 | Median KAAM–EBM correlation 0.85 at training values | `ebm_shape_corr_data.csv` | Holds, but it is the exploratory statistic; the pre-specified grid median is 0.75, with r = 0.04 for plan_mem and plan_gpu |
| 18 | Case audits: TP risk 0.861, plan_cpu +1.68, 44,191 tasks, 87.5% failed, 99.5% User 1; FP 0.878, 4.5% of 44 | `cases.json` | Holds |
| 19 | "The EBMs are fitted on the original sample only" | `summary.csv`, `selection.csv` (full-trace EBM-A on split 42: AUC 0.783, 15,646 s) | **Does not hold** |
| 20 | Holm "over the five comparisons per protocol" | `comparisons.csv`, `analyze.py` | **Does not hold on full trace** (families of 3) |
| 21 | No test label read before every model was fitted; protocol fixed beforehand | `test_access_log.jsonl` (first entry 16:20 UTC, after Deviation 4 at 15:30 UTC; all 32 entries carry SHA-256 39b91d83..., matching current `PROTOCOL.md`) | Consistent; "pre-registered" overstates it (same-day, self-hosted, untracked file) |
| 22 | Implicit: random-split inflation is about users, not jobs | own check (1.6% of test tasks have a same-job sibling in training; 29.3% have an exact feature duplicate) | Job leakage negligible; duplicate leakage substantial and unreported (M5) |

#### 7. Questions for the authors

1. How does the grouped comparison change if User 1's fold is reported separately and the four many-user folds are summarized on their own? (By my computation the MLP leads there, 0.624 vs 0.613.)
2. Why did LightGBM stop after 1.8 to 7.8 rounds in the grouped folds? Would a regularized or interaction-constrained LightGBM change the "interactions don't transfer" conclusion?
3. Were the EBMs given the protocol's grouped and time-ordered validation sets for early stopping? If not, can you refit them so that they are?
4. Why does the IEEE version leave out the full-trace EBM-A result (0.783 vs 0.766), whose shape functions appear in Fig. 4?
5. Are you aware that 29.3% of random-split test tasks have an exact feature duplicate in training, and that the duplicate's label alone gives AUC 0.927 on them? How does a de-duplicated or burst-blocked random split change Table I?
6. The time-split validation period is 73.2% User 1 tasks. How sensitive are the time-ordered results to that choice? What does rolling-origin evaluation with daily retraining show for known users?
7. For the leave-dominant-user-out protocol, will you report the full-data models scored on the same non-User-1 population? My check suggests removing User 1 improves the additive models on other users.
8. Why use a task-level bootstrap with fixed models for the random, time and leave-user-out protocols, given the user clustering? Will you add a user-cluster bootstrap and a fold-level variance estimate?
9. Under grouped CV, KAAM's one-core term is still +0.82 in the fold that excludes User 1 from training. How does that square with "the rule vanishes when that user is removed"?
10. What does the KAN parameterization add over a standard P-spline GAM (pyGAM or mgcv) or EBM-A, given that every audit step uses only additivity?
11. Which prior failure-prediction studies on GPU cluster traces used random task-level splits, as the abstract says is usual?
12. Does `Terminated` include tasks killed by users or the scheduler? Is `inst_num` the requested instance count or the number of instances actually created?
13. Will the v1 paper and the "full version" (protocol, deviations, code) be cited with persistent links?

---

## Response

Status codes: **fixed** (changed in both versions unless stated), **partly fixed**, **open** (needs new experiments or
an analysis that has not run), **declined** (with the reason). Numbers below are from the result files; in the paper
every one is a generated macro.

### Major issues

| # | Issue | Status | What changed |
|---|---|---|---|
| M1 | "KAAM scores highest" rests on the 3-user fold | **fixed** | The grouped results are now reported separately for the four many-user folds (MLP 0.624, KAAM 0.613, LR 0.588, LightGBM 0.553) and for the fold that holds User 1, which is described as in effect a test on User 1 alone, with three clusters. "Scores highest" is replaced by "no model is clearly best" in the abstracts, introductions, results, discussions and conclusions. The fold-mean interval and the margin argument are stated. A post hoc bootstrap that resamples whole users within the four folds gives wide intervals (KAAM [0.516, 0.657], MLP [0.531, 0.667], LR [0.487, 0.638], LightGBM [0.520, 0.603]), and none of KAAM's differences is significant (Holm p = 0.698, 0.306, 0.698); reported in both versions. A balanced re-grouping is **open**. |
| M2 | LightGBM's grouped collapse is a degenerate-training artifact | **partly fixed** | The text now reports that LightGBM's grouped fits stopped after 1.8 to 7.8 boosting rounds on average (versus far more on random splits) and its large seed spread, and it no longer claims that interactions fail to transfer; it says a regularised or interaction-constrained boosting model would be needed to test that. Running such models is **open** (new experiments). |
| M3 | EBM comparisons unfair and partly misreported | **(c) fixed; (a) partly fixed; (b) open** | (c) Both versions now say that under the protocols the EBMs were fitted on the original sample only and report the full-trace canonical additive EBM (0.783 vs 0.766 for KAAM). (a) The early-stopping asymmetry (the EBMs stop on an internal random holdout, not on the protocol's validation split) is stated in the model sections and the limitations; refitting the EBMs with protocol-aware early stopping is **open**. (b) Full-trace EBMs under every protocol are **open** (compute; deviation 4). |
| M4 | "Without User 1 every model loses AUC" mixes up the model and the test population | **fixed** | The text now states that this protocol changes the test population (User 1's tasks fail at 48.3% against 18.0% for the others), that its AUCs are not comparable with the random protocol's, and that it does not separate training effects from test effects. The claim that every model loses AUC is removed. The like-for-like analysis now runs (`scripts/posthoc_review.py`, which saves each part as it finishes). Scored on the 55,961 test tasks of split 42 that do not belong to User 1, the canonical models reach 0.671 (KAAM), 0.619 (LR), 0.724 (MLP) and 0.836 (LightGBM). Trained without User 1 and tested on another draw from the same population, KAAM gains +0.032 and LR +0.037, while LightGBM changes by +0.009 and the MLP by -0.003; without Users 1 and 4 the changes are +0.053, +0.092, +0.011 and +0.027. Rules learned from the dominant users cost the two models without interactions accuracy on other users. Reported in the leave-user-out results of both versions. |
| M5 | Random-split inflation is largely exact duplicates | **partly fixed** | Confirmed independently without reading test labels (`scripts/review_facts.py`): 29.2% to 29.3% of random-split test tasks have an exact ten-feature duplicate in training, 0.0% in the time-ordered split, 2.1% to 3.4% in the grouped folds, 11.8% to 15.2% in the leave-user-out splits, and 4.0% to 4.7% in the original sample. Reported in the random-split results, the leave-user-out paragraph, the abstracts, introductions, discussions and conclusions. The lookup on duplicates is now reported (post hoc): on split 42, scoring each of the 23,683 duplicated test tasks by the mean label of its identical training tasks gives an AUC of 0.927 on them; 62.9% of them belong to User 1 and they fail at 59.0% against 14.3% for the other test tasks. LightGBM scores 0.967 on duplicated and 0.780 on other test tasks, KAAM 0.811 and 0.644. Reported in both versions. A de-duplicated or burst-blocked split is **open** (new experiments). |
| M6 | Pathological validation period; one time split | **partly fixed** | The text now says the validation period is a burst in which User 1 submits 73.2% of the tasks (failure rate 67.9%, 12.7% without User 1), that selection on it overstated test accuracy, the networks' best epochs on it, and that the test week starts 2.9 days after the last training task. The untested "frequent retraining" advice is removed. Rolling-origin evaluation is **open** (new experiments). |
| M7 | Statistical inference | **fixed** (corrected resampled t-tests not added) | (a) Both versions and the table notes state that the intervals resample test tasks with the fitted models fixed and do not include training or fold variation, and the full version reports the spread over splits and folds. (b) The task-level resampling of the 748 new-user tasks from 79 users is flagged where that result is quoted; a post hoc user-cluster bootstrap now covers the random, time-ordered and leave-user-out protocols: on the random splits KAAM's interval widens to [0.678, 0.784] and all three differences stay significant; without User 1 LightGBM's lead stays significant (-0.141 [-0.177, -0.111]); no time-split difference is significant; and the new-user LightGBM lead (KAAM minus LightGBM -0.068 [-0.159, 0.011], Holm p = 0.270) is not significant once the 79 users are resampled. Reported in both versions. (c) Handled by M1. (d) The family sizes are corrected (three comparisons on the full trace, five on the original sample) in the text and the table notes. (e) Rankings that the intervals do not support are dropped, and the equivalence margin is used for the grouped KAAM-MLP comparison. |
| M8 | New-user narrative contradicts Table I | **fixed** | The time-split new-user result is now stated: LightGBM is best there and above KAAM, with the caveat on clustering, and the "KAAM scores highest on new users" claim is gone. |
| M9 | Abstract and conclusion over-claim | **fixed** | (a) "The features serve known users' recurring jobs" becomes "rank tasks well only for users and periods seen in training". (b) The shortcut is described as explaining KAAM's strongest term and one source of random-split scores; LightGBM's lead without the dominant users is attributed to other memorised structure only as a candidate (duplicates). (c) The mechanism paragraph now gives the other-task base rates per fold and says where the one-core effect transfers (folds 0 and 1) and where it reverses (folds 2 and 3). (d) The one-core claim now rests on the training data (with User 1 in training, one-core tasks fail at 75.1% to 82.6% against 19.0% to 20.6%; without User 1, at 19.5% against 19.5% and 20.6% against 17.9%), and the text explains that a centred curve value also reflects the whole curve shape, citing fold 4's +0.82. |
| M10 | Interpretability claims rest on unstable curves | **fixed** (text); bands **open** | The protocol's grid correlation (median 0.75, r = 0.04 for plan_mem) is now primary, and the data-weighted one (0.85) is labelled exploratory. Both versions say the curves are a complete account of each fitted model but not a stable account of the data (time-split plan_cpu curve correlates 0.22 with the canonical one), and that exact SHAP holds for any additive model, so it does not favour the KAN form. The IEEE shape figure now shows the three request and three telemetry features with the largest swing, a stated rule. Seed bands are in the full version's curve figure; split bands are **open**. |
| M11 | "Pre-registered" overstates the safeguards | **fixed** (text); deposit **open** | "Pre-registered" is removed. Both versions call the protocol self-registered, with its hash stored at every test access, and label the original-sample results non-confirmatory. `paper/` is committed with this change. Depositing the protocol with a timestamped registry (OSF, Zenodo) is left to the author. |
| M12 | Uncited premise; missing simple baselines | **partly fixed** | "Usually evaluated on random splits" becomes "often", and the unsupported claim that random splits are the protocol of prior work is removed. An exact request lookup is now scored on the duplicated test tasks (M5); a kNN baseline over all test tasks and a standard P-spline GAM are **open**. The additive EBM is the existing GAM peer. |

### Minor issues

| # | Status | Note |
|---|---|---|
| 1 | fixed | Holm family sizes corrected. |
| 2 | fixed | Abstracts now say 539,816 tasks (47.3%) pass the filters. |
| 3 | fixed | Attrition now adds up: 306,674 (no window) + 43,537 (machine without telemetry) + 251,776 (missing value) + 32 (other) = 602,019. |
| 4 | fixed | The canonical split, the User numbering, the case-selection rule (fixed in the first version: most confident true positive and true negative, highest-risk false negative and false positive at 0.5), the grid and data correlations, seed 0 and the mean training task are defined. |
| 5 | fixed | The two MLP architectures are named with their parameter counts. |
| 6 | fixed | ECE bins stated; PR-AUC and Brier pointed to the full version; ECE outside random splits is in the full version's calibration paragraph. |
| 7 | fixed | The one-core text now explains the two centrings (grid mean in Fig. 3; mean training task in the case audit) and relies on training-data rates. |
| 8 | partly fixed | Table notes now define the abbreviations and the bootstrap units and correct the EBM statement. Dispersion and a spanning header for the time columns were not added to the 6-page table; the fold spread is in the text. |
| 9 | partly fixed | Fold sizes and user counts are in the text and in the full version's fold table. The figure itself is unchanged. |
| 10 | fixed | Fig. 3 caption gives the centring window and seed; units kept as raw request units in Fig. 4, with "100 = 1 core" axis labels. |
| 11 | fixed | Fig. 5 caption defines pNN and the baseline logit. |
| 12 | fixed | Retraining advice removed. |
| 13 | fixed | §V now says four changes were made and which two affect the comparisons. |
| 14 | fixed | A footnote gives the repository URL, which holds the first version, the full version, the protocol and the results. |
| 15 | fixed | Key settings (learning rates, epoch cap, grids, fold seed) added to the IEEE model and protocol sections. |
| 16 | fixed | The full version's limitations say the trace does not distinguish normal completion from stopped tasks within `Terminated`. |
| 17 | fixed | Abstracts say the EBMs were fitted on the earlier sample. |
| 18 | fixed | Vague phrase rewritten. |
| 19 | fixed | "21st", "English" corrected; scikit-learn cited (Pedregosa et al., 2011, verified). The hyphen in Kolmogorov-Arnold is kept, because the author's style rules exclude en dashes. |
| 20 | declined | The Acknowledgment's wording, in the first person, is prescribed by the author and is kept verbatim; the body uses "we". |
| 21 | fixed | The IEEE shape figure's feature choice is now a stated rule, and the text no longer claims close agreement for the request features. |

### Disclosure on test access

All test-label reads of the protocol happened after every model was fitted (the claims audit checks this). The
reviewer's direct reads, the partial reads of the stopped post hoc run and the reads of the completed post hoc run
(2026-11-24) are in `results/test_access_log.jsonl`; no model, setting or protocol entry changed because of any of them.

