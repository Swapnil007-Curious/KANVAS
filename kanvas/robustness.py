"""Phase 7: does KANVAS generalize to users it has never seen?

Every AUC reported so far comes from a random task-level split, in which one
user's tasks can sit on both sides of the split. Phase 6's case audit found
that the model's largest feature is substantially one user's recurring,
repeatedly failing job, so a random split may be measuring "recognise a known
job" as much as "predict failure". This module measures that directly, with a
split grouped by submitting user: no user's tasks appear in more than one of
train, validation and test.

Three rules this module follows, because the answer is only worth having if
they hold:

- **``user`` is a split key, never a feature.** It is recovered from
  ``pai_job_table`` for grouping and for the limitations statistics, it never
  joins ``FEATURE_NAMES``, and no function here writes it to a file.
- **The partition is chosen on label balance alone**, by the fixed criterion
  in ``choose_partition``, before any model is trained on any candidate.
- **Nothing is re-tuned.** Every model uses the Phase 3/4 settings through the
  same functions the earlier phases used, so the only thing that changes
  against Phase 5's Table 1 is how the split is drawn.

A hard grouping constraint cannot also balance labels exactly: a user's tasks
move together, so a user holding a large share of the failures fixes much of
the label balance by itself. ``choose_partition`` reports how far the best
attempt lands from the dataset's failure rate instead of presenting it as
balanced.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .baselines import count_parameters, lr_predict_proba, paired_bootstrap_auc_diff, tune_logistic_regression, tune_mlp
from .stability import (CANONICAL_INIT_SEED, LR_C, MAX_EPOCHS, MLP_HIDDEN, MLP_LR, PATIENCE, THRESHOLD, phi_curve,
                        train_canonical_kaam)
from .training import classification_metrics, predict_proba

#: 20 partition attempts, seeded in advance; the choice among them is label balance only.
PARTITION_SEEDS = list(range(20))
#: Step 4 trains KAAM three times on the chosen partition, varying only the initialization.
INIT_SEEDS = [0, 1, 2]
#: Target row shares of the grouped split, and the spec's balance gate.
PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
BALANCE_TOLERANCE = 0.05
ROLES = ("train", "val", "test")

SURFACE, INK, INK_2, MUTED, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
#: Categorical slots 1 and 2 of the chart palette: random task-level split, user-grouped split.
RANDOM_COLOR, GROUPED_COLOR = "#2a78d6", "#eb6834"


# ------------------------------------------------------------------ Step 1
def load_user_ids(raw_dir, index, verbose=True):
    """Recover each task's submitting user from ``pai_job_table``, for grouping only.

    ``pai_job_table`` holds one row per job (the trace already stores ``user``
    as an opaque hash); the feature matrix is indexed by (job_name, task_name),
    so the join key is the job name.

    Args:
        raw_dir: Directory holding pai_job_table.csv.
        index: The feature matrix's (job_name, task_name) MultiIndex.
        verbose: Print the join report.

    Returns:
        ``(user, report)``: a Series of user id aligned to ``index`` (NaN where
        the job is missing from pai_job_table), and a dict with the task count,
        the matched count, the match rate and the number of distinct users.
    """
    jobs = pd.read_csv(os.path.join(raw_dir, "pai_job_table.csv"), usecols=["job_name", "user"])
    lookup = jobs.drop_duplicates("job_name").set_index("job_name")["user"]
    user = pd.Series(lookup.reindex(index.get_level_values("job_name")).to_numpy(), index=index, name="user")
    matched = user.notna()
    report = {"tasks": len(user), "matched": int(matched.sum()), "unmatched": int((~matched).sum()),
              "match rate": float(matched.mean()), "distinct users": int(user.nunique())}
    if verbose:
        print(f"pai_job_table: {len(jobs):,} rows, {lookup.size:,} distinct jobs")
        print(f"join on job_name: {report['matched']:,} of {report['tasks']:,} tasks matched a user "
              f"({report['match rate']:.2%}), {report['unmatched']:,} unmatched, "
              f"{report['distinct users']:,} distinct users")
    return user, report


# ------------------------------------------------------------------ Step 2
def user_statistics(user, y):
    """Per-user task and failure counts, with anonymous labels for display.

    The user ids are opaque hashes in the trace and are still not printed: each
    user gets a rank label ("user 1" is the largest by task count). The real id
    stays in the returned frame's ``user`` column for grouping code, and never
    goes into an exported file.

    Returns:
        DataFrame with one row per user, sorted by task count, with the task
        count, failure count, failure rate, and shares of the dataset's tasks
        and failures.
    """
    table = y.groupby(user.to_numpy(), sort=False).agg(tasks="size", failures="sum", failure_rate="mean")
    table = table.sort_values(["tasks", "failures"], ascending=False)
    table.insert(0, "user", table.index)
    table.index = [f"user {rank}" for rank in range(1, len(table) + 1)]
    table["share of tasks"] = table["tasks"] / len(y)
    table["share of failures"] = table["failures"] / y.sum()
    return table


def dominant_user_summary(stats):
    """The single largest user's exact shares, as the limitations section needs them."""
    top = stats.iloc[0]
    by_failures = stats.sort_values("failures", ascending=False).iloc[0]
    return {"largest by tasks": stats.index[0], "tasks": int(top["tasks"]),
            "share of tasks": float(top["share of tasks"]), "failures": int(top["failures"]),
            "share of failures": float(top["share of failures"]), "failure rate": float(top["failure_rate"]),
            "also largest by failures": bool(stats.index[0] == by_failures.name)}


# ------------------------------------------------------------------ Step 3
def _assign(sizes, quotas, rng, max_overshoot=0.10):
    """One randomized greedy group partition: each user joins the role it fills least badly.

    Each user is assigned whole, to the role whose row quota its tasks leave
    fullest-but-still-within ``max_overshoot``; where no role can take it
    without overshooting, the least-overshooting role wins.

    Users too large for the smallest quota are placed first, largest first.
    That placement is forced rather than chosen: a user holding a third of the
    rows cannot sit in a 15% part at all. Doing it first is what keeps the row
    shares near 70/15/15, because a huge user arriving late would have to
    overfill whichever part it landed in. Every other user is taken in random
    order, and that order is what varies between attempts.

    Args:
        sizes: Task count per user, descending.
        quotas: Target row count per role.
        rng: Source of the random order.
        max_overshoot: How far over quota a role may go and still be preferred.
    """
    smallest_quota = min(quotas[role] for role in ROLES)
    forced = [position for position, size in enumerate(sizes) if size > smallest_quota]
    rest = [position for position in range(len(sizes)) if position not in set(forced)]
    order = list(forced) + [rest[i] for i in rng.permutation(len(rest))]

    filled = dict.fromkeys(ROLES, 0)
    assignment = {}
    for position in order:
        name, size = sizes.index[position], int(sizes.iloc[position])
        fits = [role for role in ROLES if filled[role] + size <= quotas[role] * (1 + max_overshoot)]
        role = min(fits or ROLES, key=lambda r: (filled[r] + size) / quotas[r])
        assignment[name] = role
        filled[role] += size
    return assignment


def grouped_partitions(user, y, seeds=PARTITION_SEEDS, proportions=None):
    """Build one candidate user-grouped partition per seed, and score each on label balance only.

    Every candidate keeps each user's tasks together. The score is the largest
    absolute gap between a part's failure rate and the dataset's, which is the
    criterion ``choose_partition`` minimizes; no model is trained here.

    Returns:
        ``(attempts, assignments)``: a DataFrame with one row per seed (row
        shares, failure rate and task count per part, and the worst gap), and a
        dict of seed -> {user: role}.
    """
    proportions = PROPORTIONS if proportions is None else proportions
    sizes = y.groupby(user.to_numpy(), sort=False).size().sort_values(ascending=False)
    quotas = {role: max(1.0, share * len(y)) for role, share in proportions.items()}
    overall = float(y.mean())

    attempts, assignments = {}, {}
    for seed in seeds:
        assignment = _assign(sizes, quotas, np.random.default_rng(seed))
        role = pd.Series(user.map(assignment).to_numpy(), index=y.index)
        row = {}
        for part in ROLES:
            part_y = y[role == part]
            row[f"{part} tasks"] = len(part_y)
            row[f"{part} share"] = len(part_y) / len(y)
            row[f"{part} failure rate"] = float(part_y.mean()) if len(part_y) else np.nan
        row["worst failure-rate gap"] = max(abs(row[f"{part} failure rate"] - overall) for part in ROLES)
        row["worst share gap"] = max(abs(row[f"{part} share"] - proportions[part]) for part in ROLES)
        row["users in test"] = int(pd.Series(list(assignment.values())).eq("test").sum())
        attempts[seed] = row
        assignments[seed] = assignment
    return pd.DataFrame(attempts).T.infer_objects().rename_axis("partition seed"), assignments


def choose_partition(attempts, assignments, user, tolerance=BALANCE_TOLERANCE):
    """Pick the candidate whose three failure rates sit closest to the dataset's.

    The criterion is fixed here and uses label balance only: the smallest
    "worst failure-rate gap" wins, ties going to the lower seed. Nothing about
    model performance enters, and no model has been trained at this point.

    Returns:
        ``(seed, role_of_task, balanced)``: the chosen seed, a Series mapping
        each task to "train"/"val"/"test", and whether every part is within
        ``tolerance`` of the dataset failure rate.
    """
    seed = attempts["worst failure-rate gap"].idxmin()
    role = pd.Series(user.map(assignments[seed]).to_numpy(), index=user.index, name="role")
    return seed, role, bool(attempts.loc[seed, "worst failure-rate gap"] <= tolerance)


def split_by_role(X, y, role):
    """``(X_train, X_val, X_test, y_train, y_val, y_test)`` for a task -> role mapping."""
    parts = [role.index[role == part] for part in ROLES]
    if sum(len(part) for part in parts) != len(X):
        raise ValueError("every task must be assigned exactly one role")
    return tuple(X.loc[part] for part in parts) + tuple(y.loc[part] for part in parts)


def split_report(splits, user, y_all):
    """Task counts, row shares, failure rates and user counts of a split, for the notebook."""
    X_tr, X_va, X_te, y_tr, y_va, y_te = splits
    rows = {}
    for part, (X_part, y_part) in zip(ROLES, ((X_tr, y_tr), (X_va, y_va), (X_te, y_te))):
        rows[part] = {"tasks": len(X_part), "share": len(X_part) / len(y_all), "failures": int(y_part.sum()),
                      "failure rate": float(y_part.mean()), "users": int(user.loc[X_part.index].nunique()),
                      "gap vs dataset": float(y_part.mean() - y_all.mean())}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("part")


# ------------------------------------------------------------------ Steps 4-5
def train_kaam_seeds(splits, seeds=INIT_SEEDS, threshold=THRESHOLD, verbose=True):
    """Train KAAM once per initialization seed on one split, with the fixed Phase 3/4 settings.

    ``train_canonical_kaam`` is the same entry point Phases 5 and 6 used, so
    lambda, patience, epoch cap, learning rate, knots and spline order are
    exactly Phase 3's, and the input scaling is refitted on this split's
    training tasks alone.

    Returns:
        ``(table, fitted)``: one row per seed, and a dict of seed ->
        {"model", "result", "p_test"}.
    """
    X_tr, X_va, X_te, y_tr, y_va, y_te = splits
    rows, fitted = {}, {}
    for seed in seeds:
        model, result, seconds = train_canonical_kaam(X_tr, y_tr, X_va, y_va, seed)
        p_test = pd.Series(predict_proba(model, X_te, result.feature_ranges), index=X_te.index)
        m = classification_metrics(y_te.to_numpy(), p_test.to_numpy(), threshold)
        rows[seed] = {"test AUC": m["auc"], "test accuracy": m["accuracy"], "test F1": m["f1"],
                      "best epoch": result.best_epoch, "val BCE": result.best_val_loss, "seconds": seconds}
        fitted[seed] = {"model": model, "result": result, "p_test": p_test}
        if verbose:
            print(f"  init seed {seed}: test AUC {m['auc']:.4f}  acc {m['accuracy']:.4f}  F1 {m['f1']:.4f}  "
                  f"(best epoch {result.best_epoch}, {seconds:.0f}s)")
    return pd.DataFrame(rows).T.infer_objects().rename_axis("init seed"), fitted


def run_baselines(splits, threshold=THRESHOLD, verbose=True):
    """Logistic regression and the MLP on the same split, at their fixed Phase 4 settings.

    Both come from ``baselines.py`` unchanged, called with single-element grids
    so the code path is Phase 4's and Phase 5's minus the search: C = 0.1 for
    logistic regression, hidden (8, 6) at learning rate 0.03 for the MLP.

    Returns:
        ``(table, fitted)``: one row per baseline, and a dict of name ->
        {"model", "feature_ranges", "p_test"}.
    """
    X_tr, X_va, X_te, y_tr, y_va, y_te = splits
    rows, fitted = {}, {}

    lr_model, lr_ranges, lr_seconds, _ = tune_logistic_regression(X_tr, y_tr, X_va, y_va, c_grid=[LR_C])
    p_lr = pd.Series(lr_predict_proba(lr_model, X_te, lr_ranges), index=X_te.index)
    fitted["LR"] = {"model": lr_model, "feature_ranges": lr_ranges, "p_test": p_lr}
    rows["LR"] = {**_metrics(y_te, p_lr, threshold), "parameters": lr_model.coef_.size + lr_model.intercept_.size,
                  "seconds": lr_seconds}

    mlp, mlp_result, mlp_seconds, _ = tune_mlp(X_tr, y_tr, X_va, y_va, hidden_candidates=[MLP_HIDDEN],
                                               lr_candidates=[MLP_LR], patience=PATIENCE, max_epochs=MAX_EPOCHS)
    p_mlp = pd.Series(predict_proba(mlp, X_te, mlp_result.feature_ranges), index=X_te.index)
    fitted["MLP"] = {"model": mlp, "feature_ranges": mlp_result.feature_ranges, "p_test": p_mlp}
    rows["MLP"] = {**_metrics(y_te, p_mlp, threshold), "parameters": count_parameters(mlp), "seconds": mlp_seconds}

    if verbose:
        for name, row in rows.items():
            print(f"  {name:<4s}: test AUC {row['test AUC']:.4f}  acc {row['test accuracy']:.4f}  "
                  f"F1 {row['test F1']:.4f}  ({row['parameters']} parameters, {row['seconds']:.2f}s)")
    return pd.DataFrame(rows).T.infer_objects().rename_axis("model"), fitted


def _metrics(y_test, p_test, threshold):
    m = classification_metrics(y_test.to_numpy(), p_test.to_numpy(), threshold)
    return {"test AUC": m["auc"], "test accuracy": m["accuracy"], "test F1": m["f1"]}


# ------------------------------------------------------------------ Step 6
def comparison_table(random_split_metrics, kaam_table, baseline_table):
    """The six-row comparison: three models under the random task-level split and the user-grouped split.

    Args:
        random_split_metrics: Dict of model name -> {"test AUC", "test accuracy"}
            read from the numbers the earlier phase exported, not retrained.
        kaam_table: ``train_kaam_seeds`` output; its mean and standard
            deviation across seeds become the grouped-split KAAM row.
        baseline_table: ``run_baselines`` output.

    Returns:
        DataFrame with a (split, model) MultiIndex: exactly two split types by
        three models.
    """
    rows = {}
    for model, metrics in random_split_metrics.items():
        rows[("random task-level split", model)] = {"test AUC": metrics["test AUC"], "AUC std": np.nan,
                                                    "test accuracy": metrics["test accuracy"],
                                                    "runs": metrics.get("runs", 1)}
    rows[("user-grouped split", "KAAM")] = {
        "test AUC": kaam_table["test AUC"].mean(), "AUC std": kaam_table["test AUC"].std(ddof=1),
        "test accuracy": kaam_table["test accuracy"].mean(), "runs": len(kaam_table)}
    for model in ("LR", "MLP"):
        rows[("user-grouped split", model)] = {"test AUC": baseline_table.loc[model, "test AUC"], "AUC std": np.nan,
                                               "test accuracy": baseline_table.loc[model, "test accuracy"], "runs": 1}
    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=["split", "model"])
    order = [(split, model) for split in ("random task-level split", "user-grouped split")
             for model in ("KAAM", "LR", "MLP")]
    return table.loc[order]


def auc_drop(comparison):
    """Each model's change in test AUC and accuracy, grouped split minus random split."""
    random_split, grouped = comparison.loc["random task-level split"], comparison.loc["user-grouped split"]
    return pd.DataFrame({
        "random split AUC": random_split["test AUC"], "grouped split AUC": grouped["test AUC"],
        "AUC change": grouped["test AUC"] - random_split["test AUC"],
        "random split accuracy": random_split["test accuracy"], "grouped split accuracy": grouped["test accuracy"],
        "accuracy change": grouped["test accuracy"] - random_split["test accuracy"],
    }).rename_axis("model")


def paired_vs_canonical(grouped_fitted, canonical_model, canonical_ranges, X_all, y_all, canonical_test_index,
                        n_boot=2000, seed=0):
    """Paired bootstrap on the tasks that are test tasks under *both* splits.

    The two models were trained on different task sets, so a paired comparison
    needs rows neither model was trained on: the intersection of the canonical
    test set with the grouped test set. Both models score those same tasks
    (each from its own training-fitted scaling), and the same 2,000 resamples
    are scored for both, as in Phases 4-6.

    Returns:
        DataFrame with one row per grouped-split seed: the shared-task count,
        both AUCs on those tasks, the difference and its 95% interval.
    """
    rows = {}
    p_canonical_all = None
    for init_seed, fit in grouped_fitted.items():
        shared = canonical_test_index[canonical_test_index.isin(fit["p_test"].index)]
        if p_canonical_all is None:
            p_canonical_all = pd.Series(predict_proba(canonical_model, X_all.loc[shared], canonical_ranges),
                                        index=shared)
        y = y_all.loc[shared].to_numpy()
        grouped_p = fit["p_test"].loc[shared].to_numpy()
        canonical_p = p_canonical_all.loc[shared].to_numpy()
        lo, hi = paired_bootstrap_auc_diff(y, grouped_p, canonical_p, n_boot=n_boot, seed=seed)
        rows[init_seed] = {"shared test tasks": len(shared), "shared failures": int(y.sum()),
                           "grouped AUC": roc_auc_score(y, grouped_p), "canonical AUC": roc_auc_score(y, canonical_p),
                           "AUC difference": roc_auc_score(y, grouped_p) - roc_auc_score(y, canonical_p),
                           "95% CI low": lo, "95% CI high": hi, "CI includes 0": bool(lo <= 0 <= hi)}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("init seed")


def signal_by_part(splits, feature="plan_cpu", value=100.0, direction=-1.0):
    """Is the dominant feature's relationship to failure the same in each part of the split?

    A grouped split moves whole users, so a pattern that holds for the training
    users need not hold for the held-out ones. Two views per part: the failure
    rate of the tasks at ``value`` against the rest, and the AUC of scoring by
    ``direction * feature`` alone, where ``direction`` is the sign the model
    learned (for ``plan_cpu``, smaller means riskier, so -1).

    Returns:
        DataFrame with one row per part.
    """
    X_tr, X_va, X_te, y_tr, y_va, y_te = splits
    rows = {}
    for part, (X_part, y_part) in zip(ROLES, ((X_tr, y_tr), (X_va, y_va), (X_te, y_te))):
        at_value = X_part[feature] == value
        rows[part] = {
            f"tasks at {feature}={value:g}": int(at_value.sum()),
            "their failure rate": float(y_part[at_value].mean()) if at_value.any() else np.nan,
            "every other task's failure rate": float(y_part[~at_value].mean()) if (~at_value).any() else np.nan,
            "part failure rate": float(y_part.mean()),
            f"AUC of {direction:+g} x {feature} alone": roc_auc_score(y_part, direction * X_part[feature]),
        }
    return pd.DataFrame(rows).T.infer_objects().rename_axis("part")


def training_trajectory(fitted):
    """Where each run's validation loss bottomed, and where it ended.

    On a grouped split the validation tasks belong to users the model never
    trains on, so this says directly whether more training would have helped
    or hurt generalization to new users.
    """
    rows = {}
    for seed, fit in fitted.items():
        history = fit["result"].history["val_loss"]
        rows[seed] = {"epochs run": len(history), "best epoch": fit["result"].best_epoch,
                      "val BCE at epoch 1": history[0], "best val BCE": min(history),
                      "val BCE at the last epoch run": history[-1],
                      "early stopped": fit["result"].early_stopped}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("init seed")


def partition_robustness(X, y, attempts, assignments, user, chosen, n_alternatives=3,
                         init_seed=CANONICAL_INIT_SEED, threshold=THRESHOLD, verbose=True):
    """Train one KAAM on the next-best-balanced partitions, to see whether the result is partition-specific.

    This runs **after** Step 3 has already chosen a partition by label balance,
    and changes nothing about that choice: it asks only whether a different
    grouped partition would have told a different story.

    Returns:
        DataFrame with one row per partition (the chosen one first).
    """
    order = attempts["worst failure-rate gap"].sort_values().index
    seeds = [chosen] + [seed for seed in order if seed != chosen][:n_alternatives]
    rows = {}
    for seed in seeds:
        role = pd.Series(user.map(assignments[seed]).to_numpy(), index=user.index)
        splits = split_by_role(X, y, role)
        model, result, _ = train_canonical_kaam(splits[0], splits[3], splits[1], splits[4], init_seed)
        p_test = predict_proba(model, splits[2], result.feature_ranges)
        m = classification_metrics(splits[5].to_numpy(), p_test, threshold)
        rows[seed] = {"chosen in step 3": seed == chosen, "test tasks": len(splits[2]),
                      "test failure rate": float(splits[5].mean()), "test AUC": m["auc"],
                      "test accuracy": m["accuracy"], "test F1": m["f1"], "best epoch": result.best_epoch}
        if verbose:
            print(f"  partition seed {seed}{' (chosen)' if seed == chosen else ''}: test AUC {m['auc']:.4f}  "
                  f"acc {m['accuracy']:.4f}  F1 {m['f1']:.4f}  (best epoch {result.best_epoch})")
    return pd.DataFrame(rows).T.infer_objects().rename_axis("partition seed")


# ------------------------------------------------------------------ Step 7
def curve_overlay(models, name, X_reference, lower=0.05, upper=0.95, num_points=200):
    """One feature's phi curve from several models, on one shared x-range in raw units.

    Every curve is evaluated over the same window (``X_reference``'s
    ``lower``-``upper`` quantiles) and centered on its own mean there, so the
    comparison is of shape, not of an arbitrary vertical offset.

    Args:
        models: Dict of label -> ``(model, feature_ranges)``.
        name: Feature to plot.
        X_reference: Frame whose quantiles set the window (the canonical training tasks).

    Returns:
        ``(curves, summary)``: long-format (label, x_value, phi_value), and a
        per-label summary with the curve's range, its largest gap from the
        first label's curve, and the correlation of the two shapes.
    """
    x_lo, x_hi = X_reference[name].quantile([lower, upper])
    curves, values = [], {}
    for label, (model, ranges) in models.items():
        grid, phi = phi_curve(model, name, ranges, float(x_lo), float(x_hi), num_points, center="mean")
        values[label] = phi
        curves.append(pd.DataFrame({"label": label, "feature_name": name, "x_value": grid, "phi_value": phi}))
    reference = next(iter(values))
    summary = {}
    for label, phi in values.items():
        gap = np.abs(phi - values[reference])
        summary[label] = {"curve range": float(phi.max() - phi.min()),
                          f"max |dphi| vs {reference}": float(gap.max()),
                          f"mean |dphi| vs {reference}": float(gap.mean()),
                          f"shape correlation vs {reference}": float(np.corrcoef(phi, values[reference])[0, 1])}
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary).T.rename_axis("model")


def plot_split_comparison(comparison, threshold_auc=0.5):
    """Test AUC of the three models under both splits, with the chance line marked."""
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(comparison.index.get_level_values("model")))
    splits = [("random task-level split", RANDOM_COLOR), ("user-grouped split", GROUPED_COLOR)]
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (split, color) in zip((-width / 2, width / 2), splits):
        values = [comparison.loc[(split, model), "test AUC"] for model in models]
        errors = [comparison.loc[(split, model), "AUC std"] for model in models]
        errors = [0.0 if pd.isna(e) else e for e in errors]
        bars = ax.bar(x + offset, values, width, color=color, edgecolor=SURFACE, linewidth=1, label=split, zorder=3)
        ax.errorbar(x + offset, values, yerr=errors, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4, zorder=4)
        for bar, value, error in zip(bars, values, errors):
            label = f"{value:.3f}" + (f" ± {error:.3f}" if error else "")
            ax.text(bar.get_x() + bar.get_width() / 2, value + error + 0.012, label, ha="center", va="bottom",
                    fontsize=8.5, color=INK_2)
    ax.axhline(threshold_auc, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(len(models) - 0.45, threshold_auc + 0.008, "chance (AUC 0.5)", ha="right", va="bottom", fontsize=8,
            color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("test AUC", fontsize=9)
    ax.set_ylim(0, max(0.95, comparison["test AUC"].max() + 0.2))  # headroom so the legend clears the bar labels
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_curve_overlay(curves, name, colors=None, title=None, subtitle=None, rug=None):
    """The same feature's curve from two models, on one pair of axes."""
    import matplotlib.pyplot as plt

    colors = colors or {}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for label, part in curves.groupby("label", sort=False):
        ax.plot(part["x_value"], part["phi_value"], color=colors.get(label, MUTED), lw=2.4, label=label)
    if rug is not None:
        ax.plot(rug, np.full(len(rug), 0.02), "|", color=MUTED, alpha=0.25, ms=6,
                transform=ax.get_xaxis_transform())
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_xlabel(f"{name} (raw units)", fontsize=9)
    ax.set_ylabel("φ contribution to failure log-odds\n(centered on this window)", fontsize=9)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=12, pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    fig.tight_layout()
    return fig
