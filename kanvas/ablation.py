"""Phase 6, Part A: is the 1-hour telemetry lookback load-bearing or arbitrary?

The 1-hour window was chosen in Phase 2, when the original worker-keyed
telemetry join removed nearly every failure and had to be replaced by a
machine-keyed lookback. The width was a reasonable default that was flagged
for an ablation and never tested. This module runs that ablation.

Only ``lookback_hours`` moves. ``build_kanvas_dataset`` is called unchanged at
Phase 5's canonical sample (40,000 tasks, ``random_state=42``), and every model
is trained through ``kanvas.stability.train_canonical_kaam`` with the fixed
Phase 3/4 settings, so a difference between widths comes from the window and
from nothing else.

A different window does not only change feature values; it changes which tasks
survive. A task reaches the final dataset only if some other job's worker ended
on its machine inside the window and every window average is defined, so the
five datasets are five different row sets. The windows are nested, so the row
sets are nested too: every task kept at 0.5h is kept at 1h, and so on up to 6h.

Because the row sets differ, running ``train_val_test_split`` on each one would
draw five unrelated test sets, and Phase 5 showed the split alone moves test
AUC with a standard deviation of 0.0104. So each task's role is fixed once, by
task identity, for every width (``anchored_splits``): the canonical 1-hour
tasks keep Phase 5's split exactly, and tasks that only exist at other widths
get their own stratified 70/15/15 split. The paired bootstrap against 1 hour
then runs on every test task two widths share.
"""

import contextlib
import io
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .baselines import paired_bootstrap_auc_diff
from .data import FEATURE_NAMES, WINDOW_METRICS, build_kanvas_dataset
from .shap_analysis import kaam_curve_range_importance
from .stability import CANONICAL_INIT_SEED, CANONICAL_SPLIT_STATE, THRESHOLD, phi_curve, train_canonical_kaam
from .training import classification_metrics, predict_proba, train_val_test_split

#: Window widths tested, in hours. 1.0 is the width every earlier phase used.
LOOKBACK_HOURS = [0.5, 1.0, 2.0, 4.0, 6.0]
CANONICAL_LOOKBACK = 1.0

#: Phase 5's canonical build: the first (and only) attempt of build_or_load_large_dataset.
SAMPLE_N_TASKS, SAMPLE_STATE = 40_000, 42

#: The two telemetry features whose curves Step 4 compares across widths.
TELEMETRY_FEATURES = ["avg_cpu_kernel", "avg_net_receive"]
#: Every feature that is a lookback-window average. The other five never depend on the window.
WINDOW_FEATURES = list(WINDOW_METRICS.values())

ROLES = ("train", "val", "test")

SURFACE, INK, INK_2, MUTED, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
#: Width is ordered, so it takes a one-hue ordinal ramp, light (0.5h) to dark (6h).
#: Blue steps 250/350/450/550/650 of the chart palette, validated with --ordinal.
WIDTH_COLORS = {0.5: "#86b6ef", 1.0: "#5598e7", 2.0: "#2a78d6", 4.0: "#1c5cab", 6.0: "#104281"}


# ------------------------------------------------------------------ Steps 1-2
def lookback_paths(out_dir, width):
    """Feature CSV and build-log paths for one width, e.g. ``features_lookback_0.5h.csv``."""
    stem = os.path.join(out_dir, f"features_lookback_{width:g}h")
    return stem + ".csv", stem + "_buildlog.txt"


def build_lookback_datasets(raw_dir, out_dir, widths=LOOKBACK_HOURS, sample_n_tasks=SAMPLE_N_TASKS,
                            random_state=SAMPLE_STATE, verbose=True):
    """One ``build_kanvas_dataset`` call per width, each cached with its full build log.

    ``sample_n_tasks`` and ``random_state`` stay at Phase 5's canonical values,
    so every width starts from the same 40,000 sampled tasks and differs only
    in how much telemetry history each task can see. Every call gets its own
    ``out_path``; the Phase 3 development file is never touched.

    Each build streams the 2 GB instance table, so a width whose CSV and log
    both exist is loaded rather than rebuilt.

    Returns:
        ``(datasets, logs)``: dicts of width -> ``(X, y)`` (indexed by
        job_name, task_name) and width -> the build's printed survival report.
    """
    os.makedirs(out_dir, exist_ok=True)
    datasets, logs = {}, {}
    for width in widths:
        csv_path, log_path = lookback_paths(out_dir, width)
        if os.path.exists(csv_path) and os.path.exists(log_path):
            source = "cached"
        else:
            start = time.perf_counter()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                build_kanvas_dataset(raw_dir, sample_n_tasks=sample_n_tasks, lookback_hours=width,
                                     random_state=random_state, out_path=csv_path, verbose=True)
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(buffer.getvalue())
            source = f"built in {time.perf_counter() - start:.0f}s"
        df = pd.read_csv(csv_path, index_col=["job_name", "task_name"])
        with open(log_path, encoding="utf-8") as fh:
            logs[width] = fh.read()
        datasets[width] = (df[FEATURE_NAMES], df["label"])
        if verbose:
            print(f"  lookback {width:>3g}h: {len(df):>6,} clean rows, failure rate {df['label'].mean():.2%}  ({source})")
    return datasets, logs


def parse_build_log(log_text):
    """The funnel counts ``build_kanvas_dataset`` printed, read back from its log.

    Only Step 3 (telemetry in the window) and Step 6 (NaN in any feature)
    can depend on the window width; every other step is identical at every width.

    Raises:
        ValueError: if a line the funnel needs is missing, rather than
            silently reporting a zero.
    """
    def find(pattern):
        match = re.search(pattern, log_text)
        if match is None:
            raise ValueError(f"build log has no line matching {pattern!r}")
        return match

    def count(text):
        return int(text.replace(",", ""))

    sampled = find(r"stratified sample: ([\d,]+) tasks")
    step3 = find(r"dropped ([\d,]+) tasks with zero machine_metric rows in the window; ([\d,]+) remain "
                 r"\(failure rate ([\d.]+)%\)")
    no_worker = re.search(r"drop ([\d,]+) tasks: no other job's worker ended on the machine in the lookback window",
                          log_text)  # absent when no task was dropped for this reason
    rows = find(r"machine_metric rows averaged per task: median (\d+), 90th pct (\d+), max (\d+)")
    step6 = find(r"dropped ([\d,]+) tasks with any NaN \(failure rate among dropped (?:[\d.]+|nan)%\); "
                 r"([\d,]+) remain \(failure rate ([\d.]+)%\)")
    return {
        "sampled": count(sampled.group(1)),
        "telemetry found": count(step3.group(2)),
        "no worker in window": count(no_worker.group(1)) if no_worker else 0,
        "no telemetry, other reasons": count(step3.group(1)) - (count(no_worker.group(1)) if no_worker else 0),
        "dropped for NaN": count(step6.group(1)),
        "clean rows": count(step6.group(2)),
        "median rows averaged": int(rows.group(1)),
        "p90 rows averaged": int(rows.group(2)),
    }


def coverage_table(datasets, logs=None, canonical=CANONICAL_LOOKBACK):
    """How the window width changes *which tasks survive*, before any model is trained.

    Gained and lost are counted by task identity against the canonical
    width's task set, together with the failure rate of the tasks that moved.
    With ``logs``, the funnel stages that can depend on the width are added.

    Returns:
        DataFrame indexed by width, in the order of ``datasets``.
    """
    base_X, base_y = datasets[canonical]
    rows = {}
    for width, (X, y) in datasets.items():
        gained = X.index[~X.index.isin(base_X.index)]
        lost = base_X.index[~base_X.index.isin(X.index)]
        row = {}
        if logs is not None:
            row.update(parse_build_log(logs[width]))
            if row["clean rows"] != len(X):
                raise ValueError(f"{width:g}h: build log reports {row['clean rows']:,} clean rows, CSV has {len(X):,}")
        row.update({
            "clean rows": len(X), "failure rate": float(y.mean()),
            "gained vs 1h": len(gained), "lost vs 1h": len(lost),
            "failure rate of gained": float(y.loc[gained].mean()) if len(gained) else np.nan,
            "failure rate of lost": float(base_y.loc[lost].mean()) if len(lost) else np.nan,
        })
        rows[width] = row
    return pd.DataFrame(rows).T.infer_objects().rename_axis("lookback hours")


def nesting_report(datasets):
    """Is each narrower window's task set contained in the next wider one's, and do labels agree?"""
    widths = sorted(datasets)
    rows = []
    for narrow, wide in zip(widths, widths[1:]):
        X_n, y_n = datasets[narrow]
        X_w, y_w = datasets[wide]
        inside = X_n.index.isin(X_w.index)
        shared = X_n.index[inside]
        rows.append({"narrower": f"{narrow:g}h", "wider": f"{wide:g}h", "narrower tasks": len(X_n),
                     "also in wider": int(inside.sum()), "nested": bool(inside.all()),
                     "label disagreements": int((y_n.loc[shared] != y_w.loc[shared]).sum())})
    return pd.DataFrame(rows)


def static_feature_changes(datasets, canonical=CANONICAL_LOOKBACK):
    """For the five non-window features, how many shared-task values differ from the canonical width's.

    These features are read straight from the task and machine tables, so
    they can only differ where a width's own 1st/99th-percentile clip bound
    (fitted on its own row set) moved.

    Returns:
        ``(changed, upper)``: per width, the count of shared-task values that
        differ from the canonical width's, and each feature's upper clip bound.
    """
    static = [name for name in FEATURE_NAMES if name not in WINDOW_FEATURES]
    base_X = datasets[canonical][0]
    changed, upper = {}, {}
    for width, (X, _) in datasets.items():
        shared = base_X.index[base_X.index.isin(X.index)]
        changed[width] = (X.loc[shared, static] != base_X.loc[shared, static]).sum()
        upper[width] = X[static].max()
    return (pd.DataFrame(changed).T.rename_axis("lookback hours"),
            pd.DataFrame(upper).T.rename_axis("lookback hours"))


# ------------------------------------------------------------------ Step 3
def anchored_splits(datasets, canonical=CANONICAL_LOOKBACK, split_state=CANONICAL_SPLIT_STATE):
    """Train/validation/test splits in which every task keeps one role at every width.

    - Tasks in the canonical width's dataset keep exactly the split Phase 5
      used (``train_val_test_split(..., random_state=42)`` on that dataset), in
      the same row order, so the canonical arm reproduces the Phase 5 model.
    - Tasks that only other widths keep get their own stratified 70/15/15
      split with the same function and seed, collected in a fixed order
      (widths ascending, each dataset's own row order).

    Each width's split is then its own tasks, filtered by role. The split
    proportions stay 70/15/15, and a task that is a test task at one width is
    a test task at every width where it exists.

    Returns:
        Dict of width -> ``(X_train, X_val, X_test, y_train, y_val, y_test)``.

    Raises:
        ValueError: if a task's label differs between widths, or a task ends
            up with no role or with two.
    """
    X_c, y_c = datasets[canonical]
    _, _, _, c_tr, c_va, c_te = train_val_test_split(X_c, y_c, random_state=split_state)
    roles = {"train": c_tr.index, "val": c_va.index, "test": c_te.index}

    seen, extras = X_c.index, []
    for width in sorted(datasets):
        y = datasets[width][1]
        new = y[~y.index.isin(seen)]
        extras.append(new)
        seen = seen.append(new.index)
    y_extra = pd.concat(extras)
    if len(y_extra):
        _, _, _, e_tr, e_va, e_te = train_val_test_split(y_extra.to_frame(), y_extra, random_state=split_state)
        roles = {"train": roles["train"].append(e_tr.index), "val": roles["val"].append(e_va.index),
                 "test": roles["test"].append(e_te.index)}

    labels = pd.concat([y_c, y_extra])
    splits = {}
    for width, (X, y) in datasets.items():
        if not (y == labels.loc[y.index]).all():
            raise ValueError(f"{width:g}h: a task's label differs from its label at another width")
        parts = [roles[role][roles[role].isin(X.index)] for role in ROLES]
        if sum(len(p) for p in parts) != len(X) or not X.index.isin(parts[0].append(parts[1]).append(parts[2])).all():
            raise ValueError(f"{width:g}h: every task must have exactly one role")
        splits[width] = tuple(X.loc[p] for p in parts) + tuple(y.loc[p] for p in parts)
    return splits


def split_summary(splits):
    """Size, share and failure rate of each part of each width's split."""
    rows = {}
    for width, (X_tr, X_va, X_te, y_tr, y_va, y_te) in splits.items():
        n = len(X_tr) + len(X_va) + len(X_te)
        rows[width] = {"train": len(X_tr), "val": len(X_va), "test": len(X_te),
                       "train share": len(X_tr) / n, "test share": len(X_te) / n,
                       "train failure rate": y_tr.mean(), "val failure rate": y_va.mean(),
                       "test failure rate": y_te.mean()}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("lookback hours")


def train_across_widths(splits, widths=LOOKBACK_HOURS, init_seed=CANONICAL_INIT_SEED, threshold=THRESHOLD,
                        verbose=True):
    """One KAAM per width, trained with the fixed Phase 3/4 settings, scored on that width's test tasks.

    Nothing about training varies between widths: same lambda, patience,
    epoch cap, learning rate, initialization seed and threshold, with the
    input scaling refitted on each width's own training tasks.

    Returns:
        ``(table, fitted)``: test metrics with exactly one row per width, and
        a dict of width -> {"model", "result", "p_test"} where ``p_test`` is a
        Series of test probabilities indexed by task.
    """
    missing = [width for width in widths if width not in splits]
    if missing:
        raise ValueError(f"no split for widths {missing}; the ablation must report every width")
    rows, fitted = {}, {}
    for width in widths:
        X_tr, X_va, X_te, y_tr, y_va, y_te = splits[width]
        model, result, seconds = train_canonical_kaam(X_tr, y_tr, X_va, y_va, init_seed)
        p_test = pd.Series(predict_proba(model, X_te, result.feature_ranges), index=X_te.index)
        m = classification_metrics(y_te.to_numpy(), p_test.to_numpy(), threshold)
        rows[width] = {"train tasks": len(X_tr), "test tasks": len(X_te), "test failures": int(y_te.sum()),
                       "test AUC": m["auc"], "test accuracy": m["accuracy"], "test F1": m["f1"],
                       "best epoch": result.best_epoch, "val BCE": result.best_val_loss, "seconds": seconds}
        fitted[width] = {"model": model, "result": result, "p_test": p_test}
        if verbose:
            print(f"  lookback {width:>3g}h: test AUC {m['auc']:.4f}  acc {m['accuracy']:.4f}  F1 {m['f1']:.4f}  "
                  f"(best epoch {result.best_epoch}, {seconds:.0f}s)")
    return pd.DataFrame(rows).T.infer_objects().rename_axis("lookback hours"), fitted


def paired_vs_canonical(splits, fitted, canonical=CANONICAL_LOOKBACK, n_boot=2000, seed=0):
    """AUC(width) - AUC(1h) on the test tasks both widths share, with a paired-bootstrap 95% CI.

    Both models score the same tasks, each from its own width's features, and
    the same 2,000 resamples of those tasks are scored for both: the method of
    Phases 4 and 5 (``paired_bootstrap_auc_diff``, seed 0). Shared tasks are
    taken in the canonical test order, so the resamples are reproducible.

    Returns:
        DataFrame with one row per width; the canonical row is the reference
        (difference 0, no interval).
    """
    y_ref = splits[canonical][5]
    p_ref = fitted[canonical]["p_test"]
    rows = {}
    for width in fitted:
        p_width = fitted[width]["p_test"]
        shared = y_ref.index[y_ref.index.isin(p_width.index)]
        y = y_ref.loc[shared].to_numpy()
        a, b = p_width.loc[shared].to_numpy(), p_ref.loc[shared].to_numpy()
        auc_a, auc_b = roc_auc_score(y, a), roc_auc_score(y, b)
        if width == canonical:
            lo = hi = includes_zero = np.nan
        else:
            lo, hi = paired_bootstrap_auc_diff(y, a, b, n_boot=n_boot, seed=seed)
            includes_zero = bool(lo <= 0 <= hi)
        rows[width] = {"shared test tasks": len(shared), "shared failures": int(y.sum()),
                       "AUC, this width": auc_a, "AUC, 1h": auc_b, "AUC difference": auc_a - auc_b,
                       "95% CI low": lo, "95% CI high": hi, "CI includes 0": includes_zero}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("lookback hours")


def common_task_scores(fitted, splits, threshold=THRESHOLD):
    """Every width's model scored on one task set: the test tasks that every width keeps.

    The windows are nested, so these are the narrowest width's test tasks.
    Each model still scores them from its own width's features, so the
    scored population is held fixed and only the models differ.

    Returns:
        DataFrame with one row per width: task and failure counts, AUC,
        accuracy and F1 at ``threshold``.
    """
    common = None
    for fit in fitted.values():
        index = fit["p_test"].index
        common = index if common is None else common[common.isin(index)]
    rows = {}
    for width, fit in fitted.items():
        y = splits[width][5].loc[common].to_numpy()
        m = classification_metrics(y, fit["p_test"].loc[common].to_numpy(), threshold)
        rows[width] = {"common test tasks": len(common), "failures": int(y.sum()), "AUC": m["auc"],
                       "accuracy": m["accuracy"], "F1": m["f1"]}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("lookback hours")


def auc_change_decomposition(performance, paired, canonical=CANONICAL_LOOKBACK):
    """Split each width's own-test AUC change against 1h into a model effect and a coverage effect.

    ``own AUC(width) - own AUC(1h)`` has two parts. The **model effect** is
    the paired difference on the test tasks both widths share: same tasks,
    different model. The **coverage effect** is the rest: what changing
    *which* tasks get scored does to the AUC, with no change in model quality.

    Returns:
        DataFrame with one row per width.
    """
    base = performance.loc[canonical, "test AUC"]
    rows = {}
    for width in performance.index:
        total = performance.loc[width, "test AUC"] - base
        model = paired.loc[width, "AUC difference"]
        rows[width] = {"own test AUC": performance.loc[width, "test AUC"], "change vs 1h": total,
                       "model effect (shared tasks)": model, "coverage effect (which tasks)": total - model}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("lookback hours")


def run_lookback_ablation(datasets, widths=LOOKBACK_HOURS, canonical=CANONICAL_LOOKBACK, verbose=True):
    """Steps 3's whole pipeline: anchored splits, one model per width, paired CIs against 1h.

    Returns exactly one result per width in ``widths``, or raises; it never
    returns fewer.

    Returns:
        Dict with "splits", "performance", "fitted" and "paired".
    """
    missing = [width for width in widths if width not in datasets]
    if missing:
        raise ValueError(f"no dataset for widths {missing}; the ablation must report all of {list(widths)}")
    if canonical not in widths:
        raise ValueError(f"the canonical width {canonical:g}h must be one of the widths compared")
    splits = anchored_splits({width: datasets[width] for width in widths}, canonical)
    performance, fitted = train_across_widths(splits, widths, verbose=verbose)
    paired = paired_vs_canonical(splits, fitted, canonical)
    for name, table in (("performance", performance), ("paired", paired)):
        if list(table.index) != list(widths):
            raise RuntimeError(f"{name} has rows {list(table.index)}, expected exactly {list(widths)}")
    return {"splits": splits, "performance": performance, "fitted": fitted, "paired": paired}


def results_table(coverage, performance, paired):
    """The five-width survival and performance table, one row per width."""
    table = pd.DataFrame({
        "clean rows": coverage["clean rows"], "failure rate": coverage["failure rate"],
        "gained vs 1h": coverage["gained vs 1h"], "lost vs 1h": coverage["lost vs 1h"],
        "test tasks": performance["test tasks"], "test AUC": performance["test AUC"],
        "test accuracy": performance["test accuracy"], "test F1": performance["test F1"],
        "shared test tasks": paired["shared test tasks"], "AUC diff vs 1h": paired["AUC difference"],
        "95% CI low": paired["95% CI low"], "95% CI high": paired["95% CI high"],
    })
    return table.rename_axis("lookback hours")


# ------------------------------------------------------------------ Step 4
def shared_window(splits, name, lower=0.05, upper=0.95):
    """The x-range every width's curve is compared over: the overlap of their training p5-p95 windows.

    A longer window averages more telemetry rows, which narrows the spread of
    the averages, so each width's p5-p95 window differs; only their overlap is
    dense with training tasks at every width.
    """
    bounds = [splits[width][0][name].quantile([lower, upper]) for width in splits]
    lo, hi = max(b.iloc[0] for b in bounds), min(b.iloc[1] for b in bounds)
    if hi <= lo:
        raise ValueError(f"{name}: the widths' p{lower * 100:g}-p{upper * 100:g} windows do not overlap")
    return float(lo), float(hi)


def telemetry_curves(fitted, splits, features=TELEMETRY_FEATURES, lower=0.05, upper=0.95, num_points=200):
    """Long-format phi curves for each telemetry feature at every width, in raw units.

    Each curve spans its own width's training p5-p95 window and is shifted
    by its mean over the shared window, so every width's curve is centered on
    the same stretch of x and the vertical offsets compare fairly.
    """
    rows = []
    for name in features:
        s_lo, s_hi = shared_window(splits, name, lower, upper)
        for width in sorted(fitted):
            model, ranges = fitted[width]["model"], fitted[width]["result"].feature_ranges
            _, on_shared = phi_curve(model, name, ranges, s_lo, s_hi, num_points)
            x_lo, x_hi = splits[width][0][name].quantile([lower, upper])
            grid, phi = phi_curve(model, name, ranges, x_lo, x_hi, num_points, center=float(on_shared.mean()))
            rows.append(pd.DataFrame({"lookback_hours": width, "feature_name": name, "x_value": grid,
                                      "phi_value": phi}))
    return pd.concat(rows, ignore_index=True)


def curve_agreement(fitted, splits, features=TELEMETRY_FEATURES, canonical=CANONICAL_LOOKBACK, lower=0.05,
                    upper=0.95, num_points=200):
    """How far each width's curve departs from the 1-hour curve, over the shared window.

    Both curves are evaluated on the same grid and centered on their own mean
    there. ``max |dphi| / 1h range`` puts the largest departure on the scale
    of the canonical curve's own movement; ``shape correlation`` is the
    Pearson correlation of the two centered curves (1 = identical shape).

    Returns:
        DataFrame indexed by (feature, width), non-canonical widths only.
    """
    rows = {}
    for name in features:
        s_lo, s_hi = shared_window(splits, name, lower, upper)
        base_fit = fitted[canonical]
        _, base = phi_curve(base_fit["model"], name, base_fit["result"].feature_ranges, s_lo, s_hi, num_points,
                            center="mean")
        base_range = float(base.max() - base.min())
        for width in sorted(fitted):
            if width == canonical:
                continue
            fit = fitted[width]
            _, other = phi_curve(fit["model"], name, fit["result"].feature_ranges, s_lo, s_hi, num_points,
                                 center="mean")
            gap = np.abs(other - base)
            rows[(name, width)] = {"shared window low": s_lo, "shared window high": s_hi,
                                   "1h curve range": base_range, "this curve range": float(other.max() - other.min()),
                                   "max |dphi|": float(gap.max()), "mean |dphi|": float(gap.mean()),
                                   "max |dphi| / 1h range": float(gap.max() / base_range) if base_range else np.nan,
                                   "shape correlation": float(np.corrcoef(other, base)[0, 1])}
    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=["feature", "lookback hours"])
    return table


def importance_by_width(fitted, splits, canonical=CANONICAL_LOOKBACK):
    """KAAM's built-in importance (phi range over each width's training p5-p95) for all ten features.

    The same definition Phases 4 and 5 ranked features by, so this shows
    whether the window width moves the headline ranking, not just two curves.

    Returns:
        ``(shares, ranks, rho)``: importance shares (features x widths), ranks
        (1 = most important), and the Spearman correlation of each width's
        ranking with the canonical one.
    """
    shares = {}
    for width in sorted(fitted):
        importance = kaam_curve_range_importance(fitted[width]["model"], splits[width][0],
                                                 fitted[width]["result"].feature_ranges)
        shares[width] = importance / importance.sum()
    shares = pd.DataFrame(shares)
    ranks = shares.rank(ascending=False, method="min").astype(int)
    rho = pd.Series({width: spearmanr(shares[width], shares[canonical])[0] for width in shares},
                    name="Spearman rho vs 1h")
    return shares, ranks, rho


# ------------------------------------------------------------------ plots
def plot_telemetry_curves(curves, splits, features=TELEMETRY_FEATURES, canonical=CANONICAL_LOOKBACK,
                          lower=0.05, upper=0.95):
    """One panel per telemetry feature, one line per width on a light-to-dark ramp; 1h drawn heavier.

    The panels share one log-odds scale, so a feature whose whole curve spans
    a few hundredths of a log-odd looks as small as it is.
    """
    import matplotlib.pyplot as plt

    units = {"avg_cpu_kernel": ("avg_cpu_kernel: mean kernel CPU on the machine in the window (%)", 1.0),
             "avg_net_receive": ("avg_net_receive: mean network receive on the machine in the window (MB/s)", 1e6)}
    fig, axes = plt.subplots(1, len(features), figsize=(13, 4.8), squeeze=False, sharey=True)
    for ax, name in zip(axes[0], features):
        label, divisor = units.get(name, (name, 1.0))
        s_lo, s_hi = shared_window(splits, name, lower, upper)
        ax.axvspan(s_lo / divisor, s_hi / divisor, color=BASELINE, alpha=0.18, lw=0)
        sub = curves[curves["feature_name"] == name]
        for width, part in sub.groupby("lookback_hours"):
            heavy = width == canonical
            ax.plot(part["x_value"] / divisor, part["phi_value"], color=WIDTH_COLORS.get(width, MUTED),
                    lw=3.2 if heavy else 2.0, zorder=3 if heavy else 2,
                    label=f"{width:g}h" + ("  (canonical)" if heavy else ""))
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(name, loc="left", fontsize=11)
    axes[0, 0].set_ylabel("φ, log-odds of failure\n(centered on the shaded shared window)", fontsize=9)
    axes[0, -1].legend(title="lookback window", loc="best", fontsize=9, title_fontsize=9)
    fig.suptitle("Telemetry φ curves at five lookback widths (shaded: the x-range every width covers densely)",
                 x=0.005, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_paired_differences(paired, canonical=CANONICAL_LOOKBACK):
    """AUC(width) - AUC(1h) on shared test tasks with its paired-bootstrap 95% CI, one row per width."""
    import matplotlib.pyplot as plt

    widths = [width for width in paired.index if width != canonical]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    y = np.arange(len(widths))[::-1]
    for pos, width in zip(y, widths):
        row = paired.loc[width]
        color = WIDTH_COLORS.get(width, MUTED)
        ax.hlines(pos, row["95% CI low"], row["95% CI high"], color=color, lw=2.0)
        ax.plot(row["AUC difference"], pos, "o", color=color, ms=9, mec=SURFACE, mew=2)
        # Values in a column right of the plot, at 5 decimals so a bound like +0.00003 does not print as +0.0000.
        ax.text(1.02, pos, f"{row['AUC difference']:+.5f}  [{row['95% CI low']:+.5f}, {row['95% CI high']:+.5f}]"
                f"   n = {int(row['shared test tasks']):,}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=8.5, color=INK_2)
    ax.text(1.02, len(widths) - 0.45, "difference   [95% CI]              shared test tasks",
            transform=ax.get_yaxis_transform(), va="bottom", fontsize=8.5, color=MUTED)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{width:g}h" for width in widths])
    ax.set_ylim(-0.6, len(widths) - 0.2)
    ax.set_xlabel("test AUC difference vs the 1-hour model, on the test tasks both share", fontsize=9)
    lo, hi = min(paired["95% CI low"].min(), 0), max(paired["95% CI high"].max(), 0)
    ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.suptitle("Does the window width change test AUC?  (dot = difference vs 1h, bar = paired-bootstrap 95% CI)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.68, 0.93))
    return fig
