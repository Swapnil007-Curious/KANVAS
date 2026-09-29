"""Phase 5 orchestration: scale-up, multi-seed stability, publication export.

This module only *applies* the methodology that Phases 2-4 already fixed. It
re-tunes nothing: lambda stays at the Phase 3 value, the MLP keeps the Phase 4
architecture and learning rate, logistic regression keeps its Phase 4 C, and
every model still scales its inputs with ranges fitted on the training split
alone. ``data.py``, ``bspline.py``, ``model.py``, ``training.py``,
``baselines.py`` and ``shap_analysis.py`` are imported unmodified.

What changes between runs here is only what each experiment is designed to
vary: the weight-initialization seed (Experiment A) or the split seed
(Experiment B). The canonical run reported as the paper's Table 1 is fixed in
advance by ``CANONICAL_SPLIT_STATE`` and ``CANONICAL_INIT_SEED`` below, not
chosen after seeing which run scored best.
"""

import contextlib
import io
import os
import time

import numpy as np
import pandas as pd
import torch

from .baselines import MLP, count_parameters
from .data import FEATURE_NAMES, build_kanvas_dataset, scale_features
from .model import KAAM
from .training import classification_metrics, predict_proba, train_kaam, train_val_test_split
from .utils import set_seed

# ---------------------------------------------------------------- fixed choices
# Every value below was decided in Phase 3 or Phase 4 and is applied here as-is.
LAMBDA_SMOOTH = 0.001  # Phase 3, chosen on validation loss and curve shape
PATIENCE = 15  # Phase 3
MAX_EPOCHS = 500  # Phase 3
KAAM_LR = 0.01  # Phase 3
NUM_KNOTS, SPLINE_ORDER = 10, 3  # Phase 3
MLP_HIDDEN, MLP_LR = (8, 6), 0.03  # Phase 4, won the validation sweep
LR_C = 0.1  # Phase 4, won the validation AUC sweep
THRESHOLD = 0.5  # Phase 4's headline table; not re-tuned per run

# The canonical run, fixed before any Phase 5 metric was computed:
# the Phase 3/4 split seed, and the first seed of Experiment A's seed list.
CANONICAL_SPLIT_STATE = 42
CANONICAL_INIT_SEED = 0
INIT_SEEDS = [0, 1, 2, 3, 4]
SPLIT_STATES = [42, 43, 44, 45, 46]

UNIT_RANGES = {name: (0.0, 1.0) for name in FEATURE_NAMES}


# ------------------------------------------------------------------- Step 1
def build_or_load_large_dataset(raw_dir, out_path, log_path, sample_n_tasks=40_000, min_rows=15_000,
                                max_sample=150_000, random_state=42, verbose=True):
    """Build the large-scale dataset, doubling ``sample_n_tasks`` until ``min_rows`` clean rows survive.

    ``build_kanvas_dataset`` is called unchanged; only the sample size moves.
    Its ``out_path`` is passed explicitly so the 2,310-row Phase 3 development
    file is never overwritten.

    The build streams a 2 GB instance table, so the result and the full
    survival log are cached. A later call with both files present reloads
    them instead of rebuilding, and replays the saved log verbatim.

    Returns:
        ``(X, y, feature_ranges, log_text, attempts)``: the feature matrix,
        label, per-feature (min, max), the captured build log, and a
        DataFrame with one row per sample size tried.
    """
    if os.path.exists(out_path) and os.path.exists(log_path):
        df = pd.read_csv(out_path, index_col=["job_name", "task_name"])
        X, y = df[FEATURE_NAMES], df["label"]
        with open(log_path, encoding="utf-8") as fh:
            log_text = fh.read()
        if verbose:
            print(f"[cached] {out_path}: {len(X):,} rows; replaying the saved build log\n")
            print(log_text)
        ranges = {c: (float(X[c].min()), float(X[c].max())) for c in FEATURE_NAMES}
        return X, y, ranges, log_text, _attempts_from_log(log_text)

    attempts, n = [], sample_n_tasks
    while True:
        buffer = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(buffer):
            X, y, ranges = build_kanvas_dataset(raw_dir, sample_n_tasks=n, random_state=random_state,
                                                out_path=out_path, verbose=True)
        log_text = buffer.getvalue()
        attempts.append({"sample_n_tasks": n, "clean rows": len(X), "failure rate": float(y.mean()),
                         "yield": len(X) / n, "seconds": time.perf_counter() - start,
                         "reached floor": len(X) >= min_rows})
        if verbose:
            print(log_text)
            print(f"[attempt] sample_n_tasks={n:,} -> {len(X):,} clean rows "
                  f"({len(X) / n:.1%} yield, {time.perf_counter() - start:.0f}s)")
        if len(X) >= min_rows or n >= max_sample:
            break
        n = min(n * 2, max_sample)

    if len(X) < min_rows:
        print(f"WARNING: cap sample_n_tasks={max_sample:,} reached with only {len(X):,} clean rows, "
              f"below the {min_rows:,} floor. Reporting the largest sample obtained.")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(log_text)
    return X, y, ranges, log_text, pd.DataFrame(attempts)


def _attempts_from_log(log_text):
    """One-row attempts table recovered from a cached build log."""
    final = [line for line in log_text.splitlines() if line.strip().startswith("final:")]
    rows = int(final[-1].split()[1].replace(",", "")) if final else np.nan
    return pd.DataFrame([{"sample_n_tasks": np.nan, "clean rows": rows, "failure rate": np.nan,
                          "yield": np.nan, "seconds": np.nan, "reached floor": True}])


def survival_table(log_text):
    """The survival funnel that ``build_kanvas_dataset`` printed, as text."""
    marker = "Survival funnel"
    return log_text[log_text.index(marker):] if marker in log_text else "(no survival funnel in log)"


# ------------------------------------------------------------------- Step 2
def representativeness(X_small, y_small, X_large, y_large, std_threshold=0.2, window_threshold=0.2):
    """Compare the large sample's feature distributions against the small one's.

    The flagging rule is fixed here rather than chosen after looking at the
    numbers. A feature is flagged when either:

    - its mean moves by more than ``std_threshold`` of the small sample's own
      standard deviation (the conventional "small effect" boundary), or
    - its 5th-95th percentile window shifts or stretches by more than
      ``window_threshold`` of the small sample's window width.

    The second rule matters because the phi_i curves are read over exactly
    that window, so a window that moved makes the Phase 3 curves describe a
    different region of the feature than the large-sample curves do.

    Returns:
        ``(table, flagged)``: per-feature statistics for both samples with the
        two shift measures and a ``flag`` column, and the list of flagged
        feature names.
    """
    rows = []
    for name in X_small.columns:
        s, l = X_small[name], X_large[name]
        s_p5, s_p95 = s.quantile([0.05, 0.95])
        l_p5, l_p95 = l.quantile([0.05, 0.95])
        width = s_p95 - s_p5
        std_shift = abs(l.mean() - s.mean()) / s.std() if s.std() > 0 else np.nan
        window_shift = max(abs(l_p5 - s_p5), abs(l_p95 - s_p95)) / width if width > 0 else np.nan
        rows.append({
            "small mean": s.mean(), "large mean": l.mean(), "small std": s.std(), "large std": l.std(),
            "small p5": s_p5, "large p5": l_p5, "small p95": s_p95, "large p95": l_p95,
            "|mean shift| / small std": std_shift, "window shift / small width": window_shift,
            "flag": bool(std_shift > std_threshold or window_shift > window_threshold),
        })
    table = pd.DataFrame(rows, index=list(X_small.columns))
    return table, list(table.index[table["flag"]])


# ------------------------------------------------------------------- Step 3
def train_canonical_kaam(X_train, y_train, X_val, y_val, init_seed, lambda_smooth=None, verbose=False):
    """One KAAM trained exactly as Phase 3 trained it, differing only in ``init_seed``.

    ``lambda_smooth`` defaults to the Phase 3 value and is overridden only by
    the Step 6 ablation, which needs the unpenalized (lambda = 0) curves to
    ask whether sample size alone removes the sparse-region artifacts. That
    ablation is a diagnostic, not a re-tuning: no model choice is made from it.
    """
    set_seed(init_seed)
    model = KAAM(FEATURE_NAMES, UNIT_RANGES, num_knots=NUM_KNOTS, spline_order=SPLINE_ORDER)
    start = time.perf_counter()
    result = train_kaam(model, X_train, y_train, X_val, y_val,
                        lambda_smooth=LAMBDA_SMOOTH if lambda_smooth is None else lambda_smooth,
                        patience=PATIENCE, max_epochs=MAX_EPOCHS, lr=KAAM_LR, verbose=verbose)
    return model, result, time.perf_counter() - start


def _metrics_row(model, result, X_test, y_test, **extra):
    p = predict_proba(model, X_test, result.feature_ranges)
    m = classification_metrics(y_test.to_numpy(), p, THRESHOLD)
    return {**extra, "test AUC": m["auc"], "test accuracy": m["accuracy"], "test F1": m["f1"],
            "best epoch": result.best_epoch, "val BCE": result.best_val_loss}


def initialization_variance(X, y, seeds=INIT_SEEDS, split_state=CANONICAL_SPLIT_STATE, verbose=True):
    """Experiment A: one fixed split, ``len(seeds)`` weight initializations.

    Returns a DataFrame with exactly one row per seed, in the order given.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, random_state=split_state)
    rows = []
    for seed in seeds:
        model, result, seconds = train_canonical_kaam(X_train, y_train, X_val, y_val, seed)
        rows.append(_metrics_row(model, result, X_test, y_test, init_seed=seed, split_state=split_state,
                                 seconds=seconds))
        if verbose:
            print(f"  init seed {seed}: test AUC {rows[-1]['test AUC']:.4f}  "
                  f"acc {rows[-1]['test accuracy']:.4f}  F1 {rows[-1]['test F1']:.4f}  ({seconds:.0f}s)")
    return pd.DataFrame(rows)


def split_variance(X, y, split_states=SPLIT_STATES, init_seed=CANONICAL_INIT_SEED, verbose=True):
    """Experiment B: one fixed initialization, ``len(split_states)`` stratified splits.

    Returns a DataFrame with exactly one row per split state, in the order given.
    """
    rows = []
    for state in split_states:
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, random_state=state)
        model, result, seconds = train_canonical_kaam(X_train, y_train, X_val, y_val, init_seed)
        rows.append(_metrics_row(model, result, X_test, y_test, init_seed=init_seed, split_state=state,
                                 seconds=seconds))
        if verbose:
            print(f"  split state {state}: test AUC {rows[-1]['test AUC']:.4f}  "
                  f"acc {rows[-1]['test accuracy']:.4f}  F1 {rows[-1]['test F1']:.4f}  ({seconds:.0f}s)")
    return pd.DataFrame(rows)


def variance_summary(experiment_a, experiment_b, metrics=("test AUC", "test accuracy", "test F1")):
    """Mean, std and range of each metric for both experiments, side by side."""
    rows = {}
    for label, table in (("A: initialization only", experiment_a), ("B: split only", experiment_b)):
        row = {}
        for metric in metrics:
            values = table[metric]
            row[f"{metric} mean"] = values.mean()
            row[f"{metric} std"] = values.std(ddof=1)
            row[f"{metric} range"] = values.max() - values.min()
        rows[label] = row
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------- Step 6
def phi_curve(model, name, feature_ranges, x_lo, x_hi, num_points=200, center=None):
    """``(x_grid, phi)`` for one feature over raw-unit ``[x_lo, x_hi]``.

    ``center`` subtracts a constant from the curve so two models trained on
    different data can be compared on shape rather than on an arbitrary
    vertical offset; ``"mean"`` centers each curve on its own mean.
    """
    lo, hi = feature_ranges[name]
    grid = np.linspace(x_lo, x_hi, num_points)
    with torch.no_grad():
        phi = model.edges[name](torch.tensor((grid - lo) / (hi - lo), dtype=torch.float32)).numpy()
    if center == "mean":
        phi = phi - phi.mean()
    elif center is not None:
        phi = phi - center
    return grid, phi


def curve_frame(model, feature_ranges, X_ref, lower=0.05, upper=0.95, num_points=200, center="mean"):
    """Long-format ``(feature_name, x_value, phi_value)`` over each feature's p5-p95 window."""
    rows = []
    for name in model.feature_names:
        x_lo, x_hi = X_ref[name].quantile([lower, upper])
        grid, phi = phi_curve(model, name, feature_ranges, x_lo, x_hi, num_points, center=center)
        rows.append(pd.DataFrame({"feature_name": name, "x_value": grid, "phi_value": phi}))
    return pd.concat(rows, ignore_index=True)


def curve_roughness(model, feature_ranges, X_ref, lower=0.05, upper=0.95, num_points=200):
    """Per-feature curvature of phi_i over the p5-p95 window: mean |second difference|, and the curve's range.

    Scale-free enough to compare a small-sample curve with a large-sample one:
    both are evaluated on the same number of points over each model's own
    percentile window, so a wigglier curve scores higher whatever the units.
    """
    rows = {}
    for name in model.feature_names:
        x_lo, x_hi = X_ref[name].quantile([lower, upper])
        _, phi = phi_curve(model, name, feature_ranges, x_lo, x_hi, num_points, center="mean")
        second = np.diff(phi, n=2)
        rows[name] = {"phi range": float(phi.max() - phi.min()),
                      "mean |2nd diff| x1e3": float(np.abs(second).mean() * 1e3),
                      "sign changes in slope": int((np.diff(np.sign(np.diff(phi))) != 0).sum())}
    return pd.DataFrame(rows).T


def sparse_region_swing(model, feature_ranges, X_ref, lower=0.05, upper=0.95, num_points=200):
    """How much each phi_i swings in its sparse tails compared with its dense core.

    Phase 3's artifacts were curves bending hard over x-ranges holding almost
    no training tasks. That is a statement about the tails, not the middle, so
    the core (the p5-p95 window the importance ranking uses) and the two tails
    (below p5 and above p95) are measured separately.

    ``tail / core`` above 1 means the curve moves more in the thin-data tails
    than across the whole body of the data, which is the artifact signature.

    Returns:
        DataFrame indexed by feature with the core range, the larger of the
        two tail ranges, their ratio, and the share of rows in the tails.
    """
    rows = {}
    for name in model.feature_names:
        x_min, x_max = float(X_ref[name].min()), float(X_ref[name].max())
        q_lo, q_hi = X_ref[name].quantile([lower, upper])
        _, core = phi_curve(model, name, feature_ranges, q_lo, q_hi, num_points)
        core_range = float(core.max() - core.min())
        tails = []
        for lo, hi in ((x_min, q_lo), (q_hi, x_max)):
            if hi > lo:
                _, tail = phi_curve(model, name, feature_ranges, lo, hi, num_points)
                tails.append(float(tail.max() - tail.min()))
        tail_range = max(tails) if tails else 0.0
        rows[name] = {"core range (p5-p95)": core_range, "worst tail range": tail_range,
                      "tail / core": tail_range / core_range if core_range > 0 else np.nan,
                      "rows in tails": float(((X_ref[name] < q_lo) | (X_ref[name] > q_hi)).mean())}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------- Step 7
#: Columns that must never reach an exported file: Alibaba's internal identifiers.
FORBIDDEN_COLUMNS = ["user", "job_name", "task_name", "machine", "worker_name", "inst_id", "gpu_type"]


def strip_identifiers(df):
    """Drop any identifier column, and flatten a (job_name, task_name) index into an anonymous task_id.

    The feature matrix is indexed by ``(job_name, task_name)``, which are
    Alibaba's own job and task strings. Exported files get a row number
    instead, so nothing traceable to a named internal job leaves the project.
    """
    out = df.reset_index(drop=True).copy()
    dropped = [c for c in out.columns if c in FORBIDDEN_COLUMNS]
    out = out.drop(columns=dropped)
    out.insert(0, "task_id", np.arange(len(out)))
    return out, dropped


def audit_exports(paths):
    """Check every exported CSV for identifier columns and for object-dtype columns that could hold them."""
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        present = [c for c in df.columns if c in FORBIDDEN_COLUMNS]
        text_cols = [c for c in df.columns if df[c].dtype == object]
        rows.append({"file": os.path.basename(path), "rows": len(df), "columns": df.shape[1],
                     "identifier columns": ", ".join(present) if present else "none",
                     "text columns": ", ".join(text_cols) if text_cols else "none",
                     "clean": not present})
    return pd.DataFrame(rows).set_index("file")


def mlp_at_fixed_architecture(n_inputs):
    """The Phase 4 MLP, rebuilt at its fixed architecture (no re-tuning)."""
    return MLP(n_inputs, MLP_HIDDEN)


def parameter_fairness(kaam):
    """KAAM's and the fixed MLP's learnable parameter counts, and whether they stay within 20%."""
    k, m = count_parameters(kaam), count_parameters(mlp_at_fixed_architecture(len(kaam.feature_names)))
    return {"KAAM": k, "MLP": m, "relative": m / k - 1, "within 20%": abs(m - k) <= 0.2 * k}


def scaled_matrix(X, feature_ranges):
    """``scale_features`` as every model here uses it, returned as a float32 numpy array."""
    return scale_features(X, feature_ranges).to_numpy(np.float32)
