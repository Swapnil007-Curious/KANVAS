"""Phase 6, Part B: case-level audit trails from KAAM's own ``explain()``.

``KAAM.explain()`` has been in ``model.py`` since the first commit and had never
been run on a real task. It is the point of the architecture: the model is
``sigmoid(bias + sum_i phi_i(x_i))``, so one prediction splits exactly into ten
per-feature numbers, with nothing estimated and nothing approximated.

Three things live here:

1. **The canonical model.** Phase 5 saved no checkpoint and no split indices,
   so ``reproduce_canonical_model`` rebuilds both by Phase 5's own rule (split
   ``random_state=42``, initialization seed 0), and ``save_checkpoint`` stores
   the result so later work can load it instead of retraining.
2. **Case selection**, by a rule fixed before any prediction was looked at.
3. **The decomposition and its audit.** ``explain()`` returns each
   ``phi_i(x_i)`` in log-odds. Those raw values carry per-feature offsets the
   data cannot pin down (only ``bias + sum of offsets`` is identified), so the
   charts show each feature relative to the average training task:
   ``phi_i(x_i)`` minus ``phi_i``'s training mean, starting from the average
   task's logit. For an additive model these are exactly the interventional
   SHAP values (``shap_analysis.kaam_shap_values``), and they still sum to the
   model's logit. ``consistency_check`` confirms on the raw values that
   ``sum(explain()) + bias`` is the logit ``forward()`` computes.
"""

import contextlib
import hashlib

import numpy as np
import pandas as pd
import torch

from .data import FEATURE_NAMES, scale_features
from .model import KAAM
from .stability import (CANONICAL_INIT_SEED, CANONICAL_SPLIT_STATE, KAAM_LR, LAMBDA_SMOOTH, MAX_EPOCHS, NUM_KNOTS,
                        PATIENCE, SPLINE_ORDER, THRESHOLD, UNIT_RANGES, train_canonical_kaam)
from .training import train_val_test_split

ROLES = ("train", "val", "test")

#: The four cases, and the rule that picks each one; fixed before any prediction was seen.
CASE_ORDER = ["True positive", "True negative", "False negative", "False positive"]
CASE_RULES = {
    "True positive": "correctly caught failure with the highest predicted risk",
    "True negative": "correctly cleared task with the lowest predicted risk",
    "False negative": "missed failure with the highest predicted risk (the near miss)",
    "False positive": "false alarm with the highest predicted risk (the most confidently wrong)",
}

#: Diverging poles of the chart palette: pushes failure risk up (red) or down (blue).
RAISES, LOWERS = "#e34948", "#2a78d6"
SURFACE, INK, INK_2, MUTED, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"


# ------------------------------------------------------------------ the canonical model
def reproduce_canonical_model(X, y, verbose=False):
    """Phase 5's canonical KAAM, rebuilt by Phase 5's own rule.

    ``train_val_test_split(X, y, random_state=CANONICAL_SPLIT_STATE)`` (42),
    initialization seed ``CANONICAL_INIT_SEED`` (0), and the fixed Phase 3
    settings through ``train_canonical_kaam``: the same code path as notebook
    05's Table 1 run.

    Returns:
        ``(model, result, splits)`` with ``splits`` the usual 6-tuple
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    splits = train_val_test_split(X, y, random_state=CANONICAL_SPLIT_STATE)
    X_tr, X_va, _, y_tr, y_va, _ = splits
    model, result, _ = train_canonical_kaam(X_tr, y_tr, X_va, y_va, CANONICAL_INIT_SEED, verbose=verbose)
    return model, result, splits


def file_sha256(path):
    """SHA-256 of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, model, feature_ranges, splits, X, source_csv, metrics):
    """Store the canonical model, its training-time scaling ranges and its split.

    The split is stored as row positions in ``source_csv``, whose SHA-256 is
    recorded alongside, rather than as Alibaba's job and task names.

    Args:
        path: Output ``.pt`` file.
        model: The trained KAAM.
        feature_ranges: The (min, max) scaling ranges fitted on its training split.
        splits: ``(X_train, X_val, X_test, ...)`` as returned by the split.
        X: The full feature matrix the split positions refer to.
        source_csv: The CSV ``X`` was read from.
        metrics: Dict of the test metrics the model scored, kept for reference.
    """
    positions = {}
    for role, part in zip(ROLES, splits[:3]):
        pos = X.index.get_indexer(part.index)
        if (pos < 0).any():
            raise ValueError(f"{role} split holds tasks that are not in X")
        positions[role] = torch.as_tensor(pos, dtype=torch.int64)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_names": list(model.feature_names),
        "feature_ranges": {name: (float(lo), float(hi)) for name, (lo, hi) in feature_ranges.items()},
        "split_positions": positions,
        "source_csv_sha256": file_sha256(source_csv),
        "rule": {"split_random_state": CANONICAL_SPLIT_STATE, "init_seed": CANONICAL_INIT_SEED,
                 "lambda_smooth": LAMBDA_SMOOTH, "patience": PATIENCE, "max_epochs": MAX_EPOCHS, "lr": KAAM_LR,
                 "num_knots": NUM_KNOTS, "spline_order": SPLINE_ORDER, "threshold": THRESHOLD},
        "metrics": {name: float(value) for name, value in metrics.items()},
    }, path)


def load_checkpoint(path, source_csv):
    """The saved canonical model, its scaling ranges, and its split rebuilt from row positions.

    Returns:
        ``(model, feature_ranges, splits, checkpoint)``.

    Raises:
        ValueError: if ``source_csv`` is not byte-identical to the file the
            checkpoint was made from.
    """
    checkpoint = torch.load(path, weights_only=True)
    if file_sha256(source_csv) != checkpoint["source_csv_sha256"]:
        raise ValueError(f"{source_csv} is not the file this checkpoint's split positions refer to")
    df = pd.read_csv(source_csv, index_col=["job_name", "task_name"])
    X, y = df[FEATURE_NAMES], df["label"]
    model = KAAM(checkpoint["feature_names"], UNIT_RANGES, num_knots=NUM_KNOTS, spline_order=SPLINE_ORDER)
    model.load_state_dict(checkpoint["state_dict"])
    positions = [checkpoint["split_positions"][role].numpy() for role in ROLES]
    splits = tuple(X.iloc[pos] for pos in positions) + tuple(y.iloc[pos] for pos in positions)
    return model, dict(checkpoint["feature_ranges"]), splits, checkpoint


# ------------------------------------------------------------------ Step 1
def select_cases(y_true, p, threshold=THRESHOLD):
    """Pick the four audit cases mechanically from the test predictions.

    Each case is the most confident member of its outcome group: the lowest
    predicted risk for the true negative, the highest for the other three.
    Ties go to the first task in test-set order and are counted, not hidden.

    Args:
        y_true: Test labels, a Series indexed by task.
        p: Predicted failure probabilities, a Series indexed exactly like ``y_true``.
        threshold: A task is flagged when ``p >= threshold``.

    Returns:
        DataFrame indexed by case name (in ``CASE_ORDER``) with the task's
        index label, its position in the test set, its predicted risk, true
        label, whether it was flagged, the size of its outcome group, and how
        many tasks in the group share the selected risk.

    Raises:
        ValueError: if the indexes differ or any outcome group is empty,
            which would leave the audit silently incomplete.
    """
    if not p.index.equals(y_true.index):
        raise ValueError("predictions and labels must share an index, in the same order")
    flagged = p >= threshold
    groups = {
        "True positive": (y_true == 1) & flagged,
        "True negative": (y_true == 0) & ~flagged,
        "False negative": (y_true == 1) & ~flagged,
        "False positive": (y_true == 0) & flagged,
    }
    empty = [name for name, mask in groups.items() if not mask.any()]
    if empty:
        raise ValueError(f"no test task falls in {empty}; the four-case audit cannot be completed")

    rows = {}
    for name in CASE_ORDER:
        members = p[groups[name]]
        pick = members.idxmin() if name == "True negative" else members.idxmax()
        rows[name] = {"task": pick, "test row": int(y_true.index.get_loc(pick)),
                      "predicted risk": float(members[pick]), "true label": int(y_true[pick]),
                      "flagged": bool(flagged[pick]), "group size": int(len(members)),
                      "tasks tied at this risk": int((members == members[pick]).sum()),
                      "rule": CASE_RULES[name]}
    return pd.DataFrame(rows).T.rename_axis("case")


# ------------------------------------------------------------------ Step 2
def phi_training_means(model, X_train, feature_ranges):
    """Mean of each ``phi_i`` over the training tasks: the offset that centers a feature's contribution."""
    scaled = torch.tensor(scale_features(X_train, feature_ranges).to_numpy(np.float32))
    with torch.no_grad():
        return pd.Series({name: model.edges[name](scaled[:, i]).double().mean().item()
                          for i, name in enumerate(model.feature_names)}, name="training mean of phi")


def explain_case(model, X, feature_ranges, task, phi_means, X_reference=None):
    """One task's prediction, split by ``model.explain()`` into per-feature log-odds contributions.

    Args:
        model: The trained KAAM.
        X: Raw-unit features holding ``task``.
        feature_ranges: The model's training-time scaling ranges.
        task: Index label of the task to explain.
        phi_means: ``phi_training_means`` of the same model.
        X_reference: Optional raw-unit training features, to place each raw
            value at its percentile among training tasks.

    Returns:
        DataFrame with one row per feature, sorted by the size of the
        contribution relative to the average task: the raw value an engineer
        sees, optionally its training percentile, the [0, 1]-scaled value the
        model sees, explain()'s ``phi_i(x_i)``, ``phi_i``'s training mean, and
        their difference (``vs average task``).
    """
    raw = X.loc[[task]]
    scaled = scale_features(raw, feature_ranges)
    phi = pd.Series(model.explain(torch.tensor(scaled.to_numpy(np.float32)[0])))
    table = pd.DataFrame({"raw value": raw.iloc[0], "scaled value": scaled.iloc[0], "phi = explain()": phi,
                          "training mean of phi": phi_means})
    if X_reference is not None:
        # Mid-rank percentile, so a value shared by many tasks sits in the middle of its tie block.
        below = [(X_reference[name] < value).mean() for name, value in raw.iloc[0].items()]
        at_or_below = [(X_reference[name] <= value).mean() for name, value in raw.iloc[0].items()]
        table.insert(1, "training percentile", 50 * (np.array(below) + np.array(at_or_below)))
    table["vs average task"] = table["phi = explain()"] - table["training mean of phi"]
    return table.loc[table["vs average task"].abs().sort_values(ascending=False).index]


# ------------------------------------------------------------------ Step 3
@contextlib.contextmanager
def _recording_sigmoid(store):
    """Temporarily wrap ``torch.sigmoid`` so every tensor passed to it is recorded."""
    original = torch.sigmoid

    def recording(tensor, *args, **kwargs):
        store.append(tensor.detach().clone())
        return original(tensor, *args, **kwargs)

    torch.sigmoid = recording
    try:
        yield store
    finally:
        torch.sigmoid = original


def forward_logits(model, x):
    """The logit ``model.forward()`` computes, read where forward() hands it to its final sigmoid.

    ``forward()`` returns only the probability, and model.py stays unmodified,
    so the logit is captured at the input of ``torch.sigmoid`` during a real
    forward pass. Inverting the float32 probability instead would lose
    precision near 0 and 1, and would test the inversion rather than the model.

    Returns:
        ``(logits, probabilities)`` as numpy arrays from the same forward pass.
    """
    captured = []
    with torch.no_grad(), _recording_sigmoid(captured):
        probabilities = model(x)
    if len(captured) != 1:
        raise RuntimeError(f"expected forward() to call torch.sigmoid once, saw {len(captured)} calls")
    return captured[0].numpy(), probabilities.numpy()


def consistency_check(model, X, feature_ranges, tasks, labels=None):
    """Does ``sum(explain()) + bias`` equal the logit ``forward()`` computes, task by task?

    ``forward()`` runs once on the batch of all ``tasks``; ``explain()`` runs
    once per task on the same scaled inputs. The two are the same additive sum
    computed by different code paths, so they should agree to float32 rounding.

    Returns:
        DataFrame with one row per task (indexed by ``labels`` if given): the
        sum of the ten explain() values, the bias, their total, forward()'s
        logit and the absolute difference, then the same comparison after the
        sigmoid.
    """
    bias = float(model.bias.item())
    scaled = torch.tensor(scale_features(X.loc[list(tasks)], feature_ranges).to_numpy(np.float32))
    logits, probabilities = forward_logits(model, scaled)
    rows = []
    for i in range(len(tasks)):
        phi_sum = float(sum(model.explain(scaled[i]).values()))
        total = phi_sum + bias
        rows.append({"sum of explain()": phi_sum, "bias": bias, "explain() + bias": total,
                     "forward() logit": float(logits[i]), "|logit difference|": abs(total - float(logits[i])),
                     "sigmoid(explain() + bias)": 1.0 / (1.0 + np.exp(-total)),
                     "forward() risk": float(probabilities[i]),
                     "|risk difference|": abs(1.0 / (1.0 + np.exp(-total)) - float(probabilities[i]))})
    return pd.DataFrame(rows, index=list(labels) if labels is not None else list(range(len(tasks))))


# ------------------------------------------------------------------ the chart
def format_value(name, value):
    """A raw feature value in the units an engineer reads."""
    if name == "plan_cpu":
        return f"{value:,.0f} ({value / 100:g} core" + ("" if value == 100 else "s") + ")"
    if name == "plan_gpu":
        return f"{value:,.0f} ({value / 100:g} GPU" + ("s" if value > 100 else "") + ")"
    if name == "plan_mem":
        return f"{value:,.1f} GB"
    if name == "inst_num":
        return f"{value:,.0f} instance" + ("" if value == 1 else "s")
    if name == "cap_gpu":
        return f"{value:,.0f} GPUs installed"
    if name == "avg_net_receive":
        return f"{value / 1e6:,.0f} MB/s"
    if name == "avg_load_1":
        return f"{value:,.1f}"
    if name == "avg_gpu_util":
        return f"{value:,.0f}% (all GPUs)"
    return f"{value:,.1f}%"


def plot_waterfall(table, baseline_logit, title, subtitle, threshold=THRESHOLD, ax=None):
    """Horizontal waterfall of one prediction, from the average training task to this task's logit.

    Top to bottom: the average training task's logit, one bar per feature
    (largest move first), each starting where the previous one ended, then
    this task's logit. Red bars raise the failure risk, blue bars lower it.
    Every feature row is labeled with its raw value and training percentile.
    The dashed line is the decision threshold on the log-odds scale.

    Args:
        table: ``explain_case`` output (sorted, with ``vs average task``).
        baseline_logit: The average training task's logit (bias + sum of phi means).
        title, subtitle: Chart heading lines.
        threshold: Decision threshold on the probability scale.
        ax: Axes to draw on; a new figure is made if None.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6.2))
    fig = ax.figure
    contributions = table["vs average task"].to_numpy()
    ends = baseline_logit + np.cumsum(contributions)
    starts = ends - contributions
    final = float(ends[-1])
    positions = np.arange(1, len(contributions) + 1)
    risk = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731

    ax.barh(positions, contributions, left=starts, height=0.62, color=[RAISES if c > 0 else LOWERS for c in contributions],
            edgecolor=SURFACE, linewidth=1, zorder=3)
    for pos, end in zip(positions[:-1], ends[:-1]):  # connector: where one bar ends, the next begins
        ax.plot([end, end], [pos + 0.31, pos + 0.69], color=MUTED, lw=0.8, zorder=2)
    ax.plot([baseline_logit, baseline_logit], [0.25, 0.69], color=MUTED, lw=0.8, zorder=2)
    ax.plot([final, final], [positions[-1] + 0.31, positions[-1] + 0.75], color=MUTED, lw=0.8, zorder=2)

    lo = min(starts.min(), ends.min(), baseline_logit, 0.0)
    hi = max(starts.max(), ends.max(), baseline_logit, 0.0)
    span = hi - lo
    ax.set_xlim(lo - 0.08 * span, hi + 0.14 * span)
    backing = dict(facecolor=SURFACE, edgecolor="none", pad=0.6)  # keeps a label legible over the threshold line
    for pos, c, start, end in zip(positions, contributions, starts, ends):
        ax.text(max(start, end) + 0.012 * span, pos, f"{c:+.4f}" if abs(c) < 0.005 else f"{c:+.2f}", va="center",
                ha="left", fontsize=8.5, color=INK_2, zorder=4, bbox=backing)

    end_row = len(contributions) + 1
    for pos, value in ((0, baseline_logit), (end_row, final)):
        ax.plot(value, pos, "o", color=INK, ms=8, mec=SURFACE, mew=2, zorder=5)
        ax.text(value + 0.02 * span, pos, f"{value:+.2f}", va="center", ha="left", fontsize=9, color=INK, zorder=5,
                bbox=backing)

    threshold_logit = float(np.log(threshold / (1 - threshold)))
    ax.axvline(threshold_logit, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(threshold_logit, -0.85, f"decision threshold (risk {threshold:g})", ha="center", va="bottom",
            fontsize=8, color=INK_2)

    labels = [f"start: average training task, risk {risk(baseline_logit):.3f}"]
    for name, row in table.iterrows():
        label = f"{name} = {format_value(name, row['raw value'])}"
        if "training percentile" in table.columns:
            label += f"  · p{row['training percentile']:.0f}"
        labels.append(label)
    labels.append(f"= this task, risk {risk(final):.3f}")
    ax.set_yticks(np.arange(end_row + 1))
    ax.set_yticklabels(labels, fontsize=8.5)
    for tick in (ax.get_yticklabels()[0], ax.get_yticklabels()[-1]):
        tick.set_color(INK)
        tick.set_fontweight("bold")
    ax.set_ylim(end_row + 0.7, -1.4)
    ax.set_xlabel("log-odds of failure  (each bar: that feature's contribution relative to the average training task)",
                  fontsize=9)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, pad=24)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    # The waterfall drifts toward the final logit, so the lower corner on the start side stays empty.
    ax.legend(handles=[Patch(color=RAISES, label="raises failure risk"), Patch(color=LOWERS, label="lowers failure risk")],
              loc="lower left" if final >= baseline_logit else "lower right", fontsize=8.5)
    fig.tight_layout()
    return fig


def case_summary(cases, checks):
    """Selection facts and the forward/explain agreement side by side, one row per case."""
    summary = cases.drop(columns=["task", "rule"]).copy()
    for column in ["explain() + bias", "forward() logit", "|logit difference|", "|risk difference|"]:
        summary[column] = checks.loc[summary.index, column]
    return summary
