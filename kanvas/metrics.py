"""Metrics and resampling statistics for the v2 evaluation.

- ``roc_auc``/``weighted_auc``: ROC-AUC with ties counted as one half, optionally
  with non-negative row weights. A bootstrap resample is a vector of row counts,
  so one sort per score vector serves all 2,000 resamples (``AucScorer``).
- ``pr_auc``: average precision (step-wise area under the precision-recall curve).
- ``brier``, ``ece`` and ``reliability``: calibration, with 15 equal-width bins.
- ``paired_bootstrap``: the difference in mean AUC between two models over one or
  more test sets, resampling rows (or users, for user-grouped protocols) with
  replacement, the same resample scored for both models.
- ``holm``: Holm's step-down adjustment of a family of p-values.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

ECE_BINS = 15
N_BOOT = 2000


# ----------------------------------------------------------------------------- point metrics
class AucScorer:
    """ROC-AUC of one score vector under many weight vectors (bootstrap resamples)."""

    def __init__(self, y, p):
        y = np.asarray(y).astype(bool)
        p = np.asarray(p, dtype=float)
        order = np.argsort(p, kind="mergesort")
        ps = p[order]
        # tie groups in ascending score order
        new = np.empty(len(ps), dtype=bool)
        new[:1] = True
        new[1:] = ps[1:] != ps[:-1]
        group_sorted = np.cumsum(new) - 1
        self.group = np.empty(len(p), dtype=np.int64)
        self.group[order] = group_sorted
        self.n_groups = int(group_sorted[-1]) + 1 if len(p) else 0
        self.y = y

    def __call__(self, w=None):
        if w is None:
            w = np.ones(len(self.y))
        wpos = np.bincount(self.group, weights=w * self.y, minlength=self.n_groups)
        wneg = np.bincount(self.group, weights=w * ~self.y, minlength=self.n_groups)
        neg_below = np.cumsum(wneg) - wneg
        P, N = wpos.sum(), wneg.sum()
        if P == 0 or N == 0:
            return np.nan
        return float((wpos * (neg_below + 0.5 * wneg)).sum() / (P * N))


def roc_auc(y, p):
    return AucScorer(y, p)()


def weighted_auc(y, p, w):
    return AucScorer(y, p)(np.asarray(w, dtype=float))


def pr_auc(y, p):
    return float(average_precision_score(np.asarray(y), np.asarray(p)))


def brier(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability(y, p, n_bins=ECE_BINS):
    """Per-bin count, mean prediction and observed failure rate (equal-width bins on [0, 1])."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.bincount(idx, weights=p, minlength=n_bins) / count
        rate = np.bincount(idx, weights=y, minlength=n_bins) / count
    return {"lower": edges[:-1], "upper": edges[1:], "count": count, "mean_p": mean_p, "rate": rate}


def ece(y, p, n_bins=ECE_BINS):
    """Expected calibration error: the count-weighted mean |observed rate - mean prediction| over bins."""
    r = reliability(y, p, n_bins)
    used = r["count"] > 0
    return float(np.sum(r["count"][used] * np.abs(r["rate"][used] - r["mean_p"][used])) / r["count"].sum())


def all_metrics(y, p):
    return {"roc_auc": roc_auc(y, p), "pr_auc": pr_auc(y, p), "brier": brier(y, p), "ece": ece(y, p)}


# ----------------------------------------------------------------------------- resampling
def resample_weights(n, rng, clusters=None):
    """Row counts of one bootstrap resample; with ``clusters`` (an int code per row), whole clusters are drawn."""
    if clusters is None:
        return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    n_clusters = int(clusters.max()) + 1
    wc = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters).astype(float)
    return wc[clusters]


def bootstrap_p(diffs, null=0.0):
    """Two-sided bootstrap p-value of ``null``: twice the smaller tail share, with the +1 correction."""
    diffs = np.asarray(diffs)
    diffs = diffs[np.isfinite(diffs)]
    b = len(diffs)
    tail = min(int(np.sum(diffs <= null)), int(np.sum(diffs >= null)))
    return float(min(1.0, 2.0 * (tail + 1) / (b + 1)))


def bootstrap_models(test_sets, n_boot=N_BOOT, seed=0):
    """Bootstrap the mean AUC of several models over one or more test sets.

    Args:
        test_sets: list of dicts, one per test set (split or fold), each with
            ``y`` (labels), ``scores`` (dict model -> list of score vectors, one
            per seed; the model's AUC on the set is the mean over seeds) and
            optionally ``clusters`` (int code per row: resample clusters).
        n_boot: Number of resamples.
        seed: RNG seed; the same resamples serve every model, so differences are paired.

    Returns:
        ``(point, boot)``: dict model -> mean AUC over sets, and dict model ->
        array (n_boot,) of resampled mean AUCs.
    """
    rng = np.random.default_rng(seed)
    models = list(test_sets[0]["scores"])
    scorers = [{m: [AucScorer(ts["y"], s) for s in ts["scores"][m]] for m in models} for ts in test_sets]
    point = {m: float(np.mean([np.mean([sc() for sc in scorers[k][m]]) for k in range(len(test_sets))]))
             for m in models}
    boot = {m: np.empty(n_boot) for m in models}
    for b in range(n_boot):
        per_set = {m: [] for m in models}
        for k, ts in enumerate(test_sets):
            w = resample_weights(len(ts["y"]), rng, ts.get("clusters"))
            for m in models:
                per_set[m].append(np.mean([sc(w) for sc in scorers[k][m]]))
        for m in models:
            boot[m][b] = np.mean(per_set[m])
    return point, boot


def percentile_ci(values, level=0.95):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    a = (1 - level) / 2
    return float(np.quantile(values, a)), float(np.quantile(values, 1 - a))


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
