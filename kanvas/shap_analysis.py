"""SHAP agreement analysis: does KAAM's built-in explanation match a post-hoc one?

KAAM explains itself by construction: its prediction is a sum of per-feature
curves phi_i, so each curve is that feature's exact contribution. The MLP has
no such structure, so its explanation has to be estimated after training,
here with SHAP. If KAAM's intrinsic importance ranking roughly agrees with
the MLP's SHAP ranking, and the shapes behind them match, the built-in
explanation is tracking something real in the data rather than telling a
story of its own.

Both explanations are expressed in log-odds of failure: SHAP explains the
MLP's pre-sigmoid logit, and phi_i is already a log-odds contribution.
"""

import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr

from .data import scale_features

KAAM_COLOR, MLP_COLOR, SURFACE, INK_2, MUTED, BASELINE = "#2a78d6", "#eb6834", "#fcfcfb", "#52514e", "#898781", "#c3c2b7"


def kernel_shap_values(logit_fn, X_background, X_explain, feature_ranges):
    """Kernel SHAP values of ``logit_fn`` for every row of ``X_explain``.

    With 10 features there are only 2**10 feature coalitions, so every one of
    them is evaluated: the result is exact (interventional) Shapley values
    relative to the background rows, with no sampling noise and no
    regularization.

    Args:
        logit_fn: Maps an (n, n_features) array of [0, 1]-scaled features to n logits.
        X_background, X_explain: Raw-unit feature DataFrames.
        feature_ranges: The scaling ranges the model was trained with.

    Returns:
        ``(values, expected_value)``: a DataFrame of SHAP values indexed like
        ``X_explain`` with one column per feature, and the mean logit over
        the background rows.
    """
    background = scale_features(X_background, feature_ranges).to_numpy()
    explain = scale_features(X_explain, feature_ranges).to_numpy()
    explainer = shap.KernelExplainer(logit_fn, background)
    values = explainer.shap_values(explain, nsamples=2 ** explain.shape[1], l1_reg=False, silent=True)
    return pd.DataFrame(values, index=X_explain.index, columns=X_explain.columns), float(explainer.expected_value)


def mlp_logit(mlp):
    """The MLP's pre-sigmoid failure logit, as a numpy -> numpy function."""
    def logit(x):
        with torch.no_grad():
            return mlp.net(torch.as_tensor(np.asarray(x, dtype=np.float32))).numpy().ravel()
    return logit


def kaam_logit(model):
    """KAAM's pre-sigmoid failure logit, sum_i phi_i(x_i) + bias, as a numpy -> numpy function."""
    def logit(x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            return (sum(model.edges[name](x[:, i]) for i, name in enumerate(model.feature_names)) + model.bias).numpy()
    return logit


def mlp_shap_values(mlp, X_background, X_explain, feature_ranges):
    """Exact Kernel SHAP values of the MLP's failure logit; see ``kernel_shap_values``."""
    return kernel_shap_values(mlp_logit(mlp), X_background, X_explain, feature_ranges)


def kaam_shap_values(model, X_background, X_explain, feature_ranges):
    """Exact SHAP values of KAAM's logit, in closed form.

    For an additive model, feature i's interventional SHAP value is
    ``phi_i(x_i)`` minus the mean of ``phi_i`` over the background rows, so
    no estimation is needed: SHAP just returns KAAM's own curves, centered.
    """
    background = scale_features(X_background, feature_ranges)
    explain = scale_features(X_explain, feature_ranges)
    values = {}
    with torch.no_grad():
        for name in model.feature_names:
            edge = model.edges[name]
            center = edge(torch.tensor(background[name].to_numpy(np.float32))).mean().item()
            values[name] = edge(torch.tensor(explain[name].to_numpy(np.float32))).numpy() - center
    return pd.DataFrame(values, index=X_explain.index)


def shap_importance(values):
    """Mean |SHAP value| per feature, in the column order of ``values``."""
    return values.abs().mean().rename("mean |SHAP|")


def kaam_curve_range_importance(model, X_ref, feature_ranges, lower=0.05, upper=0.95, num_points=200):
    """Range (max - min) of each phi_i over the [lower, upper] quantile window of that feature in ``X_ref``.

    Only the central window is used, so thin-data extremes, where Phase 3
    showed curves are least trustworthy, can't inflate a feature's importance.
    """
    ranges = {}
    with torch.no_grad():
        for name in model.feature_names:
            q_lo, q_hi = X_ref[name].quantile([lower, upper])
            lo, hi = feature_ranges[name]
            grid = torch.tensor((np.linspace(q_lo, q_hi, num_points) - lo) / (hi - lo), dtype=torch.float32)
            phi = model.edges[name](grid).numpy()
            ranges[name] = float(phi.max() - phi.min())
    return pd.Series(ranges, name="KAAM curve range")


def compare_rankings(importance_a, importance_b):
    """Spearman rank correlation between two per-feature importance Series (same index).

    Returns:
        ``(rho, p_value, table)``. The table has each importance normalized to
        sum to 1, each rank (1 = most important), and the rank difference a - b.
    """
    if list(importance_a.index) != list(importance_b.index):
        raise ValueError("both importance Series must list the same features in the same order")
    rho, p_value = spearmanr(importance_a, importance_b)
    a, b = importance_a.name, importance_b.name
    table = pd.DataFrame({
        f"{a} share": importance_a / importance_a.sum(),
        f"{b} share": importance_b / importance_b.sum(),
        f"{a} rank": importance_a.rank(ascending=False).astype(int),
        f"{b} rank": importance_b.rank(ascending=False).astype(int),
    })
    table["rank difference"] = table[f"{a} rank"] - table[f"{b} rank"]
    return float(rho), float(p_value), table


def agreement_extremes(table, k=2):
    """The k features the two rankings agree on most, and the k they disagree on most.

    Agreement is the smallest |rank difference|, ties going to the feature
    with the larger average importance share. Disagreement is the largest
    |rank difference|, ties going to the larger gap in importance share.
    """
    shares = table[[c for c in table.columns if c.endswith("share")]]
    ranked = pd.DataFrame({
        "abs rank difference": table["rank difference"].abs(),
        "mean share": shares.mean(axis=1),
        "share gap": (shares.iloc[:, 0] - shares.iloc[:, 1]).abs(),
    })
    agree = ranked.sort_values(["abs rank difference", "mean share"], ascending=[True, False]).index[:k]
    disagree = ranked.sort_values(["abs rank difference", "share gap"], ascending=[False, False]).index[:k]
    return list(agree), list(disagree)


def plot_importance_comparison(table, kaam_col="KAAM curve range share", shap_col="MLP mean |SHAP| share"):
    """Paired horizontal bars: KAAM's normalized curve-range importance vs the MLP's normalized SHAP importance."""
    import matplotlib.pyplot as plt

    order = table.sort_values(kaam_col).index  # most important at the top
    y = np.arange(len(order))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y + height / 2, table.loc[order, kaam_col], height, color=KAAM_COLOR, edgecolor=SURFACE, linewidth=1,
            label="KAAM: φ range over the 5th–95th percentile (built in)")
    ax.barh(y - height / 2, table.loc[order, shap_col], height, color=MLP_COLOR, edgecolor=SURFACE, linewidth=1,
            label="MLP: mean |SHAP| (post hoc)")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("share of the model's total importance")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    return fig


def plot_dependence_pairs(features, labels, X_explain, mlp_values, kaam_model, X_background, feature_ranges,
                          X_ref, lower=0.05, upper=0.95):
    """For each feature: the MLP's SHAP dependence plot next to KAAM's centered phi curve.

    Both panels of a row share the x-axis (raw units) and the y-axis (log-odds
    relative to the background mean), so shape and size compare directly.
    The shaded band in each KAAM panel is the 5th-95th percentile window of
    ``X_ref`` that the importance ranking used.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(features), 2, figsize=(12, 3.3 * len(features)), sharey="row", squeeze=False)
    background = scale_features(X_background, feature_ranges)
    for row, (name, label) in enumerate(zip(features, labels)):
        ax_mlp, ax_kaam = axes[row]
        x = X_explain[name]
        ax_mlp.scatter(x, mlp_values[name], s=14, color=MLP_COLOR, alpha=0.55, edgecolors=SURFACE, linewidths=0.4)

        lo, hi = feature_ranges[name]
        grid = np.linspace(x.min(), x.max(), 300)
        edge = kaam_model.edges[name]
        with torch.no_grad():
            center = edge(torch.tensor(background[name].to_numpy(np.float32))).mean().item()
            phi = edge(torch.tensor((grid - lo) / (hi - lo), dtype=torch.float32)).numpy() - center
        ax_kaam.plot(grid, phi, color=KAAM_COLOR, lw=2)
        q_lo, q_hi = X_ref[name].quantile([lower, upper])
        ax_kaam.axvspan(q_lo, q_hi, color=KAAM_COLOR, alpha=0.06, lw=0)
        ax_kaam.plot(x, np.full(len(x), 0.03), "|", color=MUTED, alpha=0.3, ms=8, transform=ax_kaam.get_xaxis_transform())

        for ax in (ax_mlp, ax_kaam):
            ax.axhline(0, color=BASELINE, lw=0.8)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.set_xlabel(name, fontsize=9)
        ax_kaam.sharex(ax_mlp)
        ax_mlp.set_ylabel(f"{label}\nlog-odds vs. average", fontsize=9)
        if row == 0:
            ax_mlp.set_title("MLP: SHAP dependence (post hoc, test tasks)", loc="left", fontsize=11)
            ax_kaam.set_title("KAAM: φ curve (built in; band = 5th–95th pct)", loc="left", fontsize=11)
    fig.tight_layout()
    return fig
