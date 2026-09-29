"""Baselines for the KANVAS benchmark: logistic regression and a capacity-matched MLP.

Three models, three kinds of explanation:

- Logistic regression is additive and interpretable, but each feature's
  effect is a straight line.
- KAAM is additive and interpretable, and each feature's curve can take any
  smooth shape the data supports.
- An MLP is neither additive nor interpretable on its own; explaining it
  takes a post-hoc tool such as SHAP.

All three see the same [0, 1]-scaled features (ranges fitted on the training
split only) and the same train/validation/test split. Every hyperparameter
is chosen on validation data, and the test split is only scored by
``benchmark``.
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data import scale_features
from .training import classification_metrics, fit_feature_ranges, predict_proba, train_early_stopping
from .utils import set_seed

# Validation-only search spaces, fixed before any model was scored.
LR_C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
MLP_HIDDEN_CANDIDATES = [(12,), (14,), (8, 6), (8, 8)]  # all within 20% of KAAM's 151 parameters
MLP_LR_CANDIDATES = [0.003, 0.01, 0.03]


class MLP(nn.Module):
    """ReLU multilayer perceptron with a single sigmoid output: the black-box baseline."""

    def __init__(self, n_inputs, hidden):
        super().__init__()
        layers, width = [], n_inputs
        for units in hidden:
            layers += [nn.Linear(width, units), nn.ReLU()]
            width = units
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)  # maps scaled features to the failure logit

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def count_parameters(model):
    """Number of learnable scalars in a torch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kaam_parameter_breakdown(model):
    """KAAM's learnable parameters by kind; the knot grids are fixed buffers and are not counted."""
    spline = sum(edge.spline_weight.numel() for edge in model.edges.values())
    base = sum(edge.base_weight.numel() for edge in model.edges.values())
    bias = model.bias.numel()
    return {"spline coefficients": spline, "base weights": base, "global bias": bias, "total": spline + base + bias}


def tune_logistic_regression(X_train, y_train, X_val, y_val, c_grid=LR_C_GRID):
    """L2 logistic regression on [0, 1]-scaled features, with C chosen by validation AUC.

    Each candidate is fitted on the training split only; ties in validation
    AUC go to the smallest C (the strongest regularization).

    Returns:
        ``(model, feature_ranges, seconds, table)``: the model at the best C,
        the scaling ranges it was trained with, its fit time, and the
        validation AUC for every C.
    """
    feature_ranges = fit_feature_ranges(X_train)
    x_train = scale_features(X_train, feature_ranges).to_numpy()
    x_val = scale_features(X_val, feature_ranges).to_numpy()
    rows, fitted = [], {}
    for c in c_grid:
        start = time.perf_counter()
        # The default penalty is L2 (penalty="l2" before sklearn 1.8, l1_ratio=0 after).
        model = LogisticRegression(C=c, max_iter=10_000).fit(x_train, y_train)
        fitted[c] = (model, time.perf_counter() - start)
        rows.append({"C": c, "val AUC": roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])})
    table = pd.DataFrame(rows)
    best_c = float(table.loc[table["val AUC"].idxmax(), "C"])
    model, seconds = fitted[best_c]
    return model, feature_ranges, seconds, table


def lr_predict_proba(model, X, feature_ranges):
    """Failure probabilities from a fitted logistic regression, for raw-unit ``X``."""
    if list(X.columns) != list(feature_ranges):
        raise ValueError(f"X columns {list(X.columns)} must match {list(feature_ranges)} in order")
    return model.predict_proba(scale_features(X, feature_ranges).to_numpy())[:, 1]


def tune_mlp(X_train, y_train, X_val, y_val, hidden_candidates=MLP_HIDDEN_CANDIDATES,
             lr_candidates=MLP_LR_CANDIDATES, patience=15, max_epochs=500, seed=42):
    """Train every (architecture, learning rate) pair with early stopping; keep the best validation AUC.

    Each run uses ``train_early_stopping``, the same loop, patience and
    scaling as KAAM, starting from ``set_seed(seed)``.

    Returns:
        ``(model, result, seconds, table)``: the selected MLP (holding its
        best-validation weights), its TrainResult, its training time, and
        one row per candidate.
    """
    rows, runs = [], {}
    for hidden in hidden_candidates:
        for lr in lr_candidates:
            set_seed(seed)
            model = MLP(X_train.shape[1], hidden)
            start = time.perf_counter()
            result = train_early_stopping(model, X_train, y_train, X_val, y_val, patience=patience,
                                          max_epochs=max_epochs, lr=lr, verbose=False)
            seconds = time.perf_counter() - start
            val_auc = roc_auc_score(y_val, predict_proba(model, X_val, result.feature_ranges))
            runs[(hidden, lr)] = (model, result, seconds)
            rows.append({"hidden": hidden, "lr": lr, "parameters": count_parameters(model),
                         "best epoch": result.best_epoch, "stopped at": result.stopped_epoch,
                         "val BCE": result.best_val_loss, "val AUC": val_auc})
    table = pd.DataFrame(rows)
    best = table.loc[table["val AUC"].idxmax()]
    model, result, seconds = runs[(best["hidden"], best["lr"])]
    return model, result, seconds, table


def score_test_set(scorers, X_test):
    """Failure probabilities from every model for the same test rows.

    Args:
        scorers: Dict of model name -> function mapping a raw-unit DataFrame to probabilities.
        X_test: The one test DataFrame every model is scored on.

    Returns:
        Dict of model name -> Series of probabilities indexed like ``X_test``.
    """
    return {name: pd.Series(np.asarray(score(X_test)), index=X_test.index, name=name) for name, score in scorers.items()}


def benchmark(y_test, predictions, parameters, train_seconds, threshold=0.5):
    """Score every model on the same test rows with the same metrics.

    Args:
        y_test: Test labels, a Series indexed by task.
        predictions: Dict of model name -> Series of test failure
            probabilities. Every Series must be indexed exactly like ``y_test``.
        parameters: Dict of model name -> learnable parameter count.
        train_seconds: Dict of model name -> training wall-clock time.
        threshold: Decision threshold for accuracy, precision, recall and F1.

    Returns:
        DataFrame with one row per model.

    Raises:
        ValueError: if any model was scored on different rows, or in a
            different order, than ``y_test``.
    """
    expected = y_test.index.to_numpy()
    for name, p in predictions.items():
        if not np.array_equal(p.index.to_numpy(), expected):
            raise ValueError(f"{name} was not scored on exactly the test rows of y_test, in the same order")

    rows = {}
    for name, p in predictions.items():
        m = classification_metrics(y_test.to_numpy(), p.to_numpy(), threshold)
        lo, hi = bootstrap_auc_ci(y_test.to_numpy(), p.to_numpy())
        rows[name] = {"accuracy": m["accuracy"], "AUC": m["auc"], "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                      "precision": m["precision"], "recall": m["recall"], "F1": m["f1"],
                      "Brier": brier_score_loss(y_test, p), "parameters": parameters[name],
                      "train seconds": train_seconds[name]}
    return pd.DataFrame(rows).T


def paired_auc_differences(y_test, predictions, pairs, n_boot=2000, seed=0):
    """AUC(a) - AUC(b) on the test set, with a paired bootstrap 95% interval, for each (a, b) pair."""
    y = y_test.to_numpy()
    rows = []
    for a, b in pairs:
        pa, pb = predictions[a].to_numpy(), predictions[b].to_numpy()
        lo, hi = paired_bootstrap_auc_diff(y, pa, pb, n_boot=n_boot, seed=seed)
        rows.append({"comparison": f"{a} - {b}", "AUC difference": roc_auc_score(y, pa) - roc_auc_score(y, pb),
                     "95% CI low": lo, "95% CI high": hi, "CI includes 0": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows).set_index("comparison")


def bootstrap_auc_ci(y_true, p, n_boot=2000, seed=0):
    """95% percentile interval for ROC-AUC, resampling test rows with replacement."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() != y_true[idx].max():
            aucs.append(roc_auc_score(y_true[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def paired_bootstrap_auc_diff(y_true, p_a, p_b, n_boot=2000, seed=0):
    """95% interval for AUC(p_a) - AUC(p_b), scoring both models on the same resampled rows."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() != y_true[idx].max():
            diffs.append(roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(y_true[idx], p_b[idx]))
    return np.percentile(diffs, [2.5, 97.5])
