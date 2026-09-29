"""Regularized training for KAAM: smoothness penalty, early stopping, threshold tuning.

Phase 2 trained KAAM for a fixed 1,000 epochs on plain BCE. On the Alibaba
trace that left two problems: spline curves swung wildly over x-ranges with
almost no training tasks, and the held-out loss bottomed out early and then
crept up with nothing to stop training. Everything here lives in the training
loop; BSplineEdge and KAAM are used exactly as they are.

Data roles: the training split fits the weights and the scaling ranges; the
validation split drives early stopping, the choice of lambda, and the
decision threshold; the test split is only for the final report.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import scale_features

# Decision thresholds swept on validation data: 0.10, 0.15, ..., 0.90.
THRESHOLDS = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)


def smoothness_penalty(model):
    """Sum of squared second differences of every edge's spline coefficients.

    On a uniform knot grid, ``c[i+1] - 2*c[i] + c[i-1]`` measures how sharply
    the curve bends at that point of the grid (the P-spline penalty of Eilers
    & Marx, 1996). A straight run of coefficients costs nothing, so slopes
    are free and only curvature is charged; the data has to justify every
    bend. Where no training tasks sit under part of the grid, nothing
    justifies a bend, and the curve stays close to straight there.

    Only spline coefficients are penalized: each edge's SiLU base weight and
    the model bias are left free.

    Args:
        model: A KAAM whose ``edges`` ModuleDict holds BSplineEdge modules.

    Returns:
        0-dim tensor, differentiable with respect to the coefficients.
    """
    total = torch.zeros(())
    for edge in model.edges.values():
        coef = edge.spline_weight
        second_diff = coef[2:] - 2 * coef[1:-1] + coef[:-2]
        total = total + (second_diff**2).sum()
    return total


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """Stratified train/validation/test split (70/15/15 by default).

    The test rows are split off first, then the validation rows from the
    remainder; both splits are stratified on the label, so all three parts
    keep the full data's failure rate.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    n_test, n_val = int(np.ceil(test_size * len(X))), int(np.ceil(val_size * len(X)))
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=n_test, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=n_val, random_state=random_state,
                                                      stratify=y_rest)
    return X_train, X_val, X_test, y_train, y_val, y_test


@dataclass
class TrainResult:
    """What a training run produced besides the weights it leaves in the model."""

    feature_ranges: dict  # per-feature (min, max) of X_train, used to scale every input
    best_epoch: int  # epoch whose weights were restored
    stopped_epoch: int  # last epoch actually run
    early_stopped: bool  # True if patience ran out before max_epochs
    best_val_loss: float  # validation BCE at best_epoch
    history: dict = field(default_factory=dict)  # per-epoch lists: train_loss, train_bce, penalty, val_loss


def fit_feature_ranges(X):
    """Per-feature ``(min, max)`` of ``X``: the min-max scaling every model here is trained on.

    Fit it on the training split only, then pass it to ``scale_features``
    for every split, so no model sees validation or test data through its
    input scaling.
    """
    return {name: (float(X[name].min()), float(X[name].max())) for name in X.columns}


def train_early_stopping(model, X_train, y_train, X_val, y_val, penalty_fn=None, lambda_penalty=0.0, patience=15,
                         max_epochs=500, lr=0.01, verbose=True, print_every=20):
    """Train any binary classifier module, early-stopping on validation loss.

    ``model`` can be any ``nn.Module`` that maps a ``(n, n_features)`` float
    tensor to ``(n,)`` failure probabilities. Inputs are min-max scaled to
    [0, 1] with ``scale_features``, using ranges fitted on ``X_train`` only.

    Each epoch is one full-batch Adam step on
    ``BCE(train) + lambda_penalty * penalty_fn(model)`` (plain BCE when
    ``penalty_fn`` is None). The validation loss is plain BCE, the quantity
    that should generalize. Once it has not improved for ``patience``
    consecutive epochs, training stops and the weights from the best
    validation epoch are restored, not the last epoch's.

    Args:
        model: Module mapping scaled features to probabilities, shape (n,).
        X_train, X_val: Raw-unit feature DataFrames with the same columns.
        y_train, y_val: 0/1 labels.
        penalty_fn: Optional ``penalty_fn(model) -> 0-dim tensor`` added to the training loss.
        lambda_penalty: Weight of ``penalty_fn`` in the training loss.
        patience: Epochs without validation improvement before stopping.
        max_epochs: Hard cap on the number of epochs.
        lr: Adam learning rate.
        verbose: Print progress every ``print_every`` epochs, and the stop reason.
        print_every: Progress print interval, in epochs.

    Returns:
        TrainResult. The model is modified in place and ends up holding the
        best-validation weights.
    """
    _check_columns(list(X_train.columns), X_val)
    feature_ranges = fit_feature_ranges(X_train)
    x_train = _to_tensor(scale_features(X_train, feature_ranges))
    x_val = _to_tensor(scale_features(X_val, feature_ranges))
    t_train, t_val = _to_tensor(y_train), _to_tensor(y_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    history = {"train_loss": [], "train_bce": [], "penalty": [], "val_loss": []}
    best_val_loss, best_epoch, best_state, since_best = float("inf"), 0, None, 0

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        train_bce = bce(model(x_train), t_train)
        if penalty_fn is None:
            penalty, loss = None, train_bce
        else:
            penalty = penalty_fn(model)
            loss = train_bce + lambda_penalty * penalty
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_loss = bce(model(x_val), t_val).item()
        history["train_loss"].append(loss.item())
        history["train_bce"].append(train_bce.item())
        history["penalty"].append(0.0 if penalty is None else penalty.item())
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, since_best = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1

        if verbose and epoch % print_every == 0:
            detail = "" if penalty is None else (
                f" (BCE {train_bce.item():.4f} + lambda*penalty {lambda_penalty * penalty.item():.4f})")
            print(f"Epoch {epoch:4d}  train loss {loss.item():.4f}{detail}  val loss {val_loss:.4f}")
        if since_best >= patience:
            break

    early_stopped = since_best >= patience
    model.load_state_dict(best_state)
    if verbose:
        if early_stopped:
            print(f"Early stopping triggered at epoch {epoch}: val loss has not improved for {patience} epochs. "
                  f"Restored weights from epoch {best_epoch} (val loss {best_val_loss:.4f}).")
        else:
            print(f"Reached max_epochs={max_epochs} without early stopping. "
                  f"Restored weights from best epoch {best_epoch} (val loss {best_val_loss:.4f}).")
    return TrainResult(feature_ranges, best_epoch, epoch, early_stopped, best_val_loss, history)


def train_kaam(model, X_train, y_train, X_val, y_val, lambda_smooth, patience=15, max_epochs=500, lr=0.01,
               verbose=True, print_every=20):
    """Train KAAM on BCE + lambda_smooth * smoothness_penalty with early stopping.

    ``train_early_stopping`` with the KAAM-specific checks: KAAM reads
    feature i from column i and names its edges, so the columns must match
    ``model.feature_names`` in order, and every edge must be built on the
    unit range that the [0, 1] input scaling produces.

    Args:
        model: KAAM built with ``(0, 1)`` ranges for every feature.
        X_train, X_val, y_train, y_val: As for ``train_early_stopping``.
        lambda_smooth: Weight of the smoothness penalty (0 disables it).
        patience, max_epochs, lr, verbose, print_every: As for ``train_early_stopping``.

    Returns:
        TrainResult. The model ends up holding the best-validation weights.
    """
    _check_columns(model.feature_names, X_train)
    off_unit = [name for name, edge in model.edges.items() if (edge.x_min, edge.x_max) != (0.0, 1.0)]
    if off_unit:
        raise ValueError(f"train_kaam scales inputs to [0, 1], so build KAAM with (0, 1) ranges; not so for {off_unit}")
    return train_early_stopping(model, X_train, y_train, X_val, y_val, penalty_fn=smoothness_penalty,
                                lambda_penalty=lambda_smooth, patience=patience, max_epochs=max_epochs, lr=lr,
                                verbose=verbose, print_every=print_every)


def predict_proba(model, X, feature_ranges):
    """Failure probabilities for raw-unit ``X``, scaled with the ``feature_ranges`` fitted at training time."""
    _check_columns(list(feature_ranges), X)
    with torch.no_grad():
        return model(_to_tensor(scale_features(X, feature_ranges))).numpy()


def tune_threshold(y_val, p_val, thresholds=THRESHOLDS):
    """Pick the decision threshold that maximizes F1 on validation data.

    The signature takes validation labels and probabilities only; there is
    deliberately no way to pass test data in.

    Args:
        y_val: Validation 0/1 labels.
        p_val: Predicted failure probabilities for the validation rows.
        thresholds: Candidate thresholds; a task is flagged when p >= threshold.

    Returns:
        ``(best_threshold, table)``: the F1-maximizing threshold (ties go to
        the lowest) and a DataFrame of precision, recall and F1 per threshold.
        Precision is NaN where a threshold flags no task at all.
    """
    rows = []
    for threshold in thresholds:
        flagged = p_val >= threshold
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(y_val, flagged, zero_division=np.nan),
            "recall": recall_score(y_val, flagged, zero_division=0),
            "f1": f1_score(y_val, flagged, zero_division=0),
        })
    table = pd.DataFrame(rows)
    return float(table.loc[table["f1"].idxmax(), "threshold"]), table


def classification_metrics(y_true, p, threshold=0.5):
    """Accuracy, ROC-AUC, precision (NaN if nothing is flagged), recall, F1 and confusion counts."""
    flagged = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, flagged),
        "auc": roc_auc_score(y_true, p),
        "precision": precision_score(y_true, flagged, zero_division=np.nan),
        "recall": recall_score(y_true, flagged, zero_division=0),
        "f1": f1_score(y_true, flagged, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _check_columns(expected, X):
    # Models read feature i from column i, so a reordered DataFrame would
    # silently feed one feature's values into another feature's slot.
    if list(X.columns) != list(expected):
        raise ValueError(f"X columns {list(X.columns)} must match {list(expected)} in order")


def _to_tensor(values):
    return torch.tensor(np.asarray(values, dtype=np.float32))
