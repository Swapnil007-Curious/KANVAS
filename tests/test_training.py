"""Tests for regularized KAAM training (kanvas/training.py)."""

import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from kanvas.model import KAAM
from kanvas.training import predict_proba, smoothness_penalty, train_kaam, train_val_test_split, tune_threshold
from kanvas.utils import set_seed

NAMES = ["feature_1", "feature_2", "feature_3"]
UNIT_RANGES = {name: (0.0, 1.0) for name in NAMES}


def _set_linear_coefficients(model):
    """Give every edge a straight-line coefficient vector, with a different line per edge."""
    with torch.no_grad():
        for k, edge in enumerate(model.edges.values()):
            index = torch.arange(edge.spline_weight.numel(), dtype=torch.float32)
            edge.spline_weight.copy_((0.5 + k) * index - k)


def test_smoothness_penalty_is_positive_scalar_for_curved_coefficients():
    set_seed(0)
    model = KAAM(NAMES, UNIT_RANGES)
    assert smoothness_penalty(model).item() > 0  # the default random init has curvature

    _set_linear_coefficients(model)
    with torch.no_grad():
        model.edges["feature_2"].spline_weight[5] += 1.0  # one spike: second differences 1, -2, 1
    penalty = smoothness_penalty(model)
    assert penalty.dim() == 0
    assert penalty.item() == pytest.approx(6.0)

    penalty.backward()
    assert model.edges["feature_2"].spline_weight.grad.abs().sum() > 0


def test_smoothness_penalty_is_near_zero_for_linear_coefficients():
    model = KAAM(NAMES, UNIT_RANGES)
    _set_linear_coefficients(model)
    assert smoothness_penalty(model).item() == pytest.approx(0.0, abs=1e-8)


def test_early_stopping_restores_best_validation_weights():
    """Pure-noise labels make validation loss rise; the returned weights must be the best-validation ones."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 10, size=(120, 3)), columns=NAMES)
    y = pd.Series(rng.integers(0, 2, size=120))
    X_train, X_val, y_train, y_val = X[:60], X[60:], y[:60], y[60:]

    set_seed(0)
    model = KAAM(NAMES, UNIT_RANGES)
    result = train_kaam(model, X_train, y_train, X_val, y_val, lambda_smooth=0.0, patience=5, max_epochs=500,
                        lr=0.05, verbose=False)

    assert result.early_stopped
    assert result.best_epoch < result.stopped_epoch
    p_val = torch.tensor(predict_proba(model, X_val, result.feature_ranges))
    restored_val_loss = torch.nn.functional.binary_cross_entropy(p_val, torch.tensor(y_val.to_numpy(np.float32))).item()
    assert restored_val_loss <= result.history["val_loss"][-1]
    assert restored_val_loss == pytest.approx(min(result.history["val_loss"]), abs=1e-6)


def test_threshold_tuning_accepts_only_validation_data():
    """Structural guarantee: there is no parameter through which test data could reach threshold selection."""
    params = list(inspect.signature(tune_threshold).parameters)
    assert params == ["y_val", "p_val", "thresholds"]
    assert not any("test" in name for name in params)


def test_threshold_tuning_picks_the_f1_maximizing_threshold():
    y_val = np.array([0, 0, 0, 1, 1, 1])
    p_val = np.array([0.05, 0.20, 0.30, 0.42, 0.60, 0.90])
    best, table = tune_threshold(y_val, p_val)
    # F1 = 1 only for thresholds in (0.30, 0.42], i.e. 0.35 and 0.40; ties go to the lowest.
    assert best == pytest.approx(0.35)
    assert table["f1"].max() == pytest.approx(1.0)


def test_split_is_70_15_15_stratified_and_disjoint():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=NAMES)
    y = pd.Series((rng.uniform(size=1000) < 0.28).astype(int))
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, random_state=42)

    assert (len(X_train), len(X_val), len(X_test)) == (700, 150, 150)
    assert not set(X_test.index) & (set(X_train.index) | set(X_val.index))
    assert not set(X_train.index) & set(X_val.index)
    for part in (y_train, y_val, y_test):
        assert part.mean() == pytest.approx(y.mean(), abs=0.01)
