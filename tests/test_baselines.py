"""Tests for the Phase 4 baselines (kanvas/baselines.py) and SHAP analysis (kanvas/shap_analysis.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kanvas.baselines import (MLP, MLP_HIDDEN_CANDIDATES, benchmark, count_parameters, kaam_parameter_breakdown,
                              lr_predict_proba, score_test_set, tune_logistic_regression, tune_mlp)
from kanvas.data import FEATURE_NAMES
from kanvas.model import KAAM
from kanvas.shap_analysis import kaam_curve_range_importance, mlp_shap_values, shap_importance
from kanvas.training import fit_feature_ranges, predict_proba, train_kaam, train_val_test_split
from kanvas.utils import set_seed

FEATURES_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "kanvas_features.csv"
UNIT_RANGES = {name: (0.0, 1.0) for name in FEATURE_NAMES}


def test_kaam_and_mlp_parameter_counts_are_within_20_percent():
    kaam = KAAM(FEATURE_NAMES, UNIT_RANGES, num_knots=10, spline_order=3)
    kaam_params = count_parameters(kaam)
    assert kaam_parameter_breakdown(kaam)["total"] == kaam_params  # nothing learnable left uncounted
    for hidden in MLP_HIDDEN_CANDIDATES:
        mlp_params = count_parameters(MLP(len(FEATURE_NAMES), hidden))
        assert abs(mlp_params - kaam_params) <= 0.2 * kaam_params, (hidden, mlp_params, kaam_params)


def test_all_three_models_are_scored_on_identical_test_rows():
    if not FEATURES_CSV.exists():
        pytest.skip(f"{FEATURES_CSV} not found")
    df = pd.read_csv(FEATURES_CSV, index_col=["job_name", "task_name"])
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df[FEATURE_NAMES], df["label"])

    set_seed(0)
    kaam = KAAM(FEATURE_NAMES, UNIT_RANGES)
    kaam_result = train_kaam(kaam, X_train, y_train, X_val, y_val, lambda_smooth=0.001, max_epochs=3, verbose=False)
    lr, lr_ranges, _, _ = tune_logistic_regression(X_train, y_train, X_val, y_val, c_grid=[1.0])
    mlp, mlp_result, _, _ = tune_mlp(X_train, y_train, X_val, y_val, hidden_candidates=[(12,)], lr_candidates=[0.01],
                                     max_epochs=3)

    predictions = score_test_set({
        "KAAM": lambda X: predict_proba(kaam, X, kaam_result.feature_ranges),
        "LR": lambda X: lr_predict_proba(lr, X, lr_ranges),
        "MLP": lambda X: predict_proba(mlp, X, mlp_result.feature_ranges),
    }, X_test)

    index_arrays = [p.index.to_numpy() for p in predictions.values()]
    for arr in index_arrays:
        assert np.array_equal(arr, index_arrays[0])
        assert np.array_equal(arr, y_test.index.to_numpy())

    benchmark(y_test, predictions, {"KAAM": 1, "LR": 1, "MLP": 1}, {"KAAM": 0.0, "LR": 0.0, "MLP": 0.0})
    shuffled = dict(predictions, LR=predictions["LR"].iloc[::-1])
    with pytest.raises(ValueError):
        benchmark(y_test, shuffled, {"KAAM": 1, "LR": 1, "MLP": 1}, {"KAAM": 0.0, "LR": 0.0, "MLP": 0.0})


def test_shap_importance_has_ten_values_in_feature_order():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 100, size=(20, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    ranges = fit_feature_ranges(X)
    set_seed(0)
    mlp = MLP(len(FEATURE_NAMES), (12,))

    values, expected = mlp_shap_values(mlp, X.iloc[:10], X.iloc[10:13], ranges)
    importance = shap_importance(values)
    assert len(importance) == 10
    assert list(importance.index) == FEATURE_NAMES

    # SHAP values must add up to each row's logit minus the background mean.
    with torch.no_grad():
        logits = mlp.net(torch.tensor(((X.iloc[10:13] - X.min()) / (X.max() - X.min())).to_numpy(np.float32))).numpy().ravel()
    assert np.allclose(values.sum(axis=1).to_numpy() + expected, logits, atol=1e-4)

    kaam_importance = kaam_curve_range_importance(KAAM(FEATURE_NAMES, UNIT_RANGES), X, ranges)
    assert list(kaam_importance.index) == FEATURE_NAMES
