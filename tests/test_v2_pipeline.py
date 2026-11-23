"""v2 checks: train-only preprocessing, the four protocols, KAAM fitted on its basis, and the metric code."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from kanvas import models_v2 as M
from kanvas.data import FEATURE_NAMES
from kanvas.metrics import (AucScorer, bootstrap_models, bootstrap_p, brier, ece, holm, pr_auc, reliability,
                            resample_weights, roc_auc)
from kanvas.preprocess import fit_preprocessor
from kanvas.protocols import (check_split, grouped_folds, known_user_mask, leave_users_out, random_split,
                              time_split)


def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.lognormal(0, 1, n) * (i + 1) for i, c in enumerate(FEATURE_NAMES)})
    y = (rng.random(n) < 0.27).astype(int)
    user = rng.integers(0, 60, n)
    user[:600] = 999  # one dominant user
    start = rng.uniform(0, 1e6, n)
    return X, y, user, start


# ----------------------------------------------------------------------------- preprocessing
def test_clipping_and_scaling_use_training_rows_only():
    X, y, _, _ = _frame()
    split = random_split(y, 42)
    X_train = X.iloc[split["train"]]
    pp = fit_preprocessor(X_train)
    for c in FEATURE_NAMES:
        lo, hi = X_train[c].quantile([0.01, 0.99])
        assert pp.clip_bounds[c] == (pytest.approx(lo), pytest.approx(hi))
    # moving test rows to extreme values must not change anything fitted
    X_shifted = X.copy()
    X_shifted.iloc[split["test"]] = 1e9
    pp2 = fit_preprocessor(X_shifted.iloc[split["train"]])
    assert pp2.clip_bounds == pp.clip_bounds and pp2.ranges == pp.ranges
    # training values land in [0, 1]; a test value above the training range is clipped to 1
    scaled_train = pp.transform(X_train)
    assert scaled_train.min().min() >= 0 and scaled_train.max().max() <= 1 + 1e-12
    assert np.allclose(pp.transform(X_shifted.iloc[split["test"]]).to_numpy(), 1.0)


def test_constant_training_feature_keeps_scaling_defined():
    X, _, _, _ = _frame(200)
    X["cap_gpu"] = 8.0
    pp = fit_preprocessor(X)
    assert np.isfinite(pp.transform(X).to_numpy()).all()


# ----------------------------------------------------------------------------- protocols
def test_random_split_is_stratified_70_15_15_and_disjoint():
    _, y, _, _ = _frame()
    s = random_split(y, 42)
    check_split(s, n_total=len(y))
    assert len(s["test"]) == int(np.ceil(0.15 * len(y))) and len(s["val"]) == int(np.ceil(0.15 * len(y)))
    for part in s.values():
        assert abs(y[part].mean() - y.mean()) < 0.01


def test_time_split_orders_train_before_val_before_test():
    _, y, _, start = _frame()
    s = time_split(start)
    check_split(s, n_total=len(y))
    assert start[s["train"]].max() <= start[s["val"]].min()
    assert start[s["val"]].max() <= start[s["test"]].min()
    assert len(s["train"]) == int(np.floor(0.7 * len(y)))


def test_time_split_breaks_ties_stably():
    start = np.array([5.0, 1.0, 1.0, 1.0, 9.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    s = time_split(start)
    assert s["train"].tolist() == [1, 2, 3, 5, 6, 7, 8]


def test_known_user_mask():
    user = np.array([1, 1, 2, 3, 3, 4])
    s = {"train": np.array([0, 2]), "val": np.array([3]), "test": np.array([1, 4, 5])}
    assert known_user_mask(user, s).tolist() == [True, False, False]


def test_grouped_folds_keep_every_user_in_one_part():
    _, y, user, _ = _frame()
    folds = grouped_folds(y, user)
    assert len(folds) == 5
    tests = np.concatenate([f["test"] for f in folds])
    assert np.array_equal(np.sort(tests), np.arange(len(y)))  # the test folds partition the data
    for f in folds:
        check_split(f, n_total=len(y), groups=user)


def test_leave_users_out_removes_every_task_of_the_users():
    _, y, user, _ = _frame()
    s = leave_users_out(y, user, [999])
    allpos = np.concatenate(list(s.values()))
    assert not np.isin(user[allpos], [999]).any()
    assert len(allpos) == int((user != 999).sum())
    check_split(s)


# ----------------------------------------------------------------------------- KAAM on its basis
def test_kaam_fitted_on_basis_matches_module_forward():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.random((n, 10)).astype(np.float32)
    logit = 2 * np.sin(6 * x[:, 0]) - 1.5 * x[:, 1] + (x[:, 8] > 0.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    template = M.KAAM(FEATURE_NAMES, M.UNIT_RANGES, num_knots=M.NUM_KNOTS, spline_order=M.SPLINE_ORDER)
    dtr, dva = M.kaam_design(template, x[:2400]), M.kaam_design(template, x[2400:])
    fit = M.fit_kaam(dtr, y[:2400], dva, y[2400:], seed=0, lam=1e-3)
    with torch.no_grad():
        p_module = fit.model(torch.as_tensor(x[2400:])).numpy()
    assert np.allclose(fit.predict(dva), p_module, atol=1e-6)
    assert fit.info["parameters"] == 151
    assert fit.info["early_stopped"] or fit.info["epochs_run"] == M.MAX_EPOCHS
    # explain() still decomposes the module's logit exactly
    xi = torch.as_tensor(x[2400])
    contrib = fit.model.explain(xi)
    logit_sum = sum(contrib.values()) + fit.model.bias.item()
    assert 1 / (1 + np.exp(-logit_sum)) == pytest.approx(float(p_module[0]), abs=1e-6)


def test_kaam_state_round_trip():
    template = M.KAAM(FEATURE_NAMES, M.UNIT_RANGES)
    clone = M.load_kaam(M.kaam_state(template))
    x = torch.rand(50, 10)
    assert torch.allclose(template(x), clone(x))


# ----------------------------------------------------------------------------- metrics
def test_auc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500), 1)  # many ties
    assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p))
    w = rng.integers(0, 3, 500).astype(float)
    assert AucScorer(y, p)(w) == pytest.approx(roc_auc_score(y, p, sample_weight=w))


def test_pr_auc_brier_ece_reliability():
    rng = np.random.default_rng(0)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)  # perfectly calibrated in expectation
    assert pr_auc(y, p) == pytest.approx(average_precision_score(y, p))
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert ece(y, p) < 0.03
    assert ece(y, np.clip(p + 0.3, 0, 1)) > 0.2
    r = reliability(y, p)
    assert r["count"].sum() == 4000 and len(r["count"]) == 15


def test_cluster_resample_moves_whole_clusters():
    rng = np.random.default_rng(0)
    clusters = np.array([0, 0, 0, 1, 2, 2])
    w = resample_weights(6, rng, clusters)
    assert w[0] == w[1] == w[2] and w[4] == w[5]


def test_paired_bootstrap_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    good = y + rng.normal(0, 0.8, 3000)
    bad = y + rng.normal(0, 3.0, 3000)
    same = good + rng.normal(0, 1e-6, 3000)
    point, boot = bootstrap_models([{"y": y, "scores": {"good": [good], "bad": [bad], "same": [same]}}],
                                   n_boot=300)
    assert point["good"] > point["bad"]
    assert bootstrap_p(boot["good"] - boot["bad"]) < 0.01
    assert bootstrap_p(boot["good"] - boot["same"]) > 0.05


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, 0.20])
    assert adj.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.20])


# ----------------------------------------------------------------------------- early stopping (deviation 1)
def _shifted_data(seed=0):
    """Training failure rate about 0.2, validation about 0.7, with the same signal in both."""
    rng = np.random.default_rng(seed)
    x = rng.random((6000, 10)).astype(np.float32)
    score = 3 * x[:, 0] - 2 * x[:, 1]
    y = np.empty(6000, dtype=int)
    y[:4000] = (rng.random(4000) < 1 / (1 + np.exp(-(score[:4000] - 2.2)))).astype(int)
    y[4000:] = (rng.random(2000) < 1 / (1 + np.exp(-(score[4000:] + 0.6)))).astype(int)
    return x[:4000], y[:4000], x[4000:], y[4000:]


def test_early_stopping_monitors_validation_auc_and_restores_the_best_epoch():
    xtr, ytr, xva, yva = _shifted_data()
    assert ytr.mean() < 0.35 and yva.mean() > 0.55
    fit = M.fit_mlp(xtr, ytr, xva, yva, seed=0, hidden=(8, 6), lr=0.01)
    hist = fit.info["val_auc_history"]
    assert len(hist) == fit.info["epochs_run"]
    assert fit.info["best_val_auc"] == pytest.approx(max(hist))
    assert hist[fit.info["best_epoch"] - 1] == pytest.approx(max(hist))
    assert roc_auc(yva, fit.predict(xva)) == pytest.approx(max(hist), abs=1e-6)  # best epoch restored
    # under this label shift the old log-loss rule stopped at once; the AUC rule trains the network
    assert fit.info["best_epoch"] > 1 and max(hist) > 0.7
    assert fit.info["early_stopped"] or fit.info["epochs_run"] == M.MAX_EPOCHS


# ----------------------------------------------------------------------------- runner queue
def test_queue_claim_is_exclusive(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_experiments import claim

    assert claim(tmp_path) is True
    assert claim(tmp_path) is False  # a second worker cannot take the same split
    (tmp_path / ".lock").unlink()
    assert claim(tmp_path) is True
