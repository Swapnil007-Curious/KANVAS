"""Tests for Phase 6: the lookback-window ablation and the case-level explain() audit.

The case-study tests run on the real canonical model: they load
``models/kaam_canonical.pt`` (written by notebook 06) when it exists, and
otherwise rebuild the model by Phase 5's rule, which takes about a minute.
Either way the model must reproduce Phase 5's exported test AUC exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from kanvas import stability
from kanvas.ablation import (LOOKBACK_HOURS, anchored_splits, coverage_table, curve_agreement, importance_by_width,
                             lookback_paths, nesting_report, parse_build_log, results_table, run_lookback_ablation,
                             telemetry_curves, train_across_widths)
from kanvas.case_studies import (CASE_ORDER, consistency_check, explain_case, file_sha256, forward_logits,
                                 load_checkpoint, phi_training_means, reproduce_canonical_model, select_cases)
from kanvas.data import FEATURE_NAMES
from kanvas.shap_analysis import kaam_shap_values
from kanvas.stability import CANONICAL_SPLIT_STATE, THRESHOLD
from kanvas.training import predict_proba, train_val_test_split

ROOT = Path(__file__).resolve().parents[1]
LARGE_CSV = ROOT / "data" / "processed" / "kanvas_features_large.csv"
LOOKBACK_DIR = ROOT / "data" / "processed" / "lookback"
PHASE5_BENCHMARK = ROOT / "data" / "processed" / "publication" / "kanvas_benchmark_results.csv"
CHECKPOINT = ROOT / "models" / "kaam_canonical.pt"

# Phase 5 scored Table 1 at 0.5 (kanvas/stability.py: THRESHOLD = 0.5, "not re-tuned per run");
# Phase 3's validation-tuned 0.60 was never carried forward.
CANONICAL_THRESHOLD = 0.5


def _require(path):
    if not path.exists():
        pytest.fail(f"{path} is missing; run notebooks/05_scale_and_stability.ipynb and 06 first")
    return path


@pytest.fixture(scope="module")
def large():
    df = pd.read_csv(_require(LARGE_CSV), index_col=["job_name", "task_name"])
    return df[FEATURE_NAMES], df["label"]


@pytest.fixture(scope="module")
def canonical(large):
    """The Phase 5 canonical model with its training-time scaling ranges, split and test probabilities."""
    if CHECKPOINT.exists():
        model, ranges, splits, _ = load_checkpoint(CHECKPOINT, LARGE_CSV)
    else:
        X, y = large
        model, result, splits = reproduce_canonical_model(X, y)
        ranges = result.feature_ranges
    X_te, y_te = splits[2], splits[5]
    p = pd.Series(predict_proba(model, X_te, ranges), index=X_te.index)
    return model, ranges, splits, p


@pytest.fixture(scope="module")
def cases(canonical):
    model, ranges, splits, p = canonical
    return select_cases(splits[5], p, threshold=CANONICAL_THRESHOLD)


def _stand_in_datasets(X, y):
    """Five nested row sets, one per width, cut from the large dataset: shape-only stand-ins for the real builds."""
    return {width: (X.head(400 + 20 * i), y.head(400 + 20 * i)) for i, width in enumerate(LOOKBACK_HOURS)}


# ------------------------------------------------------------------ Part A
def test_lookback_ablation_returns_exactly_one_result_per_width(monkeypatch, large):
    # Two epochs per run: this test is about the shape of the output, not the numbers.
    monkeypatch.setattr(stability, "MAX_EPOCHS", 2)
    monkeypatch.setattr(stability, "PATIENCE", 1)
    datasets = _stand_in_datasets(*large)

    out = run_lookback_ablation(datasets, verbose=False)

    assert len(LOOKBACK_HOURS) == 5
    for name in ("performance", "paired"):
        assert list(out[name].index) == LOOKBACK_HOURS, f"{name} does not have exactly one row per width"
    assert list(out["fitted"]) == LOOKBACK_HOURS and list(out["splits"]) == LOOKBACK_HOURS
    assert out["performance"][["test AUC", "test accuracy", "test F1"]].notna().all().all()
    coverage = coverage_table(datasets)
    assert list(coverage.index) == LOOKBACK_HOURS
    assert len(results_table(coverage, out["performance"], out["paired"])) == 5

    curves = telemetry_curves(out["fitted"], out["splits"])
    assert sorted(curves["lookback_hours"].unique()) == LOOKBACK_HOURS
    assert len(curve_agreement(out["fitted"], out["splits"])) == 2 * (len(LOOKBACK_HOURS) - 1)
    shares, ranks, rho = importance_by_width(out["fitted"], out["splits"])
    assert list(shares.columns) == LOOKBACK_HOURS and len(rho) == 5


def test_lookback_ablation_refuses_to_report_fewer_than_five_widths(monkeypatch, large):
    monkeypatch.setattr(stability, "MAX_EPOCHS", 2)
    datasets = _stand_in_datasets(*large)
    del datasets[6.0]
    with pytest.raises(ValueError, match="6"):
        run_lookback_ablation(datasets, verbose=False)
    splits = anchored_splits(datasets)
    with pytest.raises(ValueError, match="6"):
        train_across_widths(splits, LOOKBACK_HOURS, verbose=False)


def test_anchored_splits_keep_the_canonical_split_and_one_role_per_task(large):
    datasets = _stand_in_datasets(*large)
    splits = anchored_splits(datasets)

    # The canonical width's split is Phase 5's split: same tasks, same order.
    X_c, y_c = datasets[1.0]
    expected = train_val_test_split(X_c, y_c, random_state=CANONICAL_SPLIT_STATE)
    for got, want in zip(splits[1.0], expected):
        assert got.index.equals(want.index)

    roles = {}
    for width, parts in splits.items():
        X, _ = datasets[width]
        assert sum(len(part) for part in parts[:3]) == len(X), f"{width:g}h: roles do not partition the dataset"
        for role, part in zip(("train", "val", "test"), parts[:3]):
            for task in part.index:
                assert roles.setdefault(task, role) == role, f"{task} changes role at {width:g}h"


def test_saved_build_logs_parse_to_the_funnel_counts():
    csv_path, log_path = lookback_paths(str(LOOKBACK_DIR), 1.0)
    if not Path(log_path).exists():
        pytest.skip("lookback datasets not built yet (notebook 06)")
    funnel = parse_build_log(Path(log_path).read_text(encoding="utf-8"))
    assert funnel["sampled"] == 40_000
    assert funnel["telemetry found"] == 27_717  # Phase 5's funnel, as notebook 05 printed it
    assert funnel["clean rows"] == 18_890
    assert funnel["no worker in window"] + funnel["no telemetry, other reasons"] == 40_000 - 27_717


def test_lookback_datasets_nest_and_the_1h_build_is_the_phase_5_dataset():
    paths = [Path(lookback_paths(str(LOOKBACK_DIR), width)[0]) for width in LOOKBACK_HOURS]
    if not all(path.exists() for path in paths):
        pytest.skip("lookback datasets not built yet (notebook 06)")
    assert file_sha256(paths[LOOKBACK_HOURS.index(1.0)]) == file_sha256(_require(LARGE_CSV))
    datasets = {}
    for width, path in zip(LOOKBACK_HOURS, paths):
        df = pd.read_csv(path, index_col=["job_name", "task_name"])
        datasets[width] = (df[FEATURE_NAMES], df["label"])
    report = nesting_report(datasets)
    assert report["nested"].all() and (report["label disagreements"] == 0).all()


# ------------------------------------------------------------------ Part B
def test_canonical_threshold_is_the_one_phase_5_treated_as_final():
    assert THRESHOLD == CANONICAL_THRESHOLD


def test_canonical_model_reproduces_phase_5_test_auc(canonical):
    model, ranges, splits, p = canonical
    phase5 = pd.read_csv(_require(PHASE5_BENCHMARK)).set_index("model").loc["KAAM", "AUC"]
    auc = roc_auc_score(splits[5], p)
    assert abs(auc - phase5) < 1e-9, f"reproduced test AUC {auc:.10f} != Phase 5's {phase5:.10f}"


def test_explain_plus_bias_equals_the_forward_logit_for_all_four_cases(canonical, cases):
    model, ranges, splits, p = canonical
    checks = consistency_check(model, splits[2], ranges, list(cases["task"]), labels=cases.index)

    assert list(checks.index) == CASE_ORDER
    max_difference = checks["|logit difference|"].max()
    assert max_difference < 1e-4, f"explain() + bias and forward() disagree by {max_difference:.3e} log-odds"


def test_the_four_cases_carry_the_right_outcome_labels(canonical, cases):
    model, ranges, splits, p = canonical
    y_te = splits[5]
    flagged = p >= CANONICAL_THRESHOLD
    expected = {"True positive": (1, True), "True negative": (0, False),
                "False negative": (1, False), "False positive": (0, True)}

    assert list(cases.index) == CASE_ORDER
    for name, (label, is_flagged) in expected.items():
        task = cases.loc[name, "task"]
        assert int(y_te[task]) == label, f"{name}: ground truth is {int(y_te[task])}, expected {label}"
        assert bool(flagged[task]) is is_flagged, f"{name} is on the wrong side of the {CANONICAL_THRESHOLD} threshold"
        assert cases.loc[name, "true label"] == label and cases.loc[name, "flagged"] is is_flagged

    # The rule is "most confident in its group", so nothing in the group is more extreme than the pick.
    assert p[cases.loc["True positive", "task"]] == p[(y_te == 1) & flagged].max()
    assert p[cases.loc["True negative", "task"]] == p[(y_te == 0) & ~flagged].min()
    assert p[cases.loc["False negative", "task"]] == p[(y_te == 1) & ~flagged].max()
    assert p[cases.loc["False positive", "task"]] == p[(y_te == 0) & flagged].max()


def test_centered_contributions_are_the_closed_form_shap_values_and_sum_to_the_logit(canonical, cases):
    model, ranges, splits, p = canonical
    X_tr, X_te = splits[0], splits[2]
    means = phi_training_means(model, X_tr, ranges)
    baseline = model.bias.item() + means.sum()
    tasks = list(cases["task"])
    shap_values = kaam_shap_values(model, X_tr, X_te.loc[tasks], ranges)
    logits, _ = forward_logits(model, torch.tensor(stability.scaled_matrix(X_te.loc[tasks], ranges)))

    for i, task in enumerate(tasks):
        table = explain_case(model, X_te, ranges, task, means)
        assert len(table) == 10 and set(table.index) == set(FEATURE_NAMES)
        assert np.all(np.diff(table["vs average task"].abs().to_numpy()) <= 0), "bars are not sorted by magnitude"
        gap = (table["vs average task"] - shap_values.loc[task, table.index]).abs().max()
        assert gap < 1e-5, f"centered contribution differs from kaam_shap_values by {gap:.2e}"
        assert abs(baseline + table["vs average task"].sum() - logits[i]) < 1e-4


def test_forward_logits_leaves_torch_sigmoid_untouched(canonical):
    model, ranges, splits, p = canonical
    original = torch.sigmoid
    logits, probabilities = forward_logits(model, torch.tensor(stability.scaled_matrix(splits[2].head(5), ranges)))
    assert torch.sigmoid is original
    assert np.allclose(1 / (1 + np.exp(-logits)), probabilities, atol=1e-7)


def test_select_cases_rejects_mismatched_indexes(canonical):
    model, ranges, splits, p = canonical
    with pytest.raises(ValueError):
        select_cases(splits[5], p.iloc[::-1], threshold=CANONICAL_THRESHOLD)
