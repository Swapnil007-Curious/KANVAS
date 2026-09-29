"""Tests for Phase 7: the user-grouped robustness check (kanvas/robustness.py).

These tests never train a model. They check the things that would silently
invalidate the phase: a bad user join, a user whose tasks leak across the
split, an identifier reaching an exported file, and a comparison table that
does not hold every model under both splits.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kanvas.data import FEATURE_NAMES
from kanvas.robustness import (PROPORTIONS, ROLES, choose_partition, comparison_table, grouped_partitions,
                               load_user_ids, split_by_role, user_statistics)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LARGE_CSV = ROOT / "data" / "processed" / "kanvas_features_large.csv"
PUBLICATION_DIR = ROOT / "data" / "processed" / "publication"
PUBLICATION_FILES = ["kanvas_feature_matrix.csv", "kanvas_phi_curves.csv", "kanvas_benchmark_results.csv"]

MINIMUM_MATCH_RATE = 0.90


def _require(path):
    if not path.exists():
        pytest.fail(f"{path} is missing; run notebooks/05_scale_and_stability.ipynb and 07 first")
    return path


@pytest.fixture(scope="module")
def dataset():
    df = pd.read_csv(_require(LARGE_CSV), index_col=["job_name", "task_name"])
    return df[FEATURE_NAMES], df["label"]


@pytest.fixture(scope="module")
def users(dataset):
    X, _ = dataset
    if not RAW_DIR.exists():
        pytest.skip(f"{RAW_DIR} not found")
    return load_user_ids(str(RAW_DIR), X.index, verbose=False)


@pytest.fixture(scope="module")
def partition(dataset, users):
    _, y = dataset
    user, _ = users
    attempts, assignments = grouped_partitions(user, y)
    seed, role, balanced = choose_partition(attempts, assignments, user)
    return attempts, seed, role, balanced


def test_user_join_matches_at_least_90_percent_of_tasks(users):
    user, report = users
    rate = report["match rate"]
    assert rate >= MINIMUM_MATCH_RATE, (
        f"user join matched only {rate:.2%} of tasks ({report['matched']:,} of {report['tasks']:,}); "
        f"{report['unmatched']:,} tasks have no user and grouping cannot be trusted")
    assert report["distinct users"] > 1, "a grouped split needs more than one user"


def test_no_user_appears_in_more_than_one_part(dataset, users, partition):
    X, y = dataset
    user, _ = users
    _, _, role, _ = partition

    parts = {part: set(user.loc[role.index[role == part]].dropna()) for part in ROLES}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = parts[a] & parts[b]
        assert not shared, f"{len(shared)} user(s) appear in both {a} and {b}: the grouping leaked"
    assert sum(len(part) for part in parts.values()) == user.nunique()

    # Path A keeps every task, including the dominant user's: this is a grouped split, not an exclusion.
    assert len(role) == len(X) and role.isin(ROLES).all()
    splits = split_by_role(X, y, role)
    assert sum(len(part) for part in splits[:3]) == len(X)
    for part in splits[:3]:
        assert list(part.columns) == FEATURE_NAMES, "user must never reach the feature set"


def test_partition_is_chosen_on_label_balance_only(partition):
    attempts, seed, _, balanced = partition
    assert len(attempts) == 20, "the phase runs 20 partition attempts"
    assert seed == attempts["worst failure-rate gap"].idxmin(), "the chosen partition is not the best-balanced one"
    assert balanced == bool(attempts.loc[seed, "worst failure-rate gap"] <= 0.05)
    for part, target in PROPORTIONS.items():
        assert abs(attempts.loc[seed, f"{part} share"] - target) < 0.10, f"{part} is far from its {target:.0%} target"


@pytest.mark.parametrize("filename", PUBLICATION_FILES)
def test_publication_csvs_still_carry_no_user_or_job_name_column(filename):
    columns = pd.read_csv(_require(PUBLICATION_DIR / filename), nrows=1).columns
    for forbidden in ("user", "job_name", "task_name", "user_id"):
        assert forbidden not in columns, f"{filename} exports a {forbidden!r} column"


def test_comparison_table_has_two_splits_times_three_models():
    kaam = pd.DataFrame({"test AUC": [0.7, 0.71, 0.705], "test accuracy": [0.8, 0.81, 0.805]}, index=[0, 1, 2])
    baselines = pd.DataFrame({"test AUC": [0.68, 0.72], "test accuracy": [0.78, 0.82]}, index=["LR", "MLP"])
    random_metrics = {name: {"test AUC": 0.75, "test accuracy": 0.81} for name in ("KAAM", "LR", "MLP")}

    table = comparison_table(random_metrics, kaam, baselines)

    assert len(table) == 6, f"expected 2 split types x 3 models, got {len(table)} rows"
    assert set(table.index.get_level_values("split")) == {"random task-level split", "user-grouped split"}
    assert set(table.index.get_level_values("model")) == {"KAAM", "LR", "MLP"}
    assert table[["test AUC", "test accuracy"]].notna().all().all()
    assert table.loc[("user-grouped split", "KAAM"), "test AUC"] == pytest.approx(kaam["test AUC"].mean())
    assert table.loc[("user-grouped split", "KAAM"), "runs"] == 3


def test_user_statistics_shares_sum_to_one_and_hide_the_raw_ids(dataset, users):
    _, y = dataset
    user, _ = users
    stats = user_statistics(user, y)

    assert stats["tasks"].sum() == len(y)
    assert stats["failures"].sum() == y.sum()
    assert np.isclose(stats["share of tasks"].sum(), 1.0) and np.isclose(stats["share of failures"].sum(), 1.0)
    assert list(stats.index[:3]) == ["user 1", "user 2", "user 3"], "display labels must be rank-based, not raw ids"
    assert stats["tasks"].is_monotonic_decreasing
