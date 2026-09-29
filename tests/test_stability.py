"""Tests for the Phase 5 scale-up and stability work (kanvas/stability.py).

The build test actually re-runs ``build_kanvas_dataset`` at a larger sample,
which streams the 2 GB instance table and takes minutes. It is marked
``slow``; run ``pytest -m "not slow"`` to skip it, but note that skipping it
leaves the scale-up claim itself untested.
"""

from pathlib import Path

import pandas as pd
import pytest

from kanvas import stability
from kanvas.data import FEATURE_NAMES, build_kanvas_dataset
from kanvas.stability import FORBIDDEN_COLUMNS, initialization_variance, split_variance, strip_identifiers

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SMALL_CSV = ROOT / "data" / "processed" / "kanvas_features.csv"
LARGE_CSV = ROOT / "data" / "processed" / "kanvas_features_large.csv"
PUBLICATION_DIR = ROOT / "data" / "processed" / "publication"
PUBLICATION_FILES = ["kanvas_feature_matrix.csv", "kanvas_phi_curves.csv", "kanvas_benchmark_results.csv"]

SMALL_ROWS = 2310  # the Phase 3 development sample
LARGER_SAMPLE = 8_000  # > the 5,000 tasks Phase 2 sampled


def _read(path):
    if not path.exists():
        pytest.fail(f"{path} is missing; run notebooks/05_scale_and_stability.ipynb to produce it")
    return pd.read_csv(path)


@pytest.mark.slow
def test_build_at_a_larger_sample_yields_more_rows_than_the_development_set(tmp_path):
    if not RAW_DIR.exists():
        pytest.skip(f"{RAW_DIR} not found")
    X, y, _ = build_kanvas_dataset(str(RAW_DIR), sample_n_tasks=LARGER_SAMPLE,
                                   out_path=str(tmp_path / "features.csv"), verbose=False)
    assert len(X) > SMALL_ROWS, f"sample_n_tasks={LARGER_SAMPLE:,} gave {len(X):,} rows, not more than {SMALL_ROWS:,}"
    assert list(X.columns) == FEATURE_NAMES
    assert len(y) == len(X)


def test_large_sample_failure_rate_is_within_5_points_of_the_small_sample():
    small, large = _read(SMALL_CSV), _read(LARGE_CSV)
    small_rate, large_rate = small["label"].mean(), large["label"].mean()
    gap = abs(large_rate - small_rate) * 100
    assert len(large) > len(small), f"large sample has {len(large):,} rows, not more than {len(small):,}"
    assert gap <= 5.0, (f"representativeness gate failed: small {small_rate:.2%} vs large {large_rate:.2%} "
                        f"is {gap:.2f} points apart, over the 5-point limit")


def test_multi_seed_experiments_return_exactly_one_row_per_seed(monkeypatch):
    # Two epochs per run: this test is about the shape of the output, not the numbers.
    monkeypatch.setattr(stability, "MAX_EPOCHS", 2)
    monkeypatch.setattr(stability, "PATIENCE", 1)
    df = _read(LARGE_CSV).head(400)
    X, y = df[FEATURE_NAMES], df["label"]

    seeds = [0, 1, 2, 3, 4]
    a = initialization_variance(X, y, seeds=seeds, verbose=False)
    b = split_variance(X, y, split_states=[42, 43, 44, 45, 46], verbose=False)

    assert len(a) == 5 and len(b) == 5
    assert list(a["init_seed"]) == seeds  # every requested seed present, in order
    assert list(b["split_state"]) == [42, 43, 44, 45, 46]
    assert a["split_state"].nunique() == 1, "Experiment A must hold the split fixed"
    assert b["init_seed"].nunique() == 1, "Experiment B must hold the initialization fixed"
    for table in (a, b):
        assert table[["test AUC", "test accuracy", "test F1"]].notna().all().all()


@pytest.mark.parametrize("filename", PUBLICATION_FILES)
def test_publication_csvs_carry_no_identifier_columns(filename):
    df = _read(PUBLICATION_DIR / filename)
    for forbidden in ("user", "job_name"):
        assert forbidden not in df.columns, f"{filename} exports a {forbidden!r} column"
    leaked = [c for c in df.columns if c in FORBIDDEN_COLUMNS]
    assert not leaked, f"{filename} exports identifier columns {leaked}"


def test_strip_identifiers_removes_the_job_name_index():
    df = pd.DataFrame({"plan_cpu": [1.0, 2.0], "label": [0, 1], "user": ["u1", "u2"]},
                      index=pd.MultiIndex.from_tuples([("j1", "t1"), ("j2", "t2")], names=["job_name", "task_name"]))
    out, dropped = strip_identifiers(df)
    assert "user" in dropped and "user" not in out.columns
    assert "job_name" not in out.columns and "task_name" not in out.columns
    assert list(out["task_id"]) == [0, 1]
    assert list(out["plan_cpu"]) == [1.0, 2.0]
