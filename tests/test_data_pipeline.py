"""Tests for the Alibaba Cluster-Trace-GPU-2020 data pipeline (kanvas/data.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kanvas.data import FEATURE_NAMES, _lookback_average, build_kanvas_dataset
from kanvas.model import KAAM

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_FILES = ["pai_task_table.csv", "pai_instance_table.csv", "pai_machine_metric.csv", "pai_machine_spec.csv"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Run the full pipeline once for the module (~1 min: it streams the 2 GB instance table)."""
    if not all((RAW_DIR / name).exists() for name in RAW_FILES):
        pytest.skip(f"raw Alibaba trace files not found in {RAW_DIR}")
    csv_path = tmp_path_factory.mktemp("processed") / "kanvas_features.csv"
    X, y, feature_ranges = build_kanvas_dataset(str(RAW_DIR), out_path=str(csv_path), verbose=False)
    return X, y, feature_ranges, csv_path


def test_returns_exactly_ten_feature_columns(dataset):
    X, _, _, _ = dataset
    assert X.shape[1] == 10
    assert list(X.columns) == FEATURE_NAMES


def test_no_nans_anywhere_in_the_output(dataset):
    X, y, feature_ranges, csv_path = dataset
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert all(np.isfinite(bounds).all() for bounds in feature_ranges.values())
    saved = pd.read_csv(csv_path, index_col=["job_name", "task_name"])
    assert saved.shape == (len(X), 11)
    assert not saved.isna().any().any()


def test_feature_ranges_keys_exactly_match_feature_names(dataset):
    X, _, feature_ranges, _ = dataset
    assert list(feature_ranges) == list(X.columns) == FEATURE_NAMES


def test_label_contains_only_zero_and_one(dataset):
    _, y, _, _ = dataset
    assert set(y.unique()) == {0, 1}


def test_every_feature_range_can_build_kaam(dataset):
    """Stands in for the dropped gpu_type_match 0/1 test.

    gpu_type_match turned out constant in this trace (task gpu_type is the
    placed machine's type), and a zero-width range makes KAAM fail to build.
    This guards against any feature collapsing to a constant again.
    """
    X, _, feature_ranges, _ = dataset
    assert all(hi > lo for lo, hi in feature_ranges.values())
    model = KAAM(FEATURE_NAMES, feature_ranges)
    out = model(torch.tensor(X.to_numpy(np.float32)))
    assert out.shape == (len(X),)


def test_lookback_uses_only_other_jobs_rows_that_ended_before_task_start():
    """Telemetry must come strictly from before task start, and never from the task's own job."""
    mm = pd.DataFrame(
        {
            "worker_name": ["w_old", "w_edge", "w_mid", "w_own", "w_at_start", "w_after", "w_other_machine"],
            "machine": ["m1", "m1", "m1", "m1", "m1", "m1", "m2"],
            "end_time": [100.0, 3600.0, 5000.0, 6000.0, 7200.0, 9000.0, 3600.0],
            "machine_cpu_usr": [999.0, 10.0, 30.0, 999.0, 999.0, 999.0, 999.0],
        }
    )
    tasks = pd.DataFrame(
        {
            "job_name": ["job_a", "job_b", "job_c"],
            "machine": ["m1", "m3", "m1"],
            "inst_start_time": [7200.0, 7200.0, np.nan],
        }
    )
    job_workers = {"job_a": frozenset({"w_own"})}

    means, n_rows, n_own_job, reasons = _lookback_average(tasks, mm, job_workers, 3600.0, ["machine_cpu_usr"])

    # Window for job_a is [3600, 7200): w_edge and w_mid are in; w_own is in the window but
    # belongs to the task's own job; w_at_start ends exactly at task start and is excluded.
    assert n_rows.tolist() == [2, 0, 0]
    assert n_own_job[0] == 1
    assert means[0, 0] == pytest.approx(20.0)
    assert reasons[0] == ""
    assert reasons[1] == "assigned machine has no machine_metric rows"
    assert reasons[2] == "first instance never started (NaN start_time)"
    assert np.isnan(means[1:]).all()
