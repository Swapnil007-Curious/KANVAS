"""The v2 publication files carry no identifiers and match the full build."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kanvas.data import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "processed" / "publication"
FORBIDDEN = {"user", "user_id", "job_name", "task_name", "worker_name", "machine", "inst_id", "group"}
V2_FILES = ["kanvas_feature_matrix.csv.gz", "kanvas_phi_curves.csv", "kanvas_benchmark_results.csv",
            "kanvas_comparisons.csv"]


def _cols(path):
    if not path.exists():
        pytest.skip(f"{path.name} not written yet (scripts/export_publication.py)")
    return set(pd.read_csv(path, nrows=5).columns)


@pytest.mark.parametrize("name", V2_FILES)
def test_v2_exports_carry_no_identifier_column(name):
    cols = {c.lower() for c in _cols(PUB / name)}
    assert not cols & FORBIDDEN, f"{name} exports {sorted(cols & FORBIDDEN)}"


@pytest.mark.parametrize("path", sorted((ROOT / "data" / "processed" / "publication").rglob("*.csv*")))
def test_no_exported_file_carries_an_identifier_column(path):
    cols = {c.lower() for c in pd.read_csv(path, nrows=5).columns}
    assert not cols & FORBIDDEN, f"{path.name} exports {sorted(cols & FORBIDDEN)}"


def test_feature_matrix_matches_the_full_build():
    path = PUB / "kanvas_feature_matrix.csv.gz"
    full = ROOT / "data" / "processed" / "v2" / "full_1h.parquet"
    if not path.exists() or not full.exists():
        pytest.skip("export or full build missing")
    pub = pd.read_csv(path)
    ref = pd.read_parquet(full, columns=FEATURE_NAMES + ["label"])
    assert list(pub.columns) == ["task_id"] + FEATURE_NAMES + ["label", "start_time", "original_sample"]
    assert len(pub) == len(ref) and pub["task_id"].tolist() == list(range(len(ref)))
    assert (pub["label"].to_numpy() == ref["label"].to_numpy()).all()
    assert np.allclose(pub[FEATURE_NAMES].to_numpy(), ref[FEATURE_NAMES].to_numpy(), rtol=1e-9, atol=0)
    assert int(pub["original_sample"].sum()) == 18890
