"""Build the v2 datasets from the raw trace and check them against the v1 matrices.

    python scripts/build_full_trace.py

Writes (git-ignored, they hold identifiers):
  data/interim/pai_instance_table.parquet       the cached instance-table columns
  data/processed/v2/full_<h>h.parquet            every labelled task that survives the rules, per lookback
Writes (tracked):
  results/data/attrition_full_<h>h.csv           the attrition table per lookback
  results/data/build_checks.json                 the v1 reproduction checks and the dataset summary

The v1 check: v1 clipped every feature of its 18,890-task matrix at the 1st/99th percentiles of all
18,890 rows (the leak this version removes). Taking the same tasks from the full 1 h build and
clipping them the same way must give back the v1 matrix value for value; likewise for the five
lookback matrices of the v1 ablation, built from the same 40,000-task sample.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kanvas.data import FEATURE_NAMES  # noqa: E402
from kanvas.fulltrace import build_full_dataset, cache_instance_table, v1_stratified_sample  # noqa: E402

RAW = ROOT / "data" / "raw"
INSTANCE_PARQUET = ROOT / "data" / "interim" / "pai_instance_table.parquet"
OUT = ROOT / "data" / "processed" / "v2"
RESULTS = ROOT / "results" / "data"
LOOKBACKS = [0.5, 1.0, 2.0, 4.0, 6.0]
V1_LARGE = ROOT / "data" / "processed" / "kanvas_features_large.csv"
V1_LOOKBACK = ROOT / "data" / "processed" / "lookback" / "features_lookback_{}h.csv"


def tag(h):
    return f"{h:g}".replace(".", "p")


def v1_clip(frame):
    """v1's clipping, fitted on all rows of ``frame`` (reproduced only to check the build)."""
    out = frame.copy()
    for c in FEATURE_NAMES:
        lo, hi = out[c].quantile([0.01, 0.99])
        out[c] = out[c].clip(lo, hi)
    return out


def compare_to_v1(built, v1_path):
    v1 = pd.read_csv(v1_path, index_col=["job_name", "task_name"])
    sub = built.reindex(v1.index)
    missing = int(sub["label"].isna().sum())
    sub = sub.dropna(subset=["label"])
    clipped = v1_clip(sub[FEATURE_NAMES])
    v1c = v1.loc[sub.index]
    max_rel = 0.0
    for c in FEATURE_NAMES:
        a, b = clipped[c].to_numpy(), v1c[c].to_numpy()
        rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))
        max_rel = max(max_rel, float(rel))
    return {"v1 rows": len(v1), "v1 rows missing from the v2 build": missing,
            "labels identical": bool((sub["label"].astype(int).to_numpy() == v1c["label"].to_numpy()).all()),
            "max relative feature difference": max_rel}


def main():
    t0 = time.time()
    if not INSTANCE_PARQUET.exists():
        print("caching the instance table (one pass over the 2 GB CSV) ...", flush=True)
        cache_instance_table(str(RAW), str(INSTANCE_PARQUET))
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    built = build_full_dataset(str(RAW), str(INSTANCE_PARQUET), lookback_hours=LOOKBACKS)
    checks = {"build seconds": None, "lookbacks": {}}
    for h, (frame, funnel) in built.items():
        frame.to_parquet(OUT / f"full_{tag(h)}h.parquet")
        funnel.to_csv(RESULTS / f"attrition_full_{tag(h)}h.csv", index=False)
        entry = {"tasks": len(frame), "failures": int(frame["label"].sum()),
                 "failure rate": float(frame["label"].mean()), "users": int(frame["user"].nunique()),
                 "tasks without user": int(frame["user"].isna().sum()),
                 "start_time min": float(frame["start_time"].min()),
                 "start_time max": float(frame["start_time"].max())}
        v1_file = V1_LARGE if h == 1.0 else Path(str(V1_LOOKBACK).format(f"{h:g}"))
        if v1_file.exists():
            entry["v1 check"] = compare_to_v1(frame, v1_file)
        if h == 1.0 and Path(str(V1_LOOKBACK).format("1")).exists():
            entry["v1 lookback-1h check"] = compare_to_v1(frame, Path(str(V1_LOOKBACK).format("1")))
        checks["lookbacks"][f"{h:g}"] = entry
        print(f"lookback {h:g} h: {json.dumps(entry, indent=1)}", flush=True)

    sample = v1_stratified_sample(str(RAW))
    v1 = pd.read_csv(V1_LARGE, index_col=["job_name", "task_name"])
    in_sample = v1.index.isin(pd.MultiIndex.from_frame(sample[["job_name", "task_name"]]))
    checks["v1 40,000-task sample rebuilt"] = {"sample size": len(sample),
                                              "v1 18,890 tasks inside the rebuilt sample": int(in_sample.sum())}
    sample.to_csv(OUT / "v1_sample_40000_tasks.csv", index=False)
    checks["build seconds"] = round(time.time() - t0, 1)
    (RESULTS / "build_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
