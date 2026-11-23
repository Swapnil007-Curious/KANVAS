"""KAAM and additive EBM shape functions compared at the training tasks' own values (exploratory).

    python scripts/shape_agreement.py

The protocol compares the two curves on 200 evenly spaced points between the training 5th and 95th percentiles
(scripts/interpret_v2.py). For the four request features, which take a few distinct values, most of those points
lie between observed values, so that comparison is dominated by how each model fills the gaps. This script adds
the Pearson correlation of the two centred curves evaluated at the values of up to 50,000 training tasks inside
the same window, so each value counts as often as it occurs. It reads training data only. Writes
results/interpret/ebm_shape_corr_data.csv.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kanvas.data import FEATURE_NAMES  # noqa: E402
from interpret_v2 import OUT, RUNS, kaam_models, load_pp, load_split, phi, window  # noqa: E402
from run_experiments import load_dataset  # noqa: E402


def main():
    src = json.loads((OUT / "ebm_shape_source.json").read_text(encoding="utf-8"))
    run = RUNS / src["dataset"] / "random" / "s42"
    d = load_dataset(src["dataset"])
    X_train = d["X"].iloc[load_split(run)["train"]]
    pp = load_pp(run)
    kaam = kaam_models(run)[0]
    with open(run / "ebm_seed0.pkl", "rb") as fh:
        ebm = pickle.load(fh)
    xs = pp.transform(X_train).to_numpy(np.float32)
    rng = np.random.default_rng(0)
    median_scaled = np.median(xs, axis=0)
    rows = []
    for i, n in enumerate(FEATURE_NAMES):
        lo, hi = window(X_train, n)
        inside = np.flatnonzero((X_train[n].to_numpy() >= lo) & (X_train[n].to_numpy() <= hi))
        take = rng.choice(inside, size=min(50_000, len(inside)), replace=False)
        v = xs[take, i]
        k = phi(kaam, n, v)
        Z = np.tile(median_scaled, (len(v), 1)).astype(np.float64)
        Z[:, i] = v
        e = ebm.eval_terms(Z)[:, list(ebm.term_names_).index(n)]
        k, e = k - k.mean(), e - e.mean()
        r = float(np.corrcoef(k, e)[0, 1]) if np.std(k) > 0 and np.std(e) > 0 else np.nan
        rows.append({"feature": n, "pearson_r_data": r, "distinct_values": int(X_train[n].iloc[take].nunique()),
                     "tasks": int(len(v))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ebm_shape_corr_data.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
