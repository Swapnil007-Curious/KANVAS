"""Write the v2 publication files (no user, job, task, worker or machine identifiers).

    python scripts/export_publication.py

Writes data/processed/publication/:
  kanvas_feature_matrix.csv.gz   every task of the full 1 h build: anonymous task_id, the ten raw (unclipped)
                                 features, the label, the start time in trace seconds and a flag for the
                                 18,890 tasks of the original sample, in build order
  kanvas_phi_curves.csv          the canonical KAAM's ten curves (full trace, random split 42, seed 0)
  kanvas_benchmark_results.csv   every model under every protocol on both datasets: ROC-AUC with its 95%
                                 bootstrap interval, PR-AUC, Brier score, ECE
  kanvas_comparisons.csv         KAAM minus each model: difference, interval, Holm-adjusted p-value
The v1 files are kept unchanged in data/processed/publication/v1/. Files whose inputs do not exist yet are
skipped, so the script can run before the analyses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kanvas.data import FEATURE_NAMES  # noqa: E402

PUB = ROOT / "data" / "processed" / "publication"
FULL = ROOT / "data" / "processed" / "v2" / "full_1h.parquet"
V1_LARGE = ROOT / "data" / "processed" / "kanvas_features_large.csv"
RES = ROOT / "results"
FORBIDDEN = {"user", "user_id", "job_name", "task_name", "worker_name", "machine", "inst_id", "group"}


def check(df, name):
    leaked = sorted(set(map(str.lower, df.columns)) & FORBIDDEN)
    if leaked or any(n in FORBIDDEN for n in df.index.names if n):
        raise SystemExit(f"{name} would export identifiers: {leaked or df.index.names}")


def main():
    PUB.mkdir(parents=True, exist_ok=True)
    if FULL.exists():
        df = pd.read_parquet(FULL, columns=FEATURE_NAMES + ["label", "start_time"])
        orig = pd.read_csv(V1_LARGE, index_col=["job_name", "task_name"], usecols=["job_name", "task_name"]).index
        out = df.reset_index(drop=False)
        out["original_sample"] = df.index.isin(orig).astype(int)
        out = out.drop(columns=[c for c in out.columns if c in FORBIDDEN])
        out.insert(0, "task_id", range(len(out)))
        out = out[["task_id"] + FEATURE_NAMES + ["label", "start_time", "original_sample"]]
        out["label"] = out["label"].astype(int)
        check(out, "feature matrix")
        out.to_csv(PUB / "kanvas_feature_matrix.csv.gz", index=False, float_format="%.10g",
                   compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
        print(f"feature matrix: {len(out):,} rows, {int(out['original_sample'].sum()):,} in the original sample")
    cur = RES / "interpret" / "curves_canonical.csv"
    if cur.exists():
        c = pd.read_csv(cur)
        c = c[c.model == "full seed 0"].rename(columns={"feature": "feature_name", "x": "x_value", "phi": "phi_value"})
        c = c[["feature_name", "x_value", "phi_value"]]
        check(c, "curves")
        c.to_csv(PUB / "kanvas_phi_curves.csv", index=False, float_format="%.8g")
        print(f"curves: {len(c):,} rows")
    rows, comps = [], []
    for ds in ["full", "original"]:
        a = RES / "analysis" / ds
        if not (a / "model_ci.csv").exists():
            continue
        ci = pd.read_csv(a / "model_ci.csv")
        su = pd.read_csv(a / "summary.csv")
        m = ci.merge(su[["protocol", "subset", "model", "pr_auc", "brier", "ece", "n_sets", "n_seeds", "n_test",
                         "failure_rate"]], on=["protocol", "subset", "model"])
        m.insert(0, "dataset", "full trace" if ds == "full" else "original sample")
        m = m.rename(columns={"auc": "roc_auc", "n_sets": "test_sets", "n_seeds": "seeds", "n_test": "test_tasks"})
        rows.append(m)
        cp = pd.read_csv(a / "comparisons.csv")
        cp.insert(0, "dataset", "full trace" if ds == "full" else "original sample")
        comps.append(cp)
    if rows:
        b = pd.concat(rows, ignore_index=True)
        check(b, "benchmark")
        b.to_csv(PUB / "kanvas_benchmark_results.csv", index=False, float_format="%.6g")
        c = pd.concat(comps, ignore_index=True)
        check(c, "comparisons")
        c.to_csv(PUB / "kanvas_comparisons.csv", index=False, float_format="%.6g")
        print(f"benchmark: {len(b)} rows; comparisons: {len(c)} rows")


if __name__ == "__main__":
    main()
    from publication_docs import write_docs

    write_docs()
