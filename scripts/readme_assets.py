"""The key figure and the results tables of README.md, from the result files.

    python scripts/readme_assets.py

Writes figures/readme/benchmark_full.png, a raster copy of the paper's full-trace benchmark figure that GitHub
can show, and prints the README's results tables in Markdown (from results/analysis/*/model_ci.csv). The paper's
own PDF figures are not touched.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import paper_figures as pf  # noqa: E402

FIG = ROOT / "figures" / "readme"
NAMES = {"kaam": "KAAM", "lr": "Logistic regression", "mlp": "MLP", "lgbm": "LightGBM",
         "ebm": "EBM, additive", "ebm_int": "EBM, pairwise"}
COLUMNS = [("random", "all", "Random"), ("time", "all", "Time, all"), ("time", "known users", "Time, known users"),
           ("time", "new users", "Time, new users"), ("grouped", "all", "User-grouped"),
           ("louo-d1", "all", "Without User 1"), ("louo-d14", "all", "Without Users 1, 4")]
SHORT = ["Random", "Time, known users", "Time, new users", "User-grouped"]


def figure():
    FIG.mkdir(parents=True, exist_ok=True)
    os.environ["KANVAS_FIG_PNG"] = str(FIG)
    fitted = set(pd.read_csv(ROOT / "results" / "analysis" / "full" / "model_ci.csv").model)
    pf.MODELS = [m for m in pf.MODELS if m in fitted]  # the EBMs were fitted on the original sample only
    with tempfile.TemporaryDirectory() as tmp:
        pf.OUT = Path(tmp)  # the PDF copy goes to a throwaway folder
        pf.benchmark(datasets=("full",), name="benchmark_full", width=9.0)


def table(ds, labels):
    ci = pd.read_csv(ROOT / "results" / "analysis" / ds / "model_ci.csv")
    cols = [c for c in COLUMNS if c[2] in labels]
    lines = ["| Model | " + " | ".join(c[2] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for m, name in NAMES.items():
        cells = []
        for proto, sub, _ in cols:
            r = ci[(ci.protocol == proto) & (ci.subset == sub) & (ci.model == m)]
            cells.append(f"{r.auc.iloc[0]:.3f}" if len(r) else None)
        if all(c is None for c in cells):
            continue
        lines.append(f"| {name} | " + " | ".join(c or "n/a" for c in cells) + " |")
    return "\n".join(lines)


def main():
    figure()
    all_labels = [c[2] for c in COLUMNS]
    print("\nShort table (full trace):\n")
    print(table("full", SHORT))
    print("\nFull trace:\n")
    print(table("full", all_labels))
    print("\nOriginal sample:\n")
    print(table("original", all_labels))


if __name__ == "__main__":
    main()
