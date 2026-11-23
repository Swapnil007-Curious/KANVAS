"""Leak-free lookback ablation on the v1 40,000-task sample (paper/PROTOCOL.md, section 5).

    python scripts/lookback_v2.py

For each window (0.5, 1, 2, 4, 6 h) the tasks of the v1 sample that survive the rules form one
dataset. Every task keeps the role it has in the canonical 1 h split of the original sample;
tasks that exist only at a wider window get their own stratified 70/15/15 split (seed 42). Each
window fits its own training-only preprocessing and one KAAM (seed 0, lambda chosen on the 1 h
validation split of the original sample). Writes results/lookback/:

  lookback.csv     per window: tasks, failure rate, test tasks, own-test AUC, AUC on the test tasks
                   shared with 1 h, the paired difference against 1 h with its interval and Holm p
  curves.csv       telemetry curves (avg_cpu_kernel, avg_net_receive) per window, centred
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kanvas import models_v2 as M  # noqa: E402
from kanvas.data import FEATURE_NAMES  # noqa: E402
from kanvas.metrics import bootstrap_models, bootstrap_p, holm, percentile_ci, roc_auc  # noqa: E402
from kanvas.preprocess import fit_preprocessor  # noqa: E402
from kanvas.protocols import random_split  # noqa: E402
from kanvas.testaccess import test_labels  # noqa: E402
from run_experiments import RUNS, V1_LARGE  # noqa: E402

V2 = ROOT / "data" / "processed" / "v2"
OUT = ROOT / "results" / "lookback"
WIDTHS = [0.5, 1.0, 2.0, 4.0, 6.0]


def tag(h):
    return f"{h:g}".replace(".", "p")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(V2 / "v1_sample_40000_tasks.csv")
    sample_index = pd.MultiIndex.from_frame(sample[["job_name", "task_name"]])
    lam = json.loads((RUNS / "original" / "random" / "s42" / "kaam.json").read_text())["chosen"]["lambda"]

    # canonical roles: the original sample (v1 order) with random split 42
    v1_index = pd.read_csv(V1_LARGE, index_col=["job_name", "task_name"], usecols=["job_name", "task_name"]).index
    full1 = pd.read_parquet(V2 / "full_1h.parquet", columns=["label"])
    y1 = full1.loc[v1_index, "label"].to_numpy()
    s = random_split(y1, 42)
    role = pd.Series("", index=v1_index)
    for part in ("train", "val", "test"):
        role.iloc[s[part]] = part

    fitted, data = {}, {}
    template = M.KAAM(FEATURE_NAMES, M.UNIT_RANGES, num_knots=M.NUM_KNOTS, spline_order=M.SPLINE_ORDER)
    for h in WIDTHS:
        df = pd.read_parquet(V2 / f"full_{tag(h)}h.parquet", columns=FEATURE_NAMES + ["label"])
        df = df[df.index.isin(sample_index)]
        r = role.reindex(df.index).fillna("")
        extra = np.flatnonzero((r == "").to_numpy())
        if len(extra):
            es = random_split(df["label"].to_numpy(), 42, positions=extra)
            rv = r.to_numpy().copy()
            for part in ("train", "val", "test"):
                rv[es[part]] = part
            r = pd.Series(rv, index=df.index)
        parts = {p: np.flatnonzero(r.to_numpy() == p) for p in ("train", "val", "test")}
        X = df[FEATURE_NAMES]
        yv = df["label"].to_numpy()
        pp = fit_preprocessor(X.iloc[parts["train"]])
        xs = {p: pp.transform(X.iloc[parts[p]]).to_numpy(np.float32) for p in parts}
        des = {p: M.kaam_design(template, xs[p]) for p in parts}
        f = M.fit_kaam(des["train"], yv[parts["train"]], des["val"], yv[parts["val"]], 0, lam)
        fitted[h] = (f, pp)
        data[h] = {"index": df.index[parts["test"]], "p": f.predict(des["test"]), "n": len(df),
                   "rate": float(yv.mean()), "test_idx": parts["test"], "y": yv, "early": f.info["early_stopped"]}
        print(f"{h:g} h: {len(df):,} tasks, test {len(parts['test']):,}, best epoch {f.info['best_epoch']}", flush=True)

    rows, ps = [], []
    ref = data[1.0]
    y_ref = test_labels(ref["y"], ref["test_idx"], dataset="lookback-1h", protocol="lookback", split="1h",
                        purpose="lookback ablation (scripts/lookback_v2.py)")
    ref_s = pd.Series(ref["p"], index=ref["index"])
    yref_s = pd.Series(y_ref, index=ref["index"])
    for h in WIDTHS:
        dh = data[h]
        y_h = y_ref if h == 1.0 else test_labels(dh["y"], dh["test_idx"], dataset=f"lookback-{h:g}h",
                                                 protocol="lookback", split=f"{h:g}h",
                                                 purpose="lookback ablation (scripts/lookback_v2.py)")
        own_auc = roc_auc(y_h, dh["p"])
        row = {"width_h": h, "tasks": dh["n"], "failure_rate": dh["rate"], "test_tasks": len(dh["p"]),
               "own_test_auc": own_auc, "early_stopped": dh["early"]}
        if h != 1.0:
            shared = ref["index"].intersection(dh["index"])
            p_h = pd.Series(dh["p"], index=dh["index"]).loc[shared].to_numpy()
            p_1 = ref_s.loc[shared].to_numpy()
            yy = yref_s.loc[shared].to_numpy()
            point, boot = bootstrap_models([{"y": yy, "scores": {"h": [p_h], "ref": [p_1]}}], seed=0)
            diff = boot["h"] - boot["ref"]
            lo, hi = percentile_ci(diff)
            ps.append(bootstrap_p(diff))
            row.update(shared_test=len(shared), shared_failures=int(yy.sum()), auc_shared=point["h"],
                       auc_1h_shared=point["ref"], diff=point["h"] - point["ref"], ci_low=lo, ci_high=hi, p=ps[-1])
        rows.append(row)
    adj = iter(holm(ps))
    for row in rows:
        if row["width_h"] != 1.0:
            row["p_holm"] = float(next(adj))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "lookback.csv", index=False)
    print(table.round(4).to_string())

    crow = []
    for name in ["avg_cpu_kernel", "avg_net_receive"]:
        lo = max(fitted[h][1].clip_bounds[name][0] for h in WIDTHS)
        hi = min(fitted[h][1].clip_bounds[name][1] for h in WIDTHS)
        grid = np.linspace(lo, hi, 200)
        for h in WIDTHS:
            f, pp = fitted[h]
            import torch
            with torch.no_grad():
                v = f.model.edges[name](torch.as_tensor(pp.to_scaled(name, grid), dtype=torch.float32)).numpy()
            crow += [{"feature": name, "width_h": h, "x": x, "phi": p} for x, p in zip(grid, v - v.mean())]
    pd.DataFrame(crow).to_csv(OUT / "curves.csv", index=False)


if __name__ == "__main__":
    main()
