"""Export the canonical KAAM to docs/model.json and docs/examples.json for the static demo.

    python scripts/export_web.py

The demo page (docs/index.html) runs the model in the browser from these two files. The model is the canonical
KAAM of the paper: full trace, random split 42, seed 0 (results/runs/full/random/s42, first state in
kaam_states.pt), the model behind the curves, the SHAP check and the case audits.

Everything is read from tracked files, so the export runs from a clone without the raw trace:
  results/runs/full/random/s42/{kaam_states.pt, preprocess.json, split.npz, kaam.npz}
  data/processed/publication/kanvas_feature_matrix.csv.gz   (identifier-free, rows in build order)
  results/analysis/full/{model_ci.csv, fold_auc.csv}         (the paper's test AUCs)
  results/interpret/cases.json                               (the paper's baseline and case audits, for checks)

model.json holds each feature's learned curve phi_i on 200 points across the training clip bounds (raw units),
the exact curve value at every value a discrete feature takes in training ("levels"), the bias, the training
mean of each curve (so contributions are measured against the average training task), the preprocessing
constants and the paper's AUC table. The page computes

    logit = bias + sum_i phi_i(clip(x_i)),   phi_i = exact level value if clip(x_i) is a level, else
                                             piecewise-linear interpolation on the 200-point grid

which ``web_logit`` below does in numpy, line for line with docs/kaam.js. KAAM has no categorical inputs; every
feature is numeric, and the five request and machine-size features are discrete, which is what ``levels`` covers.

examples.json holds 30 real test tasks of the canonical split (the ten features in raw units and the true label,
nothing else), spread over caught failures, correct all-clears, missed failures and false alarms at the paper's
0.5 threshold. Test labels are read through kanvas.testaccess, so the export adds one line to
results/test_access_log.jsonl.

The script stops if the model does not reproduce its saved test predictions, the paper's average-task logit or
the four case audits, or if the page's arithmetic drifts from the model by 1e-3 in probability on any test task.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = ROOT / "results" / "runs" / "full" / "random" / "s42"
MATRIX = ROOT / "data" / "processed" / "publication" / "kanvas_feature_matrix.csv.gz"
ANALYSIS = ROOT / "results" / "analysis" / "full"
CASES = ROOT / "results" / "interpret" / "cases.json"
DOCS = ROOT / "docs"
N_GRID = 200
MAX_LEVELS = 100  # a feature with at most this many distinct clipped training values is discrete
THRESHOLD = 0.5
PARITY_TOL = 1e-3
EXAMPLES = {"True positive": 8, "True negative": 7, "False negative": 8, "False positive": 7}

FEATURES = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num", "avg_cpu_usr", "avg_gpu_util", "avg_cpu_kernel",
            "avg_load_1", "cap_gpu", "avg_net_receive"]

# What a person reads. display value = raw / scale.
META = {
    "plan_cpu": dict(label="CPU request", unit="cores", scale=100, digits=1, group="request",
                     raw_unit="100 = 1 core", source="pai_task_table.plan_cpu",
                     help="CPU cores the task asked for."),
    "plan_mem": dict(label="Memory request", unit="GB", scale=1, digits=2, group="request", raw_unit="GB",
                     source="pai_task_table.plan_mem", help="Memory the task asked for."),
    "plan_gpu": dict(label="GPU request", unit="GPUs", scale=100, digits=2, group="request",
                     raw_unit="100 = 1 GPU", source="pai_task_table.plan_gpu",
                     help="GPUs the task asked for; fractions share a GPU."),
    "inst_num": dict(label="Instances", unit="instances", scale=1, digits=0, group="request", raw_unit="count",
                     source="pai_task_table.inst_num", help="Copies of the task requested."),
    "avg_cpu_usr": dict(label="Machine user CPU", unit="%", scale=1, digits=1, group="machine", raw_unit="percent",
                        source="pai_machine_metric.machine_cpu_usr",
                        help="User-space CPU on the assigned machine, averaged over other jobs that ended there "
                             "in the hour before the task started."),
    "avg_gpu_util": dict(label="Machine GPU use", unit="%", scale=1, digits=0, group="machine",
                         raw_unit="percent, summed over GPUs", source="pai_machine_metric.machine_gpu",
                         help="GPU utilisation on the machine in the same hour, summed over its GPUs, so values "
                              "above 100% are normal."),
    "avg_cpu_kernel": dict(label="Machine kernel CPU", unit="%", scale=1, digits=1, group="machine",
                           raw_unit="percent", source="pai_machine_metric.machine_cpu_kernel",
                           help="Kernel-space CPU on the machine in the same hour."),
    "avg_load_1": dict(label="Machine load", unit="", scale=1, digits=1, group="machine",
                       raw_unit="1-minute load average", source="pai_machine_metric.machine_load_1",
                       help="1-minute load average on the machine in the same hour."),
    "cap_gpu": dict(label="Machine size", unit="GPUs installed", scale=1, digits=0, group="machine",
                    raw_unit="count", source="pai_machine_spec.cap_gpu", help="GPUs installed on the machine."),
    "avg_net_receive": dict(label="Machine network in", unit="MB/s", scale=1e6, digits=0, group="machine",
                            raw_unit="bytes/s", source="pai_machine_metric.machine_net_receive",
                            help="Network receive rate on the machine in the same hour."),
}

# Columns of the paper's AUC table, in its order (results/analysis/full/model_ci.csv).
COLUMNS = [("random", "all", "Random"), ("time", "all", "Time, all"), ("time", "known users", "Time, known users"),
           ("time", "new users", "Time, new users"), ("grouped", "all", "User-grouped"),
           ("louo-d1", "all", "Without User 1"), ("louo-d14", "all", "Without Users 1, 4")]
MODELS = [("kaam", "KAAM"), ("lr", "Logistic regression"), ("mlp", "MLP"), ("lgbm", "LightGBM")]


# ----------------------------------------------------------------------------- the page's arithmetic
def web_phi(model: dict, name: str, raw) -> np.ndarray:
    """phi_name at raw values, as docs/kaam.js computes it: clip, exact level if any, else linear interpolation."""
    lo, hi = model["preprocess"]["clip"][name]
    v = np.clip(np.asarray(raw, dtype=np.float64), lo, hi)
    xs = np.asarray(model["curves"][name]["x"], dtype=np.float64)
    ys = np.asarray(model["curves"][name]["phi"], dtype=np.float64)
    # interval [xs[i], xs[i+1]] with xs[i] <= v < xs[i+1]; ends clamp to the first / last grid value
    i = np.clip(np.searchsorted(xs, v, side="right") - 1, 0, len(xs) - 2)
    t = (v - xs[i]) / (xs[i + 1] - xs[i])
    out = np.where(v <= xs[0], ys[0], np.where(v >= xs[-1], ys[-1], ys[i] + t * (ys[i + 1] - ys[i])))
    lev = model["levels"].get(name)
    if lev:
        lx = np.asarray(lev["x"], dtype=np.float64)
        lp = np.asarray(lev["phi"], dtype=np.float64)
        j = np.clip(np.searchsorted(lx, v), 0, len(lx) - 1)
        hit = np.abs(lx[j] - v) <= 1e-9 * np.maximum(1.0, np.abs(v))
        out = np.where(hit, lp[j], out)
    return out


def web_logit(model: dict, X: pd.DataFrame) -> np.ndarray:
    """Logit of each row of X (raw units, columns named by feature)."""
    return model["bias"] + sum(web_phi(model, n, X[n].to_numpy()) for n in model["features"])


def web_predict(model: dict, X: pd.DataFrame) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-web_logit(model, X)))


# ----------------------------------------------------------------------------- the PyTorch side
def load_canonical():
    """(model, preprocessor, split) of the canonical KAAM."""
    import torch

    from kanvas import models_v2 as M
    from kanvas.data import FEATURE_NAMES
    from kanvas.preprocess import Preprocessor

    assert list(FEATURE_NAMES) == FEATURES
    j = json.loads((RUN / "preprocess.json").read_text(encoding="utf-8"))
    pp = Preprocessor(tuple(FEATURES), {k: tuple(v) for k, v in j["clip_bounds"].items()},
                      {k: tuple(v) for k, v in j["ranges"].items()})
    model = M.load_kaam(torch.load(RUN / "kaam_states.pt", weights_only=False)[0])
    z = np.load(RUN / "split.npz")
    return model, pp, {k: z[k] for k in ("train", "val", "test")}


def torch_phi(model, name: str, scaled) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model.edges[name](torch.as_tensor(np.asarray(scaled, dtype=np.float32))).numpy().astype(np.float64)


def torch_predict(model, pp, X: pd.DataFrame) -> np.ndarray:
    import torch

    xs = torch.as_tensor(pp.transform(X[FEATURES]).to_numpy(np.float32))
    with torch.no_grad():
        return model(xs).numpy().astype(np.float64)


def load_matrix() -> pd.DataFrame:
    return pd.read_csv(MATRIX, usecols=["task_id"] + FEATURES)


# ----------------------------------------------------------------------------- export
def _sig(v: float, digits: int = 10) -> float:
    return float(f"{v:.{digits}g}")


def build_model_json(model, pp, X_train: pd.DataFrame, test_auc: float, n_test: int) -> dict:
    bias = float(model.bias.detach())
    phi_mean, curves, levels, train = {}, {}, {}, {}
    for n in FEATURES:
        lo, hi = pp.clip_bounds[n]
        assert tuple(pp.ranges[n]) == (lo, hi), n  # clip bounds are the scaling range here
        clipped = X_train[n].clip(lo, hi).to_numpy()
        phi_mean[n] = float(torch_phi(model, n, pp.to_scaled(n, clipped)).mean())
        # 200 points across the clip bounds; endpoints exact, inner points rounded, phi evaluated at the stored x
        grid = np.array([lo] + [_sig(v) for v in np.linspace(lo, hi, N_GRID)[1:-1]] + [hi])
        curves[n] = {"x": [float(v) for v in grid],
                     "phi": [round(float(v), 6) for v in torch_phi(model, n, pp.to_scaled(n, grid))]}
        uniq, counts = np.unique(clipped, return_counts=True)
        if len(uniq) <= MAX_LEVELS:
            levels[n] = {"x": [float(v) for v in uniq], "count": [int(c) for c in counts],
                         "phi": [round(float(v), 6) for v in torch_phi(model, n, pp.to_scaled(n, uniq))]}
            entry = {}
        else:
            c, edges = np.histogram(clipped, bins=40, range=(lo, hi))
            entry = {"hist": {"lo": lo, "hi": hi, "counts": [int(v) for v in c]}}
        entry.update({"p5": float(np.quantile(clipped, 0.05)), "p95": float(np.quantile(clipped, 0.95)),
                      "median": float(np.quantile(clipped, 0.5, method="lower")),
                      "below_clip": int((X_train[n] < lo).sum()), "above_clip": int((X_train[n] > hi).sum())})
        train[n] = entry
    baseline = bias + sum(phi_mean.values())

    ci = pd.read_csv(ANALYSIS / "model_ci.csv")
    table = []
    for key, label in MODELS:
        cells = []
        for proto, subset, _ in COLUMNS:
            r = ci[(ci.protocol == proto) & (ci.subset == subset) & (ci.model == key)].iloc[0]
            # six decimals, so the page's three-decimal display rounds once, as the paper does
            cells.append({"auc": round(float(r.auc), 6), "lo": round(float(r.ci_low), 6),
                          "hi": round(float(r.ci_high), 6)})
        table.append({"model": label, "cells": cells})
    folds = pd.read_csv(ANALYSIS / "fold_auc.csv")
    fold_rows = [{"model": label, "auc": [round(float(folds[(folds.fold == f) & (folds.model == key)].auc.iloc[0]), 6)
                                          for f in sorted(folds.fold.unique())]} for key, label in MODELS]
    fold_info = [{"fold": f, "test_users": int(g.test_users.iloc[0]), "test_tasks": int(g.n_test.iloc[0])}
                 for f, g in folds.groupby("fold", sort=True)]

    return {
        "format": "kanvas-web/1",
        "model": {"name": "KAAM", "description": "canonical KAAM: full trace, random split 42, seed 0",
                  "run": "results/runs/full/random/s42", "state": "kaam_states.pt[0]", "parameters": 151,
                  "train_tasks": int(len(X_train)), "test_tasks": n_test, "test_auc": round(test_auc, 6),
                  "threshold": THRESHOLD},
        "features": FEATURES,
        "meta": META,
        "preprocess": {"clip": {n: list(pp.clip_bounds[n]) for n in FEATURES},
                       "scale": {n: list(pp.ranges[n]) for n in FEATURES},
                       "note": "Clip each raw value to the training 1st and 99th percentiles, then min-max scale "
                               "with the clipped training range (the model's input). The curves here are already "
                               "in raw units, so the page only clips."},
        "bias": bias,
        "phi_mean": phi_mean,
        "baseline_logit": baseline,
        "curves": curves,
        "levels": levels,
        "categorical": {},
        "train": train,
        "paper": {"source": "results/analysis/full/model_ci.csv, fold_auc.csv",
                  "dataset": "full trace, 539,816 tasks",
                  "score": "test ROC-AUC: mean over test sets of the mean over five seeds, 95% bootstrap interval "
                           "(users resampled for the user-grouped split)",
                  "columns": [c[2] for c in COLUMNS], "rows": table,
                  "grouped_folds": {"folds": fold_info, "rows": fold_rows}},
    }


def pick_examples(X_test: pd.DataFrame, y_test: np.ndarray, p_test: np.ndarray) -> list:
    """30 distinct test tasks spread over risk within each outcome group; the group extremes are the paper's cases."""
    flagged = p_test >= THRESHOLD
    groups = {"True positive": (y_test == 1) & flagged, "True negative": (y_test == 0) & ~flagged,
              "False negative": (y_test == 1) & ~flagged, "False positive": (y_test == 0) & flagged}
    used, out = set(), []
    for name, k in EXAMPLES.items():
        idx = np.where(groups[name])[0]
        idx = idx[np.argsort(p_test[idx], kind="stable")]
        keys = [tuple(r) for r in X_test.iloc[idx][FEATURES].to_numpy()]
        picked = []
        for target in np.linspace(0, len(idx) - 1, k).round().astype(int):
            # nearest row to the target rank whose feature vector is not taken yet
            for step in range(len(idx)):
                for pos in (target + step, target - step):
                    if 0 <= pos < len(idx) and keys[pos] not in used:
                        break
                else:
                    continue
                break
            used.add(keys[pos])
            picked.append(int(idx[pos]))
        for i in picked:
            out.append({"group": name, "row": i, "risk": float(p_test[i]), "label": int(y_test[i])})
    return out


def main():
    from sklearn.metrics import roc_auc_score

    from kanvas.testaccess import test_labels

    model, pp, split = load_canonical()
    X = load_matrix()
    assert X["task_id"].tolist() == list(range(len(X))) and len(X) == 539_816
    X_train, X_test = X.iloc[split["train"]][FEATURES], X.iloc[split["test"]][FEATURES]

    # 1. the model reproduces its saved test predictions (the right model, the right rows)
    p_test = torch_predict(model, pp, X_test)
    saved = np.load(RUN / "kaam.npz")["test_pred"][0].astype(np.float64)
    gap = float(np.abs(p_test - saved).max())
    print(f"canonical KAAM vs saved test predictions: max gap {gap:.2e} over {len(p_test):,} tasks")
    assert gap < 1e-6

    y_all = pd.read_csv(MATRIX, usecols=["label"])["label"].to_numpy()
    y_test = test_labels(y_all, split["test"], dataset="full", protocol="random", split="s42",
                         purpose="web demo export: examples and test AUC (scripts/export_web.py)")
    test_auc = float(roc_auc_score(y_test, p_test))
    print(f"test ROC-AUC {test_auc:.4f}")

    out = build_model_json(model, pp, X_train, test_auc, len(X_test))

    # 2. the paper's average training task and case audits
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    print(f"average training task logit {out['baseline_logit']:+.6f} (paper {cases['baseline_logit']:+.6f})")
    assert abs(out["baseline_logit"] - cases["baseline_logit"]) < 1e-5

    # 3. the page's arithmetic against the model on every test task
    p_web = web_predict(out, X_test)
    gap = np.abs(p_web - p_test)
    print(f"page arithmetic vs PyTorch, all test tasks: max {gap.max():.2e}, mean {gap.mean():.2e}; "
          f"test ROC-AUC {roc_auc_score(y_test, p_web):.4f}")
    assert gap.max() < PARITY_TOL

    # 4. examples
    picks = pick_examples(X_test, y_test, p_test)
    for c in cases["cases"]:
        hit = [p for p in picks if p["group"] == c["case"] and abs(p["risk"] - c["risk"]) < 1e-6]
        assert len(hit) == 1, c["case"]
        assert all(abs(X_test.iloc[hit[0]["row"]][n] - c["raw"][n]) < 1e-9 * max(1, abs(c["raw"][n]))
                   for n in FEATURES), c["case"]
        hit[0]["paper_case"] = c["case"]
    examples = []
    for k, p in enumerate(picks, 1):
        row = X_test.iloc[p["row"]]
        e = {"id": k, "label": p["label"], "x": {n: float(row[n]) for n in FEATURES}}
        if "paper_case" in p:
            e["paper_case"] = p["paper_case"]
        examples.append(e)
    counts = pd.Series([p["group"] for p in picks]).value_counts().to_dict()
    print(f"examples: {len(examples)} ({counts}); paper cases included: "
          f"{sum('paper_case' in e for e in examples)}")
    assert len(examples) == 30 and len({tuple(e['x'].values()) for e in examples}) == 30

    DOCS.mkdir(exist_ok=True)
    (DOCS / "model.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    (DOCS / "examples.json").write_text(json.dumps({"source": "test split of the canonical KAAM (full trace, random "
                                                               "split 42); raw feature values and true label only",
                                                    "examples": examples}, separators=(",", ":")),
                                        encoding="utf-8")
    for f in ("model.json", "examples.json"):
        print(f"wrote docs/{f}: {(DOCS / f).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
