"""Interpretability analyses of paper/PROTOCOL.md (section 6), on the saved v2 models.

    python scripts/interpret_v2.py

Writes results/interpret/:
  shap_check.json          exact Kernel SHAP on the canonical KAAM against its centred curves
  ebm_shape_corr.csv       per-feature Pearson correlation of the KAAM and additive-EBM shape functions
  shapes_kaam_ebm.csv      both shape functions on the 200-point grids (for the figure)
  curves_canonical.csv     all ten canonical curves (5 seeds) and the original-sample canonical curve
  plan_cpu_protocols.csv   the seed-0 plan_cpu curve from every protocol on the full data
  plan_cpu_summary.csv     swing and shape correlation of each of those curves against the canonical one
  cases.json               the four case audits on the canonical model (same pre-set rule as v1)
  user_concentration.csv   share of tasks and failures held by the top-k users, both datasets
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kanvas import models_v2 as M  # noqa: E402
from kanvas.data import FEATURE_NAMES  # noqa: E402
from kanvas.preprocess import Preprocessor  # noqa: E402
from kanvas.testaccess import test_labels  # noqa: E402
from run_experiments import RUNS, dominant_users, load_dataset  # noqa: E402

OUT = ROOT / "results" / "interpret"
CANON = RUNS / "full" / "random" / "s42"
THRESHOLD = 0.5
N_GRID = 200


def load_pp(run_dir):
    j = json.loads((run_dir / "preprocess.json").read_text(encoding="utf-8"))
    return Preprocessor(tuple(FEATURE_NAMES), {k: tuple(v) for k, v in j["clip_bounds"].items()},
                        {k: tuple(v) for k, v in j["ranges"].items()})


def load_split(run_dir):
    z = np.load(run_dir / "split.npz")
    return {k: z[k] for k in ("train", "val", "test")}


def kaam_models(run_dir):
    return [M.load_kaam(s) for s in torch.load(run_dir / "kaam_states.pt", weights_only=False)]


def phi(model, name, scaled):
    with torch.no_grad():
        return model.edges[name](torch.as_tensor(np.asarray(scaled, dtype=np.float32))).numpy().astype(float)


def logit(model, xs):
    x = torch.as_tensor(np.asarray(xs, dtype=np.float32))
    with torch.no_grad():
        return (sum(model.edges[n](x[:, i]) for i, n in enumerate(FEATURE_NAMES)) + model.bias).numpy()


def window(X_train, name, lo=0.05, hi=0.95):
    a, b = X_train[name].quantile([lo, hi])
    return float(a), float(b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_dataset("full")
    X, y = d["X"], d["y"]
    split = load_split(CANON)
    pp = load_pp(CANON)
    X_train = X.iloc[split["train"]]
    models = kaam_models(CANON)
    canon = models[0]
    xs_train = pp.transform(X_train).to_numpy(np.float32)

    # ------------------------------------------------------------------ exact SHAP check
    import shap

    rng = np.random.default_rng(0)
    bg = xs_train[rng.choice(len(xs_train), 100, replace=False)]
    xs_test = pp.transform(X.iloc[split["test"]]).to_numpy(np.float32)
    explain_rows = xs_test[np.random.default_rng(1).choice(len(xs_test), 40, replace=False)]
    explainer = shap.KernelExplainer(lambda z: logit(canon, z), bg)
    sv = explainer.shap_values(explain_rows, nsamples=2 ** 10, l1_reg=False, silent=True)
    centred = np.column_stack([phi(canon, n, explain_rows[:, i]) - phi(canon, n, bg[:, i]).mean()
                               for i, n in enumerate(FEATURE_NAMES)])
    expl_gap = max(abs(sum(canon.explain(torch.as_tensor(r)).values()) + canon.bias.item() - logit(canon, r[None])[0])
                   for r in explain_rows)
    shap_check = {"tasks": 40, "background": 100, "coalitions": 2 ** 10,
                  "max_abs_diff_shap_vs_centred_curves": float(np.max(np.abs(sv - centred))),
                  "max_abs_explain_sum_minus_logit": float(expl_gap)}
    (OUT / "shap_check.json").write_text(json.dumps(shap_check, indent=1), encoding="utf-8")
    print("SHAP check", shap_check, flush=True)

    # ------------------------------------------------------------------ KAAM vs additive EBM shapes
    # The full-trace canonical split has an additive EBM only if its single fit finished (deviation 4); otherwise
    # the comparison uses the canonical split of the original sample, where the EBMs were fitted on every split.
    if (CANON / "ebm_seed0.pkl").exists():
        shape_ds, s_dir, s_kaam, s_pp, s_Xtr = "full", CANON, canon, pp, X_train
    else:
        shape_ds, s_dir = "original", RUNS / "original" / "random" / "s42"
        d_o = load_dataset("original")
        s_kaam, s_pp = kaam_models(s_dir)[0], load_pp(s_dir)
        s_Xtr = d_o["X"].iloc[load_split(s_dir)["train"]]
    with open(s_dir / "ebm_seed0.pkl", "rb") as fh:
        ebm = pickle.load(fh)
    median_scaled = np.median(s_pp.transform(s_Xtr).to_numpy(np.float32), axis=0)
    (OUT / "ebm_shape_source.json").write_text(json.dumps({"dataset": shape_ds, "run": str(s_dir.relative_to(ROOT))}),
                                               encoding="utf-8")
    rows, corr = [], []
    for i, n in enumerate(FEATURE_NAMES):
        lo, hi = window(s_Xtr, n)
        grid_raw = np.linspace(lo, hi, N_GRID)
        grid = s_pp.to_scaled(n, grid_raw)
        k = phi(s_kaam, n, grid)
        Z = np.tile(median_scaled, (N_GRID, 1)).astype(np.float64)
        Z[:, i] = grid
        e = ebm.eval_terms(Z)[:, list(ebm.term_names_).index(n)]
        k, e = k - k.mean(), e - e.mean()
        r = float(np.corrcoef(k, e)[0, 1]) if np.std(k) > 0 and np.std(e) > 0 else np.nan
        corr.append({"feature": n, "pearson_r": r, "kaam_swing": float(k.max() - k.min()),
                     "ebm_swing": float(e.max() - e.min()), "window_low": lo, "window_high": hi})
        rows += [{"feature": n, "x": xr, "kaam": a, "ebm": b} for xr, a, b in zip(grid_raw, k, e)]
    pd.DataFrame(corr).to_csv(OUT / "ebm_shape_corr.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "shapes_kaam_ebm.csv", index=False)
    print(pd.DataFrame(corr).round(3).to_string(), flush=True)

    # ------------------------------------------------------------------ all ten curves
    orig_dir = RUNS / "original" / "random" / "s42"
    orig_model = kaam_models(orig_dir)[0] if (orig_dir / "kaam_states.pt").exists() else None
    orig_pp = load_pp(orig_dir) if orig_model is not None else None
    rows = []
    for n in FEATURE_NAMES:
        lo, hi = window(X_train, n)
        grid_raw = np.linspace(lo, hi, N_GRID)
        for s, m in enumerate(models):
            v = phi(m, n, pp.to_scaled(n, grid_raw))
            rows += [{"feature": n, "model": f"full seed {s}", "x": xr, "phi": p} for xr, p in zip(grid_raw, v - v.mean())]
        if orig_model is not None:
            v = phi(orig_model, n, orig_pp.to_scaled(n, np.clip(grid_raw, *orig_pp.clip_bounds[n])))
            rows += [{"feature": n, "model": "original seed 0", "x": xr, "phi": p}
                     for xr, p in zip(grid_raw, v - v.mean())]
    curves = pd.DataFrame(rows)
    curves.to_csv(OUT / "curves_canonical.csv", index=False)

    # ------------------------------------------------------------------ plan_cpu across protocols
    lo, hi = window(X_train, "plan_cpu")
    grid_raw = np.linspace(lo, hi, N_GRID)
    rows, summ, ref = [], [], None
    for proto in ["random", "time", "grouped", "louo"]:
        pdir = RUNS / "full" / proto
        if not pdir.exists():
            continue
        for sdir in sorted(pdir.iterdir()):
            if not (sdir / "kaam_states.pt").exists():
                continue
            if proto == "random" and sdir.name != "s42":
                continue
            m = kaam_models(sdir)[0]
            p_ = load_pp(sdir)
            v = phi(m, "plan_cpu", p_.to_scaled("plan_cpu", np.clip(grid_raw, *p_.clip_bounds["plan_cpu"])))
            v = v - v.mean()
            label = f"{proto}/{sdir.name}"
            if ref is None:
                ref = v
            rows += [{"run": label, "x": xr, "phi": p} for xr, p in zip(grid_raw, v)]
            summ.append({"run": label, "swing": float(v.max() - v.min()),
                         "corr_vs_canonical": float(np.corrcoef(v, ref)[0, 1]) if np.std(v) > 0 else np.nan,
                         "max_gap_vs_canonical": float(np.max(np.abs(v - ref)))})
    pd.DataFrame(rows).to_csv(OUT / "plan_cpu_protocols.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "plan_cpu_summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string(), flush=True)

    # ------------------------------------------------------------------ case audits (canonical model, seed 0)
    te = split["test"]
    p_test = np.load(CANON / "kaam.npz")["test_pred"][0].astype(float)
    y_test = test_labels(y, te, dataset="full", protocol="random", split="s42",
                         purpose="case audits on the canonical KAAM (scripts/interpret_v2.py)")
    flagged = p_test >= THRESHOLD
    groups = {"True positive": (y_test == 1) & flagged, "True negative": (y_test == 0) & ~flagged,
              "False negative": (y_test == 1) & ~flagged, "False positive": (y_test == 0) & flagged}
    means = {n: float(phi(canon, n, xs_train[:, i]).mean()) for i, n in enumerate(FEATURE_NAMES)}
    baseline = canon.bias.item() + sum(means.values())
    u1, _ = dominant_users()
    req_cols = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num"]
    train_req = X_train[req_cols].round(6)
    train_users = d["user_raw"][split["train"]]
    y_train = y[split["train"]]
    cases = {"baseline_logit": baseline, "baseline_risk": float(1 / (1 + np.exp(-baseline))),
             "threshold": THRESHOLD, "train_tasks": int(len(split["train"])), "cases": []}
    for name, mask in groups.items():
        idx = np.flatnonzero(mask)
        pick = idx[np.argmin(p_test[idx])] if name == "True negative" else idx[np.argmax(p_test[idx])]
        row_raw = X.iloc[te[pick]]
        xs = pp.transform(X.iloc[[te[pick]]]).to_numpy(np.float32)[0]
        contrib = canon.explain(torch.as_tensor(xs))
        centred_c = {n: contrib[n] - means[n] for n in FEATURE_NAMES}
        model_logit = float(logit(canon, xs[None])[0])
        same_req = (train_req == row_raw[req_cols].round(6).to_numpy()).all(axis=1).to_numpy()
        pct = {n: float(50 * ((X_train[n] < row_raw[n]).mean() + (X_train[n] <= row_raw[n]).mean()))
               for n in FEATURE_NAMES}
        cases["cases"].append({
            "case": name, "risk": float(p_test[pick]), "label": int(y_test[pick]), "group_size": int(len(idx)),
            "tied_at_risk": int(np.sum(p_test[idx] == p_test[pick])),
            "raw": {n: float(row_raw[n]) for n in FEATURE_NAMES}, "percentile": pct,
            "contribution": centred_c, "model_logit": model_logit,
            "sum_check": float(abs(sum(contrib.values()) + canon.bias.item() - model_logit)),
            "same_request_train_tasks": int(same_req.sum()),
            "same_request_failure_rate": float(y_train[same_req].mean()) if same_req.any() else None,
            "same_request_user1_share": float(np.mean(train_users[same_req] == u1)) if same_req.any() else None,
            "user1": bool(d["user_raw"][te[pick]] == u1)})
    (OUT / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    for c in cases["cases"]:
        top = sorted(c["contribution"].items(), key=lambda kv: -abs(kv[1]))[:3]
        print(c["case"], round(c["risk"], 3), [(k, round(v, 3)) for k, v in top], "same req:",
              c["same_request_train_tasks"], c["same_request_failure_rate"], c["same_request_user1_share"], flush=True)

    # ------------------------------------------------------------------ user concentration
    rows = []
    for name in ["full", "original"]:
        dd = d if name == "full" else load_dataset("original")
        t = pd.DataFrame({"u": dd["user_raw"], "y": dd["y"]}).groupby("u")["y"].agg(["size", "sum"])
        by_fail = np.sort(t["sum"].to_numpy())[::-1]
        by_task = np.sort(t["size"].to_numpy())[::-1]
        for k in range(1, min(len(t), 200) + 1):
            rows.append({"dataset": name, "k": k, "failure_share": by_fail[:k].sum() / by_fail.sum(),
                         "task_share": by_task[:k].sum() / by_task.sum(), "users": len(t)})
    pd.DataFrame(rows).to_csv(OUT / "user_concentration.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
