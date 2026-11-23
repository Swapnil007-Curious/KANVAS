"""Build app/kaam_demo.json from the canonical checkpoint and the published feature matrix.

    python app/export_demo.py

No torch needed. Every number the demo shows is checked here first, and the
script stops if any of them disagrees with the checkpoint or the paper:

  * test AUC / accuracy / F1 against the metrics stored inside the checkpoint
    (this also proves the published CSV keeps the checkpoint's row order)
  * every phi curve against data/processed/publication/v1/kanvas_phi_curves.csv
  * the average-task logit and the four audit cases against Section 6.10
  * the exact-request lookup against the 1,492-run / 88.5% figure in 6.10
"""

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from kaam_numpy import KAAMNumpy, load_torch_checkpoint

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CKPT = ROOT / "models" / "kaam_canonical.pt"
MATRIX = ROOT / "data" / "processed" / "publication" / "v1" / "kanvas_feature_matrix.csv"
CURVES = ROOT / "data" / "processed" / "publication" / "v1" / "kanvas_phi_curves.csv"
OUT = HERE / "kaam_demo.json"

REQUEST = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num"]
# Notebook 06's "exact request": the four request fields plus the machine type it ran on.
TEMPLATE_KEY = REQUEST + ["cap_gpu"]
CASES = [("True positive", 1, True, "max", "correctly caught failure with the highest predicted risk"),
         ("True negative", 0, False, "min", "correctly cleared task with the lowest predicted risk"),
         ("Missed failure", 1, False, "max", "failure the model missed, with the highest predicted risk"),
         ("False alarm", 0, True, "max", "false alarm with the highest predicted risk")]
# Section 6.10, Table: risk of each case as the paper reports it (3 d.p.).
PAPER_CASE_RISK = {"True positive": 0.778, "True negative": 0.024, "Missed failure": 0.494, "False alarm": 0.782}
PAPER_TP = {"plan_cpu": 1.397, "cap_gpu": 0.271, "avg_cpu_kernel": 0.263, "plan_gpu": 0.172, "avg_cpu_usr": 0.117,
            "plan_mem": 0.075, "inst_num": 0.070, "avg_load_1": 0.038, "avg_net_receive": 0.011}

# Numbers that need the user column, which no exported file carries. Quoted from the paper as written.
PAPER = {
    "users": 738,
    "top_user": {"tasks": 5864, "task_share": 31.04, "failures": 2816, "failure_share": 54.22, "failure_rate": 48.02},
    "false_alarm_request_one_user_share": 99.7,
    "auc": {  # random split, user-grouped split (KAAM: mean +/- sd of 3 seeds)
        "KAAM": [0.7520, 0.4535, 0.0418], "Logistic regression": [0.7346, 0.4193, None], "MLP": [0.7558, 0.4025, None]},
    "one_core_failure_rate": {"Training users": [80.4, 18.1], "Validation users": [23.8, 24.0],
                              "Test users": [34.4, 22.8]},
    "plan_cpu_swing": [2.02, 0.38],
    "auc_without_top_user": 0.661,
}
# Printed by notebooks/06_ablation_and_case_studies.ipynb (cell 42); they need the user column too.
NOTEBOOK06 = {
    "one_core_top_user": {"tasks": 1496, "failure_rate": 88.0},
    "one_core_other_users": {"tasks": 325, "users": 78, "failure_rate": 16.6},
    "not_one_core": {"tasks": 11401, "failure_rate": 19.9},
    "template_true_positives": [289, 366],
}


def main():
    ck = load_torch_checkpoint(CKPT)
    names = ck["feature_names"]
    sd = ck["state_dict"]
    params = {
        "feature_names": names,
        "ranges": {n: list(ck["feature_ranges"][n]) for n in names},
        "spline_order": ck["rule"]["spline_order"],
        "bias": float(sd["bias"]),
        "edges": {n: {"knots": sd[f"edges.{n}.knots"].tolist(), "spline_weight": sd[f"edges.{n}.spline_weight"].tolist(),
                      "base_weight": float(sd[f"edges.{n}.base_weight"])} for n in names},
    }
    n_params = sum(len(e["spline_weight"]) + 1 for e in params["edges"].values()) + 1
    assert n_params == 151, n_params

    df = pd.read_csv(MATRIX)
    X, y = df[names].to_numpy(np.float64), df["label"].to_numpy()
    pos = {role: ck["split_positions"][role] for role in ("train", "val", "test")}
    assert sum(len(p) for p in pos.values()) == len(df) == 18_890
    assert len(set(np.concatenate(list(pos.values())))) == len(df)

    # ---- 1. the model reproduces the checkpoint's own test metrics (float32, like torch)
    m32 = KAAMNumpy(params, dtype=np.float32)
    p_te = m32.predict(X[pos["test"]])
    auc = roc_auc_score(y[pos["test"]], p_te)
    acc = ((p_te >= 0.5) == y[pos["test"]]).mean()
    f1 = f1_score(y[pos["test"]], p_te >= 0.5)
    print(f"test AUC {auc:.10f} (checkpoint {ck['metrics']['test_auc']:.10f})  acc {acc:.6f}  F1 {f1:.6f}")
    assert abs(auc - ck["metrics"]["test_auc"]) < 1e-9
    assert abs(acc - ck["metrics"]["test_accuracy"]) < 1e-12
    assert abs(f1 - ck["metrics"]["test_f1"]) < 1e-12

    # the float64 model the app runs agrees with the float32 one
    m64 = KAAMNumpy(params)
    gap = np.abs(m64.logit(X) - m32.logit(X).astype(np.float64)).max()
    print(f"float64 vs float32 logit, all 18,890 tasks: max gap {gap:.2e}")
    assert gap < 1e-5

    # ---- 2. every phi matches the published curves
    Xtr = df.iloc[pos["train"]][names]
    pub = pd.read_csv(CURVES)
    worst = 0.0
    for n in names:
        lo, hi = Xtr[n].quantile([0.05, 0.95])
        grid = np.linspace(lo, hi, 200)
        mine = m32.phi(n, grid)
        mine = mine - mine.mean()
        theirs = pub[pub.feature_name == n]
        assert np.allclose(theirs.x_value.to_numpy(), grid, rtol=1e-9, atol=1e-6), n
        worst = max(worst, float(np.abs(mine - theirs.phi_value.to_numpy()).max()))
    print(f"phi curves vs published kanvas_phi_curves.csv: max gap {worst:.2e}")
    assert worst < 1e-5

    # ---- 3. centring offsets and the average task
    phi_mean = {n: float(m32.phi(n, Xtr[n].to_numpy()).astype(np.float64).mean()) for n in names}
    params["phi_mean"] = phi_mean
    m64 = KAAMNumpy(params)
    print(f"average training task logit {m64.baseline_logit:+.4f} (paper -1.163)")
    assert abs(m64.baseline_logit - (-1.163)) < 5e-4

    # ---- 4. the four audit cases, by the paper's fixed rule (threshold 0.5)
    yt = y[pos["test"]]
    flagged = p_te >= 0.5
    cases = []
    for label, truth, is_flagged, pick, rule in CASES:
        members = np.where((yt == truth) & (flagged == is_flagged))[0]
        i = members[np.argmin(p_te[members])] if pick == "min" else members[np.argmax(p_te[members])]
        row = X[pos["test"]][i]
        risk = float(m64.predict(row[None])[0])
        print(f"  {label:15s} risk {risk:.4f} (paper {PAPER_CASE_RISK[label]:.3f})  label {truth}  group {len(members)}")
        assert abs(risk - PAPER_CASE_RISK[label]) < 5e-4
        cases.append({"name": label, "rule": rule, "values": dict(zip(names, map(float, row))),
                      "true_label": int(truth), "risk": risk, "group_size": int(len(members))})
    tp = m64.explain([cases[0]["values"][n] for n in names])
    for n, v in PAPER_TP.items():
        assert abs(tp[n] - v) < 5e-4, (n, tp[n], v)
    print("  true-positive breakdown matches Section 6.10 to 3 d.p.")

    # ---- 5. exact-request lookup on the training split
    req = (Xtr[TEMPLATE_KEY].assign(label=y[pos["train"]]).groupby(TEMPLATE_KEY)["label"]
           .agg(["size", "sum"]).reset_index())
    fa = cases[3]["values"]
    template = {k: fa[k] for k in TEMPLATE_KEY}
    hit = req[(req[TEMPLATE_KEY] == pd.Series(template)).all(axis=1)].iloc[0]
    print(f"false-alarm request in training: {int(hit['size'])} runs, {hit['sum'] / hit['size']:.1%} failed "
          f"(paper 1,492, 88.5%)  template {template}")
    assert int(hit["size"]) == 1492 and round(100 * hit["sum"] / hit["size"], 1) == 88.5
    assert all(cases[0]["values"][k] == v for k, v in template.items()), "TP and FP should share the template"

    # test AUC with the template's test tasks removed (notebook 06: 0.6257; paper 6.10: 0.626)
    Xte = X[pos["test"]]
    is_tpl = np.all([Xte[:, names.index(k)] == v for k, v in template.items()], axis=0)
    auc_wo = roc_auc_score(yt[~is_tpl], p_te[~is_tpl])
    tpl_tp = int((is_tpl & (yt == 1) & flagged).sum())
    print(f"test AUC without the template: {auc_wo:.4f} (notebook 0.6257); template tasks in test: {is_tpl.sum()}, "
          f"true positives from it: {tpl_tp} of {int(((yt == 1) & flagged).sum())}")
    assert round(auc_wo, 4) == 0.6257
    assert [tpl_tp, int(((yt == 1) & flagged).sum())] == NOTEBOOK06["template_true_positives"]

    # ---- 6. what the sliders and curves need to know about the training data
    dist = {}
    for n in names:
        v = Xtr[n].to_numpy()
        uniq, counts = np.unique(v, return_counts=True)
        entry = {"p5": float(np.quantile(v, 0.05)), "p95": float(np.quantile(v, 0.95)),
                 "median": float(np.median(v)),
                 "lo": float(df[n].min()), "hi": float(df[n].max())}  # slider bounds: every split's values fit
        if len(uniq) <= 80:
            entry["values"], entry["counts"] = uniq.tolist(), counts.tolist()
        else:
            entry["quantiles"] = np.quantile(v, np.linspace(0, 1, 201)).tolist()
            c, edges = np.histogram(v, bins=40, range=params["ranges"][n])
            entry["hist"] = {"counts": c.tolist(), "edges": edges.tolist()}
        dist[n] = entry
    # discrete request features: every value seen in any split is selectable (so all four cases are reachable)
    for n in REQUEST + ["cap_gpu"]:
        dist[n]["choices"] = np.unique(df[n]).tolist()

    bundle = {
        "model": params,
        "threshold": 0.5,
        "metrics": {"test_auc": auc, "test_accuracy": float(acc), "test_tasks": int(len(pos["test"])),
                    "train_tasks": int(len(pos["train"])), "tasks": int(len(df)), "failure_rate": float(y.mean())},
        "train_dist": dist,
        "requests": {"columns": TEMPLATE_KEY, "rows": req[TEMPLATE_KEY].to_numpy().tolist(),
                     "runs": req["size"].astype(int).tolist(), "failed": req["sum"].astype(int).tolist()},
        "cases": cases,
        "template": {"values": template, "train_runs": int(hit["size"]), "train_failed": int(hit["sum"]),
                     "test_tasks": int(is_tpl.sum()), "test_auc_without": float(auc_wo)},
        "paper": PAPER,
        "notebook06": NOTEBOOK06,
    }
    OUT.write_text(json.dumps(bundle, separators=(",", ":")))
    print(f"wrote {OUT.name}: {OUT.stat().st_size / 1024:.0f} KB, {len(req)} distinct training requests")


if __name__ == "__main__":
    main()
