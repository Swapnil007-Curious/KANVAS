"""Score the saved test predictions and run the statistics of paper/PROTOCOL.md (section 5).

    python scripts/analyze.py --dataset full
    python scripts/analyze.py --dataset original

Test labels are obtained only through kanvas.testaccess.test_labels (one logged access per
dataset, protocol and split). Writes results/analysis/<dataset>/:

  metrics_runs.csv        one row per (protocol, split, subset, model, seed): ROC-AUC, PR-AUC, Brier, ECE
  summary.csv             one row per (protocol, subset, model): the protocol score (mean over test sets of
                          the mean over seeds), std over test sets, mean seed std, and the other metrics
  comparisons.csv         KAAM minus each model per (protocol, subset): difference, 95% CI, p, Holm p
  chance_tests.csv        user-grouped protocol: each model's fold-mean AUC against 0.5 (user bootstrap)
  fold_auc.csv            user-grouped protocol: per-fold AUC with a user-bootstrap interval
  mechanism.csv           user-grouped protocol: the one-core signal among training and test users
  reliability.csv         reliability-diagram bins per (protocol, model), seed 0, pooled over test sets
  selection.csv           the chosen hyperparameters and early-stopping facts per split and model
  model_ci.csv            each model's protocol score with its 95% bootstrap interval
  split_facts.csv         sizes, failure rates, users, known-user share and start-time range of every split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kanvas.metrics import (N_BOOT, AucScorer, bootstrap_models, bootstrap_p, holm, percentile_ci,  # noqa: E402
                            all_metrics, reliability, resample_weights, roc_auc)
from kanvas.protocols import known_user_mask  # noqa: E402
from kanvas.testaccess import test_labels  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_experiments import ALL_MODELS, RUNS, load_dataset  # noqa: E402

OUT = ROOT / "results" / "analysis"
REFERENCE = "kaam"
PROTOCOL_ORDER = ["random", "time", "grouped", "louo"]


def available(dataset):
    """(protocol, split) -> {model: predictions (n_seeds, n_test)} for every finished split."""
    found = {}
    for proto in PROTOCOL_ORDER:
        pdir = RUNS / dataset / proto
        if not pdir.exists():
            continue
        for sdir in sorted(pdir.iterdir()):
            split = sdir / "split.npz"
            if not split.exists():
                continue
            preds = {m: np.load(sdir / f"{m}.npz")["test_pred"] for m in ALL_MODELS
                     if (sdir / f"{m}.npz").exists() and (sdir / f"{m}.json").exists()}
            if preds:
                z = np.load(split)
                found[(proto, sdir.name)] = {"split": {k: z[k] for k in ("train", "val", "test")}, "preds": preds,
                                             "dir": sdir}
    return found


def families(found):
    """Group test sets into the reported units: (protocol, subset) -> list of (split name, row mask)."""
    fam = {}
    for (proto, split), item in found.items():
        if proto == "louo":
            fam.setdefault((f"louo-{split}", "all"), []).append((proto, split, None))
        elif proto == "time":
            fam.setdefault(("time", "all"), []).append((proto, split, None))
            fam.setdefault(("time", "known users"), []).append((proto, split, "known"))
            fam.setdefault(("time", "new users"), []).append((proto, split, "new"))
        else:
            fam.setdefault((proto, "all"), []).append((proto, split, None))
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["original", "full"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    d = load_dataset(args.dataset)
    out = OUT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    found = available(args.dataset)

    labels, masks = {}, {}
    for (proto, split), item in found.items():
        te = item["split"]["test"]
        labels[(proto, split)] = test_labels(d["y"], te, dataset=args.dataset, protocol=proto, split=split,
                                             purpose="score saved test predictions (scripts/analyze.py)")
        if proto == "time":
            masks[(proto, split, "known")] = known_user_mask(d["user"], item["split"])
            masks[(proto, split, "new")] = ~masks[(proto, split, "known")]

    # ------------------------------------------------------------------ per-run metrics
    rows = []
    for (proto_name, subset), members in families(found).items():
        for proto, split, mask_key in members:
            item = found[(proto, split)]
            y = labels[(proto, split)]
            m = np.ones(len(y), bool) if mask_key is None else masks[(proto, split, mask_key)]
            for model, P in item["preds"].items():
                for s in range(P.shape[0]):
                    rows.append({"protocol": proto_name, "subset": subset, "split": split, "model": model, "seed": s,
                                 "n_test": int(m.sum()), "failure_rate": float(y[m].mean()),
                                 **all_metrics(y[m], P[s][m])})
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "metrics_runs.csv", index=False)

    # ------------------------------------------------------------------ summary
    per_set = runs.groupby(["protocol", "subset", "model", "split"]).agg(
        roc_auc=("roc_auc", "mean"), seed_std=("roc_auc", "std"), pr_auc=("pr_auc", "mean"),
        brier=("brier", "mean"), ece=("ece", "mean"), n_seeds=("seed", "size"), n_test=("n_test", "first"),
        failure_rate=("failure_rate", "first")).reset_index()
    summary = per_set.groupby(["protocol", "subset", "model"]).agg(
        roc_auc=("roc_auc", "mean"), set_std=("roc_auc", "std"), seed_std=("seed_std", "mean"),
        pr_auc=("pr_auc", "mean"), brier=("brier", "mean"), ece=("ece", "mean"), n_sets=("split", "size"),
        n_seeds=("n_seeds", "max"), n_test=("n_test", "sum"), failure_rate=("failure_rate", "mean")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    # ------------------------------------------------------------------ paired comparisons with KAAM
    comp_rows, chance_rows, boot_store = [], [], {}
    for (proto_name, subset), members in families(found).items():
        test_sets = []
        for proto, split, mask_key in members:
            item = found[(proto, split)]
            y = labels[(proto, split)]
            m = np.ones(len(y), bool) if mask_key is None else masks[(proto, split, mask_key)]
            ts = {"y": y[m], "scores": {k: [P[s][m] for s in range(P.shape[0])] for k, P in item["preds"].items()}}
            if proto == "grouped":
                ts["clusters"] = pd.factorize(d["user"][item["split"]["test"]][m])[0]
            test_sets.append(ts)
        common = set.intersection(*(set(ts["scores"]) for ts in test_sets))
        for ts in test_sets:
            ts["scores"] = {k: v for k, v in ts["scores"].items() if k in common}
        point, boot = bootstrap_models(test_sets, n_boot=args.n_boot, seed=0)
        boot_store[(proto_name, subset)] = (point, boot)
        others = [m for m in ALL_MODELS if m in common and m != REFERENCE]
        if REFERENCE in common:
            ps = []
            for other in others:
                diff = boot[REFERENCE] - boot[other]
                lo, hi = percentile_ci(diff)
                ps.append(bootstrap_p(diff))
                comp_rows.append({"protocol": proto_name, "subset": subset, "model": other,
                                  "kaam_auc": point[REFERENCE], "other_auc": point[other],
                                  "diff": point[REFERENCE] - point[other], "ci_low": lo, "ci_high": hi,
                                  "p": ps[-1], "unit": "user" if proto_name == "grouped" else "task",
                                  "n_sets": len(test_sets)})
            adj = holm(ps) if ps else []
            for row, a in zip(comp_rows[-len(ps):], adj):
                row["p_holm"] = float(a)
                row["significant"] = bool(a < 0.05)
                row["within_margin"] = bool(-0.01 <= row["ci_low"] and row["ci_high"] <= 0.01)
        if proto_name == "grouped" and subset == "all":
            ps = []
            for model in [m for m in ALL_MODELS if m in common]:
                lo, hi = percentile_ci(boot[model])
                ps.append(bootstrap_p(boot[model], null=0.5))
                chance_rows.append({"model": model, "auc": point[model], "ci_low": lo, "ci_high": hi, "p": ps[-1]})
            for row, a in zip(chance_rows, holm(ps)):
                row["p_holm"] = float(a)
                row["significant"] = bool(a < 0.05)
    pd.DataFrame(comp_rows).to_csv(out / "comparisons.csv", index=False)
    pd.DataFrame(chance_rows).to_csv(out / "chance_tests.csv", index=False)
    ci_rows = []
    for (proto_name, subset), (point, boot) in boot_store.items():
        for model in [m for m in ALL_MODELS if m in point]:
            lo, hi = percentile_ci(boot[model])
            ci_rows.append({"protocol": proto_name, "subset": subset, "model": model, "auc": point[model],
                            "ci_low": lo, "ci_high": hi, "unit": "user" if proto_name == "grouped" else "task"})
    pd.DataFrame(ci_rows).to_csv(out / "model_ci.csv", index=False)

    # ------------------------------------------------------------------ split facts
    fact_rows = []
    day = 86400.0
    for (proto, split), item in found.items():
        sp = item["split"]
        y_te = labels[(proto, split)]
        tr_users = np.unique(d["user"][sp["train"]])
        known = np.isin(d["user"][sp["test"]], tr_users)
        row = {"protocol": proto, "split": split, "n_train": len(sp["train"]), "n_val": len(sp["val"]),
               "n_test": len(sp["test"]), "rate_train": float(d["y"][sp["train"]].mean()),
               "rate_val": float(d["y"][sp["val"]].mean()), "rate_test": float(y_te.mean()),
               "users_train": len(tr_users), "users_val": int(len(np.unique(d["user"][sp["val"]]))),
               "users_test": int(len(np.unique(d["user"][sp["test"]]))), "test_known_share": float(known.mean()),
               "test_known_n": int(known.sum()), "test_new_n": int((~known).sum()),
               "test_known_rate": float(y_te[known].mean()) if known.any() else np.nan,
               "test_new_rate": float(y_te[~known].mean()) if (~known).any() else np.nan}
        for part in ("train", "val", "test"):
            st = d["start"][sp[part]]
            row[f"start_{part}_min_day"], row[f"start_{part}_max_day"] = float(st.min() / day), float(st.max() / day)
        fact_rows.append(row)
    pd.DataFrame(fact_rows).to_csv(out / "split_facts.csv", index=False)

    # ------------------------------------------------------------------ user-grouped: per fold and mechanism
    fold_rows, mech_rows = [], []
    rng = np.random.default_rng(0)
    plan_cpu = d["X"]["plan_cpu"].to_numpy()
    for (proto, split), item in sorted(found.items()):
        if proto != "grouped":
            continue
        te, tr = item["split"]["test"], item["split"]["train"]
        y = labels[(proto, split)]
        clusters = pd.factorize(d["user"][te])[0]
        weights = [resample_weights(len(y), rng, clusters) for _ in range(args.n_boot)]
        for model, P in item["preds"].items():
            scorers = [AucScorer(y, P[s]) for s in range(P.shape[0])]
            pt = float(np.mean([sc() for sc in scorers]))
            bs = np.array([np.mean([sc(w) for sc in scorers]) for w in weights])
            lo, hi = percentile_ci(bs)
            fold_rows.append({"fold": split, "model": model, "auc": pt, "ci_low": lo, "ci_high": hi,
                              "p_vs_chance": bootstrap_p(bs, null=0.5), "n_test": len(y),
                              "test_users": int(clusters.max() + 1), "failure_rate": float(y.mean())})
        ytr = d["y"][tr]  # training labels are not test labels
        one_tr, one_te = plan_cpu[tr] == 100, plan_cpu[te] == 100
        mech_rows.append({
            "fold": split, "train_tasks": len(tr), "test_tasks": len(te),
            "train_users": int(len(np.unique(d["user"][tr]))), "test_users": int(clusters.max() + 1),
            "train_one_core_share": float(one_tr.mean()), "test_one_core_share": float(one_te.mean()),
            "train_one_core_rate": float(ytr[one_tr].mean()) if one_tr.any() else np.nan,
            "train_other_rate": float(ytr[~one_tr].mean()),
            "test_one_core_rate": float(y[one_te].mean()) if one_te.any() else np.nan,
            "test_other_rate": float(y[~one_te].mean()),
            "train_auc_neg_plan_cpu": roc_auc(ytr, -plan_cpu[tr]), "test_auc_neg_plan_cpu": roc_auc(y, -plan_cpu[te]),
            "test_has_user1": bool(np.isin(d["user_raw"][te], [_user1(d)]).any())})
    pd.DataFrame(fold_rows).to_csv(out / "fold_auc.csv", index=False)
    pd.DataFrame(mech_rows).to_csv(out / "mechanism.csv", index=False)

    # ------------------------------------------------------------------ reliability, seed 0 pooled over test sets
    rel_rows = []
    for (proto_name, subset), members in families(found).items():
        if subset != "all":
            continue
        for model in ALL_MODELS:
            if not all(model in found[(proto, split)]["preds"] for proto, split, _ in members):
                continue  # only models fitted on every test set of the protocol (not the canonical-only EBM)
            ys, ps = [], []
            for proto, split, _ in members:
                item = found[(proto, split)]
                ys.append(labels[(proto, split)])
                ps.append(item["preds"][model][0])
            r = reliability(np.concatenate(ys), np.concatenate(ps))
            for b in range(len(r["count"])):
                rel_rows.append({"protocol": proto_name, "model": model, "bin": b, "lower": r["lower"][b],
                                 "upper": r["upper"][b], "count": int(r["count"][b]), "mean_p": r["mean_p"][b],
                                 "rate": r["rate"][b]})
    pd.DataFrame(rel_rows).to_csv(out / "reliability.csv", index=False)

    # ------------------------------------------------------------------ selection facts
    sel = []
    for (proto, split), item in found.items():
        for model in item["preds"]:
            meta = json.loads((item["dir"] / f"{model}.json").read_text(encoding="utf-8"))
            ps_ = meta["per_seed"]
            sel.append({"protocol": proto, "split": split, "model": model, "chosen": json.dumps(meta["chosen"]),
                        "n_seeds": len(ps_), "val_auc_seed0": ps_[0]["val_auc"],
                        "early_stopped_all": all(p.get("early_stopped", True) for p in ps_),
                        "max_epochs_run": max((p.get("epochs_run") or 0) for p in ps_),
                        "mean_best_epoch": float(np.mean([p.get("best_epoch") or p.get("best_iteration") or 0
                                                          for p in ps_])),
                        "seconds": float(sum(p.get("seconds", 0) for p in ps_))})
    pd.DataFrame(sel).to_csv(out / "selection.csv", index=False)
    print(summary.round(4).to_string())
    print(pd.DataFrame(comp_rows).round(4).to_string())


_U1 = {}


def _user1(d):
    if "u1" not in _U1:
        from run_experiments import dominant_users
        _U1["u1"] = dominant_users()[0]
    return _U1["u1"]


if __name__ == "__main__":
    main()
