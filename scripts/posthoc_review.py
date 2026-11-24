"""Post hoc analyses added after the independent review (not in paper/PROTOCOL.md).

    python scripts/posthoc_review.py

Every analysis uses the saved test predictions; no model is fitted. Test labels are read only through
kanvas.testaccess.test_labels, so each read is logged with the purpose "post hoc review analysis". Writes
results/posthoc/:

  grouped_split.csv    user-grouped protocol summarised separately over the folds whose test users exclude User 1
                       (user-level bootstrap over those folds, paired differences with KAAM, Holm) and for the fold
                       that holds User 1
  louo_like.csv        like-for-like check: the canonical models (trained with User 1) scored on the test tasks of
                       split 42 that do not belong to User 1 (or to Users 1 and 4)
  duplicates.csv       share of test tasks whose ten raw features exactly match a training task, per split; on split
                       42, the AUC of the identical training tasks' mean label on those tasks and each model's AUC on
                       duplicated and non-duplicated test tasks
  cluster_boot.csv     KAAM minus each model under the random, time-ordered and leave-user-out protocols with whole
                       users resampled (instead of tasks), Holm within each protocol and subset
  time_facts.json      composition of the time-ordered split: User 1's share and the failure rate without User 1 in
                       each period, the gap between periods, and the users behind the new-user test tasks
  parts/               each analysis's rows per dataset, saved as soon as it finishes, so an interrupted run
                       resumes where it stopped (delete the folder to recompute everything)
  onecore.csv          the plan_cpu contribution of a one-core request relative to the mean training task, for the
                       first seed of every KAAM run of the full trace, with the one-core and other failure rates of
                       its training tasks
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
sys.path.insert(0, str(ROOT / "scripts"))

from kanvas.data import FEATURE_NAMES  # noqa: E402
from kanvas.metrics import AucScorer, bootstrap_models, bootstrap_p, holm, percentile_ci, roc_auc  # noqa: E402
from kanvas.testaccess import test_labels  # noqa: E402
from run_experiments import ALL_MODELS, RUNS, dominant_users, load_dataset  # noqa: E402

OUT = ROOT / "results" / "posthoc"
PARTS = OUT / "parts"  # one JSON per (dataset, analysis); a rerun skips the parts already saved
PURPOSE = "post hoc review analysis (scripts/posthoc_review.py)"
N_BOOT = 2000


def split_of(run_dir):
    z = np.load(run_dir / "split.npz")
    return {k: z[k] for k in ("train", "val", "test")}


def preds_of(run_dir):
    return {m: np.load(run_dir / f"{m}.npz")["test_pred"].astype(float) for m in ALL_MODELS
            if (run_dir / f"{m}.npz").exists()}


def compare(test_sets, ref="kaam"):
    """Bootstrap point/CI per model and paired KAAM-minus-model differences with Holm."""
    common = set.intersection(*(set(ts["scores"]) for ts in test_sets))
    for ts in test_sets:
        ts["scores"] = {k: v for k, v in ts["scores"].items() if k in common}
    point, boot = bootstrap_models(test_sets, n_boot=N_BOOT, seed=0)
    rows, ps = [], []
    for m in [m for m in ALL_MODELS if m in common]:
        lo, hi = percentile_ci(boot[m])
        row = {"model": m, "auc": point[m], "ci_low": lo, "ci_high": hi}
        if m != ref:
            d = boot[ref] - boot[m]
            dlo, dhi = percentile_ci(d)
            ps.append(bootstrap_p(d))
            row.update(diff=point[ref] - point[m], diff_low=dlo, diff_high=dhi, p=ps[-1])
        rows.append(row)
    adj = iter(holm(ps)) if ps else iter([])
    for row in rows:
        if "p" in row:
            row["p_holm"] = float(next(adj))
    return rows


def _jsonable(o):
    """JSON fallback for numpy scalars."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def log(msg):
    try:
        import psutil

        mem = f" [free RAM {psutil.virtual_memory().available / 2**30:.1f} GB]"
    except ImportError:
        mem = ""
    print(f"{time.strftime('%H:%M:%S')} {msg}{mem}", flush=True)


def cached(ds, name, fn):
    """Run one analysis part unless its result is already saved; parts survive an interrupted run."""
    path = PARTS / f"{ds}_{name}.json"
    if path.exists():
        log(f"{ds} {name}: saved part found, skipped")
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    rows = fn()
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(rows, default=_jsonable), encoding="utf-8")
    tmp.replace(path)
    log(f"{ds} {name}: done in {time.perf_counter() - t0:.0f} s")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    u1, u4 = dominant_users()
    out_rows = {}
    for ds in ["full", "original"]:
        d = load_dataset(ds)
        y, users, X = d["y"], d["user_raw"], d["X"]
        ucode = d["user"]
        labels = {}

        def lab(proto, split, idx):
            key = (proto, split)
            if key not in labels:
                labels[key] = test_labels(y, idx, dataset=ds, protocol=proto, split=split, purpose=PURPOSE)
            return labels[key]

        # ---------------------------------------------------------------- grouped: User-1 fold apart
        def grouped_split():
            g_rows = []
            sets_nou, set_u1 = [], None
            for k in range(5):
                rd = RUNS / ds / "grouped" / f"f{k}"
                sp, pr = split_of(rd), preds_of(rd)
                yt = lab("grouped", f"f{k}", sp["test"])
                ts = {"y": yt, "scores": {m: list(P) for m, P in pr.items()},
                      "clusters": pd.factorize(ucode[sp["test"]])[0]}
                if np.isin(u1, users[sp["test"]]):
                    set_u1 = (f"f{k}", ts)
                else:
                    sets_nou.append(ts)
            for r in compare(sets_nou):
                g_rows.append({"dataset": ds, "folds": "without User 1", "n_folds": len(sets_nou), **r})
            name, ts = set_u1
            for r in compare([ts]):
                g_rows.append({"dataset": ds, "folds": f"with User 1 ({name})", "n_folds": 1, **r})
            return g_rows

        out_rows.setdefault("grouped_split", []).extend(cached(ds, "grouped_split", grouped_split))

        # ---------------------------------------------------------------- louo like-for-like on split 42
        def louo_like():
            rows = []
            rd = RUNS / ds / "random" / "s42"
            sp, pr = split_of(rd), preds_of(rd)
            yt = lab("random", "s42", sp["test"])
            ut = users[sp["test"]]
            for name, drop in [("without User 1", [u1]), ("without Users 1 and 4", [u1, u4])]:
                keep = ~np.isin(ut, drop)
                ts = {"y": yt[keep], "scores": {m: [p[keep] for p in P] for m, P in pr.items()}}
                for r in compare([ts]):
                    rows.append({"dataset": ds, "population": name, "trained_with": "all users (random split 42)",
                                 "n_test": int(keep.sum()), **r})
                for split_name in (["d1"] if name == "without User 1" else ["d14"]):
                    rl = RUNS / ds / "louo" / split_name
                    spl, prl = split_of(rl), preds_of(rl)
                    ytl = lab("louo", split_name, spl["test"])
                    for r in compare([{"y": ytl, "scores": {m: list(P) for m, P in prl.items()}}]):
                        rows.append({"dataset": ds, "population": name,
                                     "trained_with": f"{name} (leave-user-out split {split_name})",
                                     "n_test": int(len(ytl)), **r})
            return rows

        out_rows.setdefault("louo_like", []).extend(cached(ds, "louo_like", louo_like))

        # ---------------------------------------------------------------- exact duplicates
        def duplicates():
            rows = []
            h = pd.util.hash_pandas_object(X[FEATURE_NAMES], index=False).to_numpy()
            for proto, splits in [("random", [f"s{s}" for s in range(42, 47)]), ("time", ["t"]),
                                  ("grouped", [f"f{k}" for k in range(5)]), ("louo", ["d1", "d14"])]:
                for s_ in splits:
                    rd = RUNS / ds / proto / s_
                    sp = split_of(rd)
                    dup = np.isin(h[sp["test"]], h[sp["train"]])
                    row = {"dataset": ds, "protocol": proto, "split": s_, "n_test": int(len(dup)),
                           "dup_share": float(dup.mean()), "dup_n": int(dup.sum())}
                    if proto == "random" and s_ == "s42":
                        yt = lab(proto, s_, sp["test"])
                        tr = pd.DataFrame({"h": h[sp["train"]], "y": y[sp["train"]]}).groupby("h")["y"].mean()
                        look = pd.Series(h[sp["test"]][dup]).map(tr).to_numpy()
                        row["lookup_auc_on_dups"] = roc_auc(yt[dup], look)
                        row["dup_user1_share"] = float(np.mean(users[sp["test"]][dup] == u1))
                        row["dup_rate"] = float(yt[dup].mean())
                        row["nondup_rate"] = float(yt[~dup].mean())
                        for m, P in preds_of(rd).items():
                            row[f"{m}_auc_dup"] = float(np.mean([roc_auc(yt[dup], p[dup]) for p in P]))
                            row[f"{m}_auc_nondup"] = float(np.mean([roc_auc(yt[~dup], p[~dup]) for p in P]))
                    rows.append(row)
            return rows

        out_rows.setdefault("duplicates", []).extend(cached(ds, "duplicates", duplicates))

        # ---------------------------------------------------------------- user-cluster bootstrap
        fams = {"random": [("random", f"s{s}", None) for s in range(42, 47)],
                "time": [("time", "t", None)], "time known users": [("time", "t", "known")],
                "time new users": [("time", "t", "new")], "louo-d1": [("louo", "d1", None)],
                "louo-d14": [("louo", "d14", None)]}
        for fam, members in fams.items():
            def cluster_boot(fam=fam, members=members):
                rows, sets = [], []
                for proto, s_, sub in members:
                    rd = RUNS / ds / proto / s_
                    sp, pr = split_of(rd), preds_of(rd)
                    yt = lab(proto, s_, sp["test"])
                    mask = np.ones(len(yt), bool)
                    if sub is not None:
                        known = np.isin(ucode[sp["test"]], np.unique(ucode[sp["train"]]))
                        mask = known if sub == "known" else ~known
                    sets.append({"y": yt[mask], "scores": {m: [p[mask] for p in P] for m, P in pr.items()},
                                 "clusters": pd.factorize(ucode[sp["test"]][mask])[0]})
                n_users = int(sum(ts["clusters"].max() + 1 for ts in sets))
                for r in compare(sets):
                    rows.append({"dataset": ds, "protocol": fam, "test_users": n_users, **r})
                return rows

            part = "cluster_boot_" + fam.replace(" ", "_")
            out_rows.setdefault("cluster_boot", []).extend(cached(ds, part, cluster_boot))

        # ---------------------------------------------------------------- time split composition
        def time_facts():
            rd = RUNS / ds / "time" / "t"
            sp = split_of(rd)
            yt = lab("time", "t", sp["test"])
            facts = {}
            for part in ("train", "val", "test"):
                idx = sp[part]
                yy = yt if part == "test" else y[idx]
                u = users[idx]
                facts[part] = {"user1_share": float(np.mean(u == u1)), "rate": float(yy.mean()),
                               "rate_without_user1": float(yy[u != u1].mean()),
                               "start_min_day": float(d["start"][idx].min() / 86400),
                               "start_max_day": float(d["start"][idx].max() / 86400)}
            known = np.isin(ucode[sp["test"]], np.unique(ucode[sp["train"]]))
            facts["new_user_tasks"] = int((~known).sum())
            facts["new_users"] = int(len(np.unique(ucode[sp["test"]][~known])))
            facts["gap_train_to_test_days"] = facts["test"]["start_min_day"] - facts["train"]["start_max_day"]
            return facts

        out_rows.setdefault("time_facts", {})[ds] = cached(ds, "time_facts", time_facts)

        # ---------------------------------------------------------------- one-core contribution per run (full)
        if ds == "full":
            def onecore():
                from interpret_v2 import kaam_models, load_pp, phi
                rows = []
                for proto, splits in [("random", ["s42"]), ("time", ["t"]), ("grouped", [f"f{k}" for k in range(5)]),
                                      ("louo", ["d1", "d14"])]:
                    for s_ in splits:
                        rd = RUNS / ds / proto / s_
                        sp, pp = split_of(rd), load_pp(rd)
                        m = kaam_models(rd)[0]
                        xtr = X["plan_cpu"].to_numpy()[sp["train"]]
                        i = FEATURE_NAMES.index("plan_cpu")
                        scaled_tr = pp.transform(X.iloc[sp["train"]]).to_numpy(np.float32)[:, i]
                        one = float(pp.to_scaled("plan_cpu", np.clip(100.0, *pp.clip_bounds["plan_cpu"])))
                        contrib = float(phi(m, "plan_cpu", [one])[0] - phi(m, "plan_cpu", scaled_tr).mean())
                        ytr = y[sp["train"]]
                        is1 = xtr == 100
                        rows.append({"run": f"{proto}/{s_}", "onecore_vs_mean_task": contrib,
                                     "train_onecore_share": float(is1.mean()),
                                     "train_onecore_rate": float(ytr[is1].mean()),
                                     "train_other_rate": float(ytr[~is1].mean()),
                                     "train_has_user1": bool(np.isin(u1, users[sp["train"]]))})
                return rows

            out_rows["onecore"] = cached(ds, "onecore", onecore)
        del d, X

    for name, rows in out_rows.items():
        if name == "time_facts":
            (OUT / "time_facts.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        else:
            pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
            print(f"\n{name}\n", pd.DataFrame(rows).round(4).to_string())
    print(json.dumps(out_rows["time_facts"], indent=1))


if __name__ == "__main__":
    main()
