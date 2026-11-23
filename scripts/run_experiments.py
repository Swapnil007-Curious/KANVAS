"""Fit every model on every split of paper/PROTOCOL.md and write test-set predictions.

    python scripts/run_experiments.py --dataset original
    python scripts/run_experiments.py --dataset full --protocols random:s42,time,grouped,louo,random

This script never reads a test label. For each (dataset, protocol, split) it fits the training-only
preprocessing, chooses each model's hyperparameters on the validation split with seed 0, fits the
five seeds at the chosen setting and writes, under results/runs/<dataset>/<protocol>/<split>/:

  split.npz             train / val / test positions into the dataset arrays
  preprocess.json       the training-split clip bounds and scaling ranges
  <model>.npz           test-set failure probabilities, one row per seed
  <model>.json          the validation grid, the chosen setting and per-seed training facts
  kaam_states.pt        the fitted KAAM parameters per seed (curves, case audits, SHAP)
  ebm_seed0.pkl         the additive EBM of seed 0 (shape functions), canonical split only

Finished files are skipped, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kanvas import models_v2 as M  # noqa: E402
from kanvas.data import FEATURE_NAMES  # noqa: E402
from kanvas.metrics import roc_auc  # noqa: E402
from kanvas.preprocess import fit_preprocessor  # noqa: E402
from kanvas.protocols import (RANDOM_SPLIT_SEEDS, check_split, grouped_folds, leave_users_out,  # noqa: E402
                              random_split, time_split)

DATA = ROOT / "data" / "processed" / "v2" / "full_1h.parquet"
V1_LARGE = ROOT / "data" / "processed" / "kanvas_features_large.csv"
RUNS = ROOT / "results" / "runs"
SEEDS = [0, 1, 2, 3, 4]
TIE = 1e-4
ALL_MODELS = ["kaam", "lr", "mlp", "lgbm", "ebm", "ebm_int"]

torch.set_num_threads(M.N_THREADS)  # KANVAS_THREADS; several runner processes share the 4-thread CPU


def log(msg):
    try:
        import psutil

        mem = f" [free RAM {psutil.virtual_memory().available / 2**30:.1f} GB]"
    except ImportError:
        mem = ""
    print(f"{time.strftime('%H:%M:%S')} {msg}{mem}", flush=True)


def load_dataset(name):
    """Features (raw units), labels, start times and integer user codes, in a fixed row order."""
    cols = FEATURE_NAMES + ["label", "start_time", "user"]
    df = pd.read_parquet(DATA, columns=cols)
    if name == "original":
        v1_index = pd.read_csv(V1_LARGE, index_col=["job_name", "task_name"], usecols=["job_name", "task_name"]).index
        df = df.loc[v1_index]  # v1 row order, so split 42 reproduces the v1 canonical split
    elif name != "full":
        raise ValueError(name)
    users = df["user"].to_numpy()
    out = {"X": df[FEATURE_NAMES].reset_index(drop=True), "y": df["label"].to_numpy().astype(np.int64),
           "start": df["start_time"].to_numpy(dtype=float), "user_raw": users,
           "user": pd.factorize(users)[0].astype(np.int64)}
    return out


def dominant_users():
    """User 1 (most tasks) and User 4 (fourth by tasks) of the original sample, as raw ids."""
    d = load_dataset("original")
    counts = pd.Series(d["y"]).groupby(d["user_raw"]).agg(["size", "sum"]).sort_values(["size", "sum"],
                                                                                        ascending=False)
    return counts.index[0], counts.index[3]


def plan(name, d, protocols):
    u1, u4 = dominant_users()
    items = []
    for spec in protocols:
        proto, _, only = spec.partition(":")
        if proto == "random":
            items += [("random", f"s{s}", lambda s=s: random_split(d["y"], s)) for s in RANDOM_SPLIT_SEEDS]
        elif proto == "time":
            items.append(("time", "t", lambda: time_split(d["start"])))
        elif proto == "grouped":
            folds = None

            def fold(k):
                nonlocal folds
                if folds is None:
                    folds = grouped_folds(d["y"], d["user"])
                return folds[k]
            items += [("grouped", f"f{k}", lambda k=k: fold(k)) for k in range(5)]
        elif proto == "louo":
            items.append(("louo", "d1", lambda: leave_users_out(d["y"], d["user_raw"], [u1])))
            items.append(("louo", "d14", lambda: leave_users_out(d["y"], d["user_raw"], [u1, u4])))
        else:
            raise ValueError(spec)
        if only:
            items = [it for it in items if not (it[0] == proto and it[1] != only)]
    seen, ordered = set(), []
    for it in items:
        if (it[0], it[1]) not in seen:
            seen.add((it[0], it[1]))
            ordered.append(it)
    return ordered


def select(grid_scores, order):
    """Index of the chosen setting: best validation AUC; ties within TIE go to the earliest in ``order``."""
    best = max(grid_scores[i] for i in order)
    return next(i for i in order if grid_scores[i] >= best - TIE)


def run_model(model, out_dir, data, dataset_name, protocol):
    npz, meta_path = out_dir / f"{model}.npz", out_dir / f"{model}.json"
    if npz.exists() and meta_path.exists():
        return
    xtr, ytr, xva, yva, xte = data["xtr"], data["ytr"], data["xva"], data["yva"], data["xte"]
    # Seeds of the EBMs (PROTOCOL.md section 3, deviations 2 and 3): both use seed 0 only on the full trace; on the
    # original sample the additive EBM uses five seeds everywhere and the EBM with interactions uses five seeds on
    # the random splits (where seed variation is measured) and seed 0 under the other protocols.
    if model in ("ebm", "ebm_int") and dataset_name == "full":
        seeds = [0]
    elif model == "ebm_int" and protocol != "random":
        seeds = [0]
    else:
        seeds = SEEDS
    meta = {"model": model, "seeds": seeds}
    preds, per_seed = [], []

    if model == "kaam":
        dtr, dva, dte = data["kaam_designs"]
        grid = M.KAAM_LAMBDAS
        fits = [M.fit_kaam(dtr, ytr, dva, yva, 0, lam) for lam in grid]
        scores = [roc_auc(yva, f.predict(dva)) for f in fits]
        order = sorted(range(len(grid)), key=lambda i: -grid[i])  # largest lambda is the most regularised
        c = select(scores, order)
        meta.update(grid=[{"lambda": g} for g in grid], grid_val_auc=scores, chosen={"lambda": grid[c]})
        chosen = {0: fits[c]}
        for s in seeds[1:]:
            chosen[s] = M.fit_kaam(dtr, ytr, dva, yva, s, grid[c])
        states = []
        for s in seeds:
            f = chosen[s]
            preds.append(f.predict(dte))
            info = {k: v for k, v in f.info.items() if not k.endswith("_history")}
            per_seed.append({"seed": s, "val_auc": roc_auc(yva, f.predict(dva)), **info})
            states.append(M.kaam_state(f.model))
        meta["grid_info"] = [{k: v for k, v in f.info.items() if not k.endswith("_history")} for f in fits]
        meta["val_auc_history_seed0"] = chosen[0].info["val_auc_history"]
        meta["val_loss_history_seed0"] = chosen[0].info["val_loss_history"]
        torch.save(states, out_dir / "kaam_states.pt")

    elif model == "mlp":
        grid = M.MLP_GRID
        fits = [M.fit_mlp(xtr, ytr, xva, yva, 0, g["hidden"], g["lr"]) for g in grid]
        scores = [roc_auc(yva, f.predict(xva)) for f in fits]
        order = [2, 3, 0, 1]  # (12,) has fewer parameters than (8, 6); lower learning rate first
        c = select(scores, order)
        meta.update(grid=[{"hidden": list(g["hidden"]), "lr": g["lr"]} for g in grid], grid_val_auc=scores,
                    chosen={"hidden": list(grid[c]["hidden"]), "lr": grid[c]["lr"]})
        chosen = {0: fits[c]}
        for s in seeds[1:]:
            chosen[s] = M.fit_mlp(xtr, ytr, xva, yva, s, grid[c]["hidden"], grid[c]["lr"])
        for s in seeds:
            f = chosen[s]
            preds.append(f.predict(xte))
            info = {k: v for k, v in f.info.items() if not k.endswith("_history")}
            per_seed.append({"seed": s, "val_auc": roc_auc(yva, f.predict(xva)), **info})
        meta["grid_info"] = [{k: v for k, v in f.info.items() if not k.endswith("_history")} for f in fits]
        meta["val_auc_history_seed0"] = chosen[0].info["val_auc_history"]

    elif model == "lr":
        grid = M.LR_C_GRID
        fits = [M.fit_lr(xtr, ytr, c_) for c_ in grid]
        scores = [roc_auc(yva, f.predict(xva)) for f in fits]
        c = select(scores, list(range(len(grid))))  # smallest C is the most regularised
        f = fits[c]
        meta.update(grid=[{"C": g} for g in grid], grid_val_auc=scores, chosen={"C": grid[c]},
                    coef=f.model.coef_.ravel().tolist(), intercept=float(f.model.intercept_[0]),
                    note="deterministic: the five seeds give identical predictions, so it is fitted once")
        p = f.predict(xte)
        for s in seeds:
            preds.append(p)
            per_seed.append({"seed": s, "val_auc": scores[c], **f.info})

    elif model == "lgbm":
        grid = M.LGBM_GRID
        fits = [M.fit_lgbm(xtr, ytr, xva, yva, 0, **g) for g in grid]
        scores = [roc_auc(yva, f.predict(xva)) for f in fits]
        order = [1, 0, 3, 2]  # fewer leaves, larger leaves first
        c = select(scores, order)
        meta.update(grid=grid, grid_val_auc=scores, chosen=grid[c],
                    grid_info=[f.info for f in fits])
        chosen = {0: fits[c]}
        for s in seeds[1:]:
            chosen[s] = M.fit_lgbm(xtr, ytr, xva, yva, s, **grid[c])
        for s in seeds:
            f = chosen[s]
            preds.append(f.predict(xte))
            per_seed.append({"seed": s, "val_auc": roc_auc(yva, f.predict(xva)), **f.info})

    elif model in ("ebm", "ebm_int"):
        interactions = 0 if model == "ebm" else None
        for s in seeds:
            f = M.fit_ebm(xtr, ytr, s, interactions)
            preds.append(f.predict(xte))
            per_seed.append({"seed": s, "val_auc": roc_auc(yva, f.predict(xva)), **f.info})
            if model == "ebm" and s == 0 and protocol == "random" and out_dir.name == "s42":
                with open(out_dir / "ebm_seed0.pkl", "wb") as fh:
                    pickle.dump(f.model, fh)
            log(f"    {model} seed {s}: {f.info['seconds']:.0f}s")
        meta.update(grid=[], chosen={"interactions": "0" if interactions == 0 else "default"})
    else:
        raise ValueError(model)

    meta["per_seed"] = per_seed
    np.savez_compressed(npz, test_pred=np.asarray(preds, dtype=np.float32))
    meta_path.write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    log(f"  {model}: chosen {meta['chosen']}, val AUC (seed 0) {per_seed[0]['val_auc']:.4f}, "
        f"{sum(p.get('seconds', 0) for p in per_seed):.0f}s over {len(seeds)} seed(s)")


def claim(out_dir):
    """Queue mode: claim a split by creating its lock file atomically; False when another worker holds it."""
    try:
        fd = os.open(out_dir / ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()} {time.strftime('%H:%M:%S')}".encode())
    os.close(fd)
    return True


def run_split(dataset_name, d, protocol, split_name, make_split, models, queue=False):
    out_dir = RUNS / dataset_name / protocol / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name == "full":
        # deviation 4: on the full trace the EBMs are fitted only on the canonical split (additive EBM, seed 0)
        models = [m for m in models if m != "ebm_int" and (m != "ebm" or (protocol, split_name) == ("random", "s42"))]
    todo = [m for m in models if not ((out_dir / f"{m}.npz").exists() and (out_dir / f"{m}.json").exists())]
    if not todo:
        return
    if queue and not claim(out_dir):
        return
    try:
        _run_split(dataset_name, d, protocol, split_name, make_split, models, out_dir, todo)
    finally:
        if queue:
            (out_dir / ".lock").unlink(missing_ok=True)


def _run_split(dataset_name, d, protocol, split_name, make_split, models, out_dir, todo):
    split_file = out_dir / "split.npz"
    if split_file.exists():
        z = np.load(split_file)
        split = {k: z[k] for k in ("train", "val", "test")}
    else:
        split = make_split()
        check_split(split, groups=d["user"] if protocol == "grouped" else None)
        np.savez_compressed(split_file, **split)
    log(f"{dataset_name}/{protocol}/{split_name}: train {len(split['train']):,}, val {len(split['val']):,}, "
        f"test {len(split['test']):,}; models {todo}")
    X = d["X"]
    pp = fit_preprocessor(X.iloc[split["train"]])
    (out_dir / "preprocess.json").write_text(json.dumps({"clip_bounds": pp.clip_bounds, "ranges": pp.ranges},
                                                        indent=1), encoding="utf-8")
    data = {"xtr": pp.transform(X.iloc[split["train"]]).to_numpy(np.float32),
            "xva": pp.transform(X.iloc[split["val"]]).to_numpy(np.float32),
            "xte": pp.transform(X.iloc[split["test"]]).to_numpy(np.float32),
            "ytr": d["y"][split["train"]], "yva": d["y"][split["val"]]}
    if "kaam" in todo:
        template = M.KAAM(FEATURE_NAMES, M.UNIT_RANGES, num_knots=M.NUM_KNOTS, spline_order=M.SPLINE_ORDER)
        data["kaam_designs"] = tuple(M.kaam_design(template, data[k]) for k in ("xtr", "xva", "xte"))
    for m in todo:
        run_model(m, out_dir, data, dataset_name, protocol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["original", "full"])
    ap.add_argument("--protocols", default="random,time,grouped,louo")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--queue", action="store_true",
                    help="several workers share the splits; each claims a split with a lock file")
    args = ap.parse_args()
    models = args.models.split(",")
    d = load_dataset(args.dataset)
    log(f"dataset {args.dataset}: {len(d['y']):,} tasks")
    for protocol, split_name, make in plan(args.dataset, d, args.protocols.split(",")):
        run_split(args.dataset, d, protocol, split_name, make, models, queue=args.queue)
    log("done")


if __name__ == "__main__":
    main()
