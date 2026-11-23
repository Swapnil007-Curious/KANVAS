"""Facts for the response to the independent review that read no test label (light on memory).

    python scripts/review_facts.py

Uses only features, users, start times, and training and validation labels. Writes results/posthoc/review_facts.json:
  time      User 1's share of each period of the time-ordered split, the validation failure rate without User 1, the
            gap between the training and test periods, and the number of users behind the new-user test tasks
  onecore   for the canonical, leave-user-out and user-grouped training splits of the full trace: the failure rate of
            one-core training tasks against other training tasks, with and without User 1 in training
  dups      per split, the share of test tasks whose ten raw features exactly match a training task
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

from kanvas.data import FEATURE_NAMES  # noqa: E402
from run_experiments import RUNS, dominant_users, load_dataset  # noqa: E402

OUT = ROOT / "results" / "posthoc"


def split_of(run_dir):
    z = np.load(run_dir / "split.npz")
    return {k: z[k] for k in ("train", "val", "test")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u1, _ = dominant_users()
    facts = {"time": {}, "onecore": {}, "dups": {}}
    for ds in ["full", "original"]:
        d = load_dataset(ds)
        y, users, start, ucode = d["y"], d["user_raw"], d["start"], d["user"]
        sp = split_of(RUNS / ds / "time" / "t")
        t = {}
        for part in ("train", "val", "test"):
            idx = sp[part]
            t[part] = {"user1_share": float(np.mean(users[idx] == u1)),
                       "start_min_day": float(start[idx].min() / 86400), "start_max_day": float(start[idx].max() / 86400)}
        val = sp["val"]
        t["val"]["rate_without_user1"] = float(y[val][users[val] != u1].mean())  # validation labels only
        t["gap_train_to_test_days"] = t["test"]["start_min_day"] - t["train"]["start_max_day"]
        known = np.isin(ucode[sp["test"]], np.unique(ucode[sp["train"]]))
        t["new_users"] = int(len(np.unique(ucode[sp["test"]][~known])))
        facts["time"][ds] = t

        h = pd.util.hash_pandas_object(d["X"][FEATURE_NAMES], index=False).to_numpy()
        dups = {}
        for proto in ["random", "time", "grouped", "louo"]:
            for rd in sorted((RUNS / ds / proto).iterdir()):
                s_ = split_of(rd)
                dups[f"{proto}/{rd.name}"] = float(np.isin(h[s_["test"]], h[s_["train"]]).mean())
        facts["dups"][ds] = dups

        if ds == "full":
            plan = d["X"]["plan_cpu"].to_numpy()
            oc = {}
            for run in ["random/s42", "louo/d1", "louo/d14"] + [f"grouped/f{k}" for k in range(5)]:
                tr = split_of(RUNS / ds / run)["train"]
                one = plan[tr] == 100
                oc[run] = {"train_has_user1": bool(np.isin(u1, users[tr])),
                           "onecore_rate": float(y[tr][one].mean()), "other_rate": float(y[tr][~one].mean()),
                           "onecore_share_user1": float(np.mean(users[tr][one] == u1))}
            facts["onecore"] = oc
        del d, h
    (OUT / "review_facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
