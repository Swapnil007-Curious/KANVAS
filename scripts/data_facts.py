"""Descriptive facts about the raw trace and the v2 datasets, quoted in the data section.

    python scripts/data_facts.py

Writes results/data/data_facts.json: raw table row counts, the trace span, the telemetry coverage
of Terminated and Failed instances (why the worker-keyed join leaks), the user concentration of
both datasets, and the machine the build ran on. Reads no model output and no split.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "data" / "data_facts.json"


def user_stats(y, user):
    t = pd.DataFrame({"u": user, "y": y}).groupby("u")["y"].agg(tasks="size", failures="sum")
    t = t.sort_values(["tasks", "failures"], ascending=False)
    return t


def main():
    facts = {}
    rows = {}
    for name in ["pai_task_table", "pai_instance_table", "pai_machine_metric", "pai_machine_spec", "pai_job_table"]:
        path = RAW / f"{name}.csv"
        rows[name] = {"bytes": os.path.getsize(path)}
    task = pd.read_csv(RAW / "pai_task_table.csv", usecols=["status", "start_time", "end_time"])
    rows["pai_task_table"]["rows"] = len(task)
    facts["status_counts"] = task["status"].value_counts().to_dict()
    span = (np.nanmax(np.r_[task["start_time"], task["end_time"]]) - np.nanmin(task["start_time"])) / 86400
    facts["trace_span_days"] = float(span)

    mm = pd.read_csv(RAW / "pai_machine_metric.csv", usecols=["worker_name"])
    rows["pai_machine_metric"]["rows"] = len(mm)
    metric_workers = set(mm["worker_name"].dropna().unique())
    cov = {"Terminated": [0, 0], "Failed": [0, 0]}
    failed_no_end = [0, 0]
    n_inst = 0
    for chunk in pd.read_csv(RAW / "pai_instance_table.csv", usecols=["worker_name", "status", "end_time"],
                             chunksize=1_000_000):
        n_inst += len(chunk)
        for st in cov:
            sel = chunk[chunk["status"] == st]
            cov[st][0] += int(sel["worker_name"].isin(metric_workers).sum())
            cov[st][1] += len(sel)
        f = chunk[chunk["status"] == "Failed"]
        failed_no_end[0] += int(f["end_time"].isna().sum())
        failed_no_end[1] += len(f)
    rows["pai_instance_table"]["rows"] = n_inst
    rows["pai_machine_spec"]["rows"] = len(pd.read_csv(RAW / "pai_machine_spec.csv", usecols=["machine"]))
    rows["pai_job_table"]["rows"] = len(pd.read_csv(RAW / "pai_job_table.csv", usecols=["job_name"]))
    facts["tables"] = rows
    facts["instance_metric_coverage"] = {st: {"with_metric_row": a, "instances": b, "share": a / b}
                                         for st, (a, b) in cov.items()}
    facts["failed_instances_without_end_time"] = {"count": failed_no_end[0], "instances": failed_no_end[1],
                                                  "share": failed_no_end[0] / failed_no_end[1]}

    df = pd.read_parquet(ROOT / "data" / "processed" / "v2" / "full_1h.parquet", columns=["label", "user", "start_time"])
    v1 = pd.read_csv(ROOT / "data" / "processed" / "kanvas_features_large.csv", index_col=["job_name", "task_name"],
                     usecols=["job_name", "task_name"]).index
    orig = df.loc[v1]
    st_orig = user_stats(orig["label"].to_numpy(), orig["user"].to_numpy())
    u1, u4 = st_orig.index[0], st_orig.index[3]
    for name, d in [("full", df), ("original", orig)]:
        st = user_stats(d["label"].to_numpy(), d["user"].to_numpy())
        nf = st["failures"].sum()
        top10 = st["failures"].sort_values(ascending=False).iloc[:10].sum()
        facts[f"users_{name}"] = {
            "users": int(len(st)), "tasks": int(len(d)), "failures": int(nf),
            "u1_rank_by_tasks": int(list(st.index).index(u1) + 1), "u4_rank_by_tasks": int(list(st.index).index(u4) + 1),
            "u1_tasks": int(st.loc[u1, "tasks"]), "u1_failures": int(st.loc[u1, "failures"]),
            "u1_task_share": float(st.loc[u1, "tasks"] / len(d)), "u1_failure_share": float(st.loc[u1, "failures"] / nf),
            "u1_rate": float(st.loc[u1, "failures"] / st.loc[u1, "tasks"]),
            "u4_tasks": int(st.loc[u4, "tasks"]), "u4_failures": int(st.loc[u4, "failures"]),
            "u4_task_share": float(st.loc[u4, "tasks"] / len(d)), "u4_failure_share": float(st.loc[u4, "failures"] / nf),
            "u4_rate": float(st.loc[u4, "failures"] / st.loc[u4, "tasks"]),
            "u14_failure_share": float((st.loc[u1, "failures"] + st.loc[u4, "failures"]) / nf),
            "top10_failure_share": float(top10 / nf),
            "u1_is_top_by_failures": bool(st["failures"].idxmax() == u1),
            "start_span_days": float((d["start_time"].max() - d["start_time"].min()) / 86400)}
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2**30
    except ImportError:
        ram = None
    facts["machine"] = {"processor": platform.processor(), "logical_cpus": os.cpu_count(), "ram_gb": ram,
                        "python": platform.python_version()}
    OUT.write_text(json.dumps(facts, indent=1, default=float), encoding="utf-8")
    print(json.dumps(facts, indent=1, default=float))


if __name__ == "__main__":
    main()
