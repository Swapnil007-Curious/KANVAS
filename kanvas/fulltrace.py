"""Full-trace build of the KANVAS feature matrix, memory-safe on a 20 GB laptop.

``data.build_kanvas_dataset`` (v1) streamed the 2 GB instance table once per
build and ran a per-task Python loop, which was fine for a 40,000-task sample.
This module applies the same frozen rules to every labelled task in the trace
(1,141,835 tasks) without loading any raw table whole more than once:

1. ``cache_instance_table`` streams ``pai_instance_table.csv`` in chunks and
   keeps the five columns the pipeline reads, plus the row's position in the
   file, in a compressed parquet file (about 0.3 GB). Every later build reads
   that file instead of the CSV.
2. ``build_full_dataset`` joins tasks to their first instance, attaches the
   leak-free lookback telemetry with vectorised searches (no Python loop over
   tasks), joins the machine specification and the submitting user, and drops
   tasks with a missing feature. It never clips or scales: clipping and
   min-max scaling are fitted on each training split by ``preprocess.py``.

The rules are exactly those of v1 (``data.py``):

- label: ``status`` Failed -> 1, Terminated -> 0; Running and Waiting dropped;
- first instance: earliest ``start_time`` (never-started instances last), ties
  broken by file order;
- telemetry: ``pai_machine_metric`` rows on the first instance's machine whose
  ``end_time`` lies in ``[t0 - lookback, t0)``, excluding every worker of the
  task's own job, averaged per metric over non-missing values;
- ``cap_gpu`` from ``pai_machine_spec``;
- a task with any missing feature is dropped.

``tests/test_fulltrace.py`` checks the vectorised window against the v1 loop,
and ``scripts/build_full_trace.py`` checks that the build reproduces the v1
18,890-task matrix row for row.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import FEATURE_NAMES, LABEL_MAP, WINDOW_METRICS

INSTANCE_COLS = ["job_name", "task_name", "worker_name", "start_time", "machine"]
STRING_COLS = ["job_name", "task_name", "worker_name", "machine"]
# end_time offsets fit in [0, 1e8) seconds (the trace spans about 6e6 s), so machine_code * 1e8 + end_time
# is an exact float64 key that orders rows by machine, then end time.
_KEY_STRIDE = 1e8


def cache_instance_table(raw_dir, out_path, chunksize=1_000_000, verbose=True):
    """Stream ``pai_instance_table.csv`` once and keep the pipeline's columns as parquet.

    Adds ``file_order`` (the row's 0-based position in the CSV) so ties in the
    first-instance rule are broken exactly as v1 broke them.

    Returns:
        Number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = os.path.join(raw_dir, "pai_instance_table.csv")
    schema = pa.schema([("job_name", pa.string()), ("task_name", pa.string()), ("worker_name", pa.string()),
                        ("start_time", pa.float64()), ("machine", pa.string()), ("file_order", pa.int64())])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".partial"
    n = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for chunk in pd.read_csv(src, usecols=INSTANCE_COLS, chunksize=chunksize,
                                 dtype={**{c: "object" for c in STRING_COLS}, "start_time": "float64"}):
            chunk = chunk[INSTANCE_COLS].copy()
            chunk["file_order"] = np.arange(n, n + len(chunk), dtype="int64")
            n += len(chunk)
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
            if verbose:
                print(f"  instance table: {n:,} rows cached", flush=True)
    os.replace(tmp, out_path)
    return n


def lookback_window_means(task_machine, task_t0, task_job, mm_machine, mm_end, mm_job, mm_values,
                          lookback_seconds):
    """Vectorised lookback averages: the same quantity as ``data._lookback_average``.

    All identifiers arrive as integer codes (-1 for missing). For task ``i`` the
    rows used are those with ``mm_machine == task_machine[i]``,
    ``task_t0[i] - lookback_seconds <= mm_end < task_t0[i]`` and
    ``mm_job != task_job[i]``; each metric is averaged over its non-missing
    values among those rows.

    Returns:
        ``(means, n_rows, n_own_job, reason)``: means of shape
        (n_tasks, n_metrics) (NaN where nothing matched), the number of rows
        averaged, the number of in-window rows dropped for belonging to the
        task's own job, and a reason code per task (0 matched, 1 machine
        missing, 2 start time missing, 3 machine has no metric rows, 4 no
        other job's row in the window).
    """
    task_machine = np.asarray(task_machine, dtype="int64")
    task_t0 = np.asarray(task_t0, dtype="float64")
    task_job = np.asarray(task_job, dtype="int64")
    mm_machine = np.asarray(mm_machine, dtype="int64")
    mm_end = np.asarray(mm_end, dtype="float64")
    mm_job = np.asarray(mm_job, dtype="int64")
    mm_values = np.asarray(mm_values, dtype="float64")
    n_tasks, n_metrics = len(task_machine), mm_values.shape[1]

    # Rows without an end time can never fall in a window (v1 sorted them last and its searches never
    # reached them); rows without a machine match no task.
    keep = np.isfinite(mm_end) & (mm_machine >= 0)
    if np.any(mm_end[keep] < 0) or np.any(mm_end[keep] >= _KEY_STRIDE):
        raise ValueError("machine_metric end_time outside [0, 1e8): the composite sort key would not be exact")
    key = mm_machine[keep] * _KEY_STRIDE + mm_end[keep]
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    job_sorted = mm_job[keep][order]
    values_sorted = mm_values[keep][order]
    machines_with_rows = np.zeros(max(int(mm_machine.max(initial=-1)), int(task_machine.max(initial=-1))) + 2,
                                  dtype=bool)
    machines_with_rows[mm_machine[keep]] = True

    reason = np.zeros(n_tasks, dtype="int8")
    reason[task_machine < 0] = 1
    reason[(reason == 0) & ~np.isfinite(task_t0)] = 2
    ok = reason == 0
    reason[ok & ~machines_with_rows[np.where(task_machine >= 0, task_machine, 0)]] = 3
    ok = reason == 0

    a = np.zeros(n_tasks, dtype="int64")
    b = np.zeros(n_tasks, dtype="int64")
    base = task_machine[ok] * _KEY_STRIDE
    a[ok] = np.searchsorted(key, base + task_t0[ok] - lookback_seconds, side="left")
    b[ok] = np.searchsorted(key, base + task_t0[ok], side="left")
    lengths = b - a

    total = int(lengths.sum())
    owner = np.repeat(np.arange(n_tasks), lengths)
    starts = np.repeat(a - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)
    rows = starts + np.arange(total)
    own = job_sorted[rows] == task_job[owner]
    n_own_job = np.bincount(owner[own], minlength=n_tasks)
    owner, rows = owner[~own], rows[~own]
    n_rows = np.bincount(owner, minlength=n_tasks)
    reason[ok & (n_rows == 0)] = 4

    means = np.full((n_tasks, n_metrics), np.nan)
    for m in range(n_metrics):
        v = values_sorted[rows, m]
        finite = np.isfinite(v)
        s = np.bincount(owner[finite], weights=v[finite], minlength=n_tasks)
        c = np.bincount(owner[finite], minlength=n_tasks)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[:, m] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    means[n_rows == 0] = np.nan
    return means, n_rows, n_own_job, reason


REASONS = {1: "first instance has no machine", 2: "first instance never started (NaN start_time)",
           3: "assigned machine has no machine_metric rows",
           4: "no other job's worker ended on the machine in the lookback window"}


def build_full_dataset(raw_dir, instance_parquet, lookback_hours=(1.0,), tasks=None, verbose=True):
    """Apply the frozen v1 rules to every labelled task (or to ``tasks``).

    Args:
        raw_dir: Directory with pai_task_table, pai_machine_metric, pai_machine_spec and pai_job_table.
        instance_parquet: Output of ``cache_instance_table``.
        lookback_hours: One or more window lengths; the join is shared and one
            frame is returned per window.
        tasks: Optional DataFrame with ``job_name`` and ``task_name``; when
            given, only those labelled tasks are built (used to rebuild a v1 sample).
        verbose: Print the attrition report.

    Returns:
        Dict ``{lookback_hours: (frame, funnel)}``. ``frame`` is indexed by
        (job_name, task_name) and holds the ten features (unclipped, raw
        units), ``label``, ``start_time`` (the first instance's start, trace
        seconds), ``user`` (opaque hash, a split key only), ``machine`` and
        ``n_metric_rows``. ``funnel`` is the attrition table.
    """
    import pyarrow.parquet as pq

    log = print if verbose else (lambda *a, **k: None)
    if np.isscalar(lookback_hours):
        lookback_hours = (float(lookback_hours),)

    task = pd.read_csv(os.path.join(raw_dir, "pai_task_table.csv"),
                       usecols=["job_name", "task_name", "status", "plan_cpu", "plan_mem", "plan_gpu", "inst_num"])
    n_raw = len(task)
    labelled = task["status"].isin(list(LABEL_MAP))
    task = task[labelled].copy()
    task["label"] = task["status"].map(LABEL_MAP).astype("int64")
    task = task.drop(columns="status")
    n_labelled, rate_labelled = len(task), task["label"].mean()
    if tasks is not None:
        task = task.merge(tasks[["job_name", "task_name"]].drop_duplicates(), on=["job_name", "task_name"],
                          how="inner")
    log(f"[1] pai_task_table: {n_raw:,} rows; {n_labelled:,} labelled (failure rate {rate_labelled:.2%}); "
        f"building {len(task):,} tasks")
    funnel = [{"stage": "raw pai_task_table rows", "tasks": n_raw, "failure_rate": np.nan, "reason": ""},
              {"stage": "labelled (Failed or Terminated)", "tasks": n_labelled, "failure_rate": rate_labelled,
               "reason": "status Running or Waiting"}]
    if tasks is not None:
        funnel.append({"stage": "requested subset", "tasks": len(task), "failure_rate": task["label"].mean(),
                       "reason": "sampling"})

    # ------------------------------------------------------------- first instance
    inst = pq.read_table(instance_parquet).to_pandas()
    inst = inst.merge(task[["job_name", "task_name"]], on=["job_name", "task_name"], how="inner")
    first = (inst.sort_values(["start_time", "file_order"], na_position="last", kind="mergesort")
             .drop_duplicates(["job_name", "task_name"], keep="first"))
    first = first[["job_name", "task_name", "start_time", "machine"]]
    df = task.merge(first, on=["job_name", "task_name"], how="inner")
    log(f"[2] first instance found for {len(df):,} of {len(task):,} tasks")
    funnel.append({"stage": "first instance found", "tasks": len(df), "failure_rate": df["label"].mean(),
                   "reason": "no instance row"})
    del inst

    # ------------------------------------------------------------- worker -> job, for the own-job guard
    metric_cols = list(WINDOW_METRICS)
    mm = pd.read_csv(os.path.join(raw_dir, "pai_machine_metric.csv"),
                     usecols=["worker_name", "machine", "end_time"] + metric_cols,
                     dtype={"worker_name": "object", "machine": "object"})
    workers_in_mm = pd.Index(mm["worker_name"].dropna().unique())
    wj = pq.read_table(instance_parquet, columns=["worker_name", "job_name"]).to_pandas()
    wj = wj.dropna(subset=["worker_name"])
    wj = wj[wj["worker_name"].isin(workers_in_mm)].drop_duplicates()
    if wj["worker_name"].duplicated().any():
        raise ValueError("a worker_name belongs to more than one job: the own-job guard needs the set form")
    worker_job = wj.set_index("worker_name")["job_name"]
    log(f"[3] machine_metric: {len(mm):,} rows; {len(worker_job):,} of {len(workers_in_mm):,} metric workers "
        "map to a job (each to exactly one)")

    job_codes = pd.Index(pd.concat([df["job_name"], worker_job]).dropna().unique())
    machine_codes = pd.Index(pd.concat([df["machine"], mm["machine"]]).dropna().unique())

    def codes(index, values):
        c = index.get_indexer(values)
        return np.where(pd.isna(values), -1, c)

    task_job = codes(job_codes, df["job_name"].to_numpy())
    task_machine = codes(machine_codes, df["machine"].to_numpy())
    mm_job = codes(job_codes, worker_job.reindex(mm["worker_name"]).to_numpy())
    mm_machine = codes(machine_codes, mm["machine"].to_numpy())

    spec = pd.read_csv(os.path.join(raw_dir, "pai_machine_spec.csv"), usecols=["machine", "cap_gpu"])
    jobs = pd.read_csv(os.path.join(raw_dir, "pai_job_table.csv"), usecols=["job_name", "user"])
    user_of_job = jobs.drop_duplicates("job_name").set_index("job_name")["user"]

    out = {}
    for hours in lookback_hours:
        means, n_rows, n_own, reason = lookback_window_means(
            task_machine, df["start_time"].to_numpy(), task_job, mm_machine, mm["end_time"].to_numpy(),
            mm_job, mm[metric_cols].to_numpy(), hours * 3600.0)
        d = df.copy()
        for j, col in enumerate(metric_cols):
            d[WINDOW_METRICS[col]] = means[:, j]
        d["n_metric_rows"] = n_rows
        fun = list(funnel)
        log(f"[4] lookback {hours:g} h: excluded {int(n_own.sum()):,} in-window rows of the task's own job")
        for code, text in REASONS.items():
            log(f"    drop {int((reason == code).sum()):,} tasks: {text}")
        d = d[reason == 0].copy()
        fun.append({"stage": f"telemetry in {hours:g} h lookback", "tasks": len(d),
                    "failure_rate": d["label"].mean(), "reason": "no metric row in the window",
                    **{f"dropped: {text}": int((reason == code).sum()) for code, text in REASONS.items()}})

        d = d.merge(spec, on="machine", how="left", validate="many_to_one", indicator=True)
        in_spec = d["_merge"] == "both"
        d = d[in_spec].drop(columns="_merge")
        fun.append({"stage": "machine in machine_spec", "tasks": len(d), "failure_rate": d["label"].mean(),
                    "reason": "machine missing from machine_spec"})

        nan = d[FEATURE_NAMES].isna()
        any_nan = nan.any(axis=1)
        fun.append({"stage": "no missing feature", "tasks": int((~any_nan).sum()),
                    "failure_rate": d.loc[~any_nan, "label"].mean(), "reason": "a feature is missing",
                    "dropped: CPU-only (plan_gpu missing)": int(nan["plan_gpu"].sum()),
                    **{f"missing: {c}": int(nan[c].sum()) for c in FEATURE_NAMES}})
        d = d[~any_nan].copy()
        d["user"] = user_of_job.reindex(d["job_name"]).to_numpy()
        d = d.set_index(["job_name", "task_name"])
        cols = FEATURE_NAMES + ["label", "start_time", "user", "machine", "n_metric_rows"]
        d = d[cols]
        d[FEATURE_NAMES] = d[FEATURE_NAMES].astype("float64")
        log(f"    final: {len(d):,} tasks, failure rate {d['label'].mean():.2%}, "
            f"{d['user'].nunique():,} users ({int(d['user'].isna().sum()):,} tasks without a user)")
        out[float(hours)] = (d, pd.DataFrame(fun))
    return out


def v1_stratified_sample(raw_dir, n=40_000, random_state=42):
    """Rebuild the v1 stratified task sample (``data._stratified_sample`` on the labelled task table)."""
    from .data import _stratified_sample

    task = pd.read_csv(os.path.join(raw_dir, "pai_task_table.csv"))
    task = task[task["status"].isin(list(LABEL_MAP))].copy()
    task["label"] = task["status"].map(LABEL_MAP).astype("int64")
    return _stratified_sample(task, n, random_state)[["job_name", "task_name", "label"]]
