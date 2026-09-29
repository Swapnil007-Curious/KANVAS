"""Data pipeline: Alibaba Cluster-Trace-GPU-2020 -> KANVAS feature matrix.

Turns five raw PAI trace tables into a 10-column feature matrix ``X``, a
binary failure label ``y``, and a ``feature_ranges`` dict that plugs straight
into ``KAAM(feature_names, feature_ranges)``.

Join path (``pai_task_table`` has no worker or machine column, so
``pai_instance_table`` is the bridge from task requests to machines)::

    pai_task_table --(job_name, task_name)--> pai_instance_table (first instance)
        --(machine + 1h lookback before instance start)--> pai_machine_metric
        --(machine)--> pai_machine_spec

Label: ``pai_task_table.status`` -> 1 = 'Failed', 0 = 'Terminated'. Every
other status ('Running', 'Waiting') is an unresolved outcome and is dropped.

Why telemetry is joined by machine and strictly before task start, not by
worker_name over the task's own window: pai_machine_metric holds one row per
worker, and those rows exist almost only for workers that finished normally
(42% of Terminated instances have one vs 0.09% of Failed instances), while
99% of failed tasks' instances never record an end_time. A worker-keyed
window join therefore removes nearly every failure. Reading the assigned
machine's state from *other* jobs' workers that finished in the hour before
the task started works for failed and successful tasks alike, and it never
uses information from after the task began.

Every step prints how many tasks survived and why the rest were dropped, so
the path from raw trace to training matrix stays auditable end to end.
"""

import os
import warnings

import numpy as np
import pandas as pd

LABEL_MAP = {"Failed": 1, "Terminated": 0}

# The 10 model inputs, each traceable to exactly one named raw column.
FEATURE_NAMES = [
    "plan_cpu",  # pai_task_table.plan_cpu: requested CPU (100 = 1 core)
    "plan_mem",  # pai_task_table.plan_mem: requested memory (GB)
    "plan_gpu",  # pai_task_table.plan_gpu: requested GPU (100 = 1 GPU)
    "inst_num",  # pai_task_table.inst_num: instances requested
    "avg_cpu_usr",  # mean pai_machine_metric.machine_cpu_usr in the lookback window
    "avg_gpu_util",  # mean pai_machine_metric.machine_gpu in the lookback window
    "avg_cpu_kernel",  # mean pai_machine_metric.machine_cpu_kernel in the lookback window
    "avg_load_1",  # mean pai_machine_metric.machine_load_1 in the lookback window
    "cap_gpu",  # pai_machine_spec.cap_gpu: GPUs installed on the assigned machine
    "avg_net_receive",  # mean pai_machine_metric.machine_net_receive in the lookback window
]

# machine_metric column -> name of its lookback-window average.
WINDOW_METRICS = {
    "machine_cpu_usr": "avg_cpu_usr",
    "machine_gpu": "avg_gpu_util",
    "machine_cpu_kernel": "avg_cpu_kernel",
    "machine_load_1": "avg_load_1",
    "machine_net_receive": "avg_net_receive",
}

INSTANCE_COLS = ["job_name", "task_name", "worker_name", "start_time", "machine"]


def build_kanvas_dataset(raw_dir, sample_n_tasks=5000, lookback_hours=1.0, random_state=42, out_path=None,
                         chunksize=1_000_000, verbose=True):
    """Build the KANVAS feature matrix from the raw Alibaba PAI GPU-2020 trace.

    Args:
        raw_dir: Directory holding pai_task_table.csv, pai_instance_table.csv,
            pai_machine_metric.csv and pai_machine_spec.csv.
        sample_n_tasks: Number of labeled tasks to sample, stratified so the
            sample keeps the full labeled set's failure rate.
        lookback_hours: Telemetry window length. A task's machine features
            average the machine_metric rows of other jobs' workers on its
            assigned machine that ended within this many hours before the
            task's first instance started.
        random_state: Seed for the stratified sample.
        out_path: Where to write the cleaned features + label CSV. Defaults to
            ``<raw_dir>/../processed/kanvas_features.csv``.
        chunksize: Rows per chunk when streaming the two large tables
            (instance table ~2 GB, machine_metric ~0.4 GB), which are never
            loaded whole.
        verbose: Print the per-step survival report.

    Returns:
        Tuple ``(X, y, feature_ranges)``:
            X: DataFrame with exactly the columns in ``FEATURE_NAMES``,
                indexed by (job_name, task_name).
            y: int64 Series named ``label`` (1 = Failed, 0 = Terminated).
            feature_ranges: Dict mapping each feature name to its
                ``(min, max)`` in the cleaned data, ready for ``KAAM``.
    """
    log = print if verbose else (lambda *args, **kwargs: None)
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(raw_dir)), "processed", "kanvas_features.csv")
    funnel = []

    def record(stage, labels, reason):
        funnel.append({"stage": stage, "tasks": len(labels), "failure_rate": labels.mean(),
                       "dropped": funnel[-1]["tasks"] - len(labels), "reason": reason})

    # ------------------------------------------------------------------ Step 1
    task = pd.read_csv(os.path.join(raw_dir, "pai_task_table.csv"))
    n_raw = len(task)
    funnel.append({"stage": "raw pai_task_table rows", "tasks": n_raw, "failure_rate": np.nan, "dropped": np.nan,
                   "reason": ""})
    labeled = task["status"].isin(list(LABEL_MAP))
    log(f"[Step 1] pai_task_table: {n_raw:,} rows")
    for status, n in task.loc[~labeled, "status"].value_counts(dropna=False).items():
        log(f"  drop status={status!r}: {n:,} rows (unresolved outcome, not a failure/success label)")
    log(f"  dropped {(~labeled).sum():,} of {n_raw:,} rows ({(~labeled).mean():.2%}); "
        f"{labeled.sum():,} labeled rows remain")

    task = task[labeled].copy()
    task["label"] = task["status"].map(LABEL_MAP).astype("int64")
    record("labeled (Failed/Terminated)", task["label"], "status Running/Waiting")
    full_rate = task["label"].mean()
    log(f"  full labeled set: {task['label'].sum():,} Failed / {(task['label'] == 0).sum():,} Terminated "
        f"-> failure rate {full_rate:.2%}")
    if not 0.05 <= full_rate <= 0.95:
        log(f"  WARNING: heavily imbalanced target (failure rate {full_rate:.2%}); "
            "the stratified sample keeps this ratio")

    sample = _stratified_sample(task, sample_n_tasks, random_state)
    log(f"  stratified sample: {len(sample):,} tasks, failure rate {sample['label'].mean():.2%} "
        f"(full labeled set {full_rate:.2%})")
    record("stratified sample", sample["label"], "sampling, not a data-quality drop")

    # ------------------------------------------------------------------ Step 2
    sampled_jobs = sample["job_name"].unique()
    parts, n_scanned = [], 0
    for chunk in pd.read_csv(os.path.join(raw_dir, "pai_instance_table.csv"), usecols=INSTANCE_COLS,
                             dtype={"start_time": "float64"}, chunksize=chunksize):
        n_scanned += len(chunk)
        parts.append(chunk[chunk["job_name"].isin(sampled_jobs)])
    job_inst = pd.concat(parts, ignore_index=True)
    job_inst["file_order"] = np.arange(len(job_inst))
    job_inst = job_inst.rename(columns={"start_time": "inst_start_time"})
    # Every worker of a sampled job, including sibling tasks, so Step 3 can
    # keep a job's own workers out of its telemetry.
    job_workers = job_inst.dropna(subset=["worker_name"]).groupby("job_name")["worker_name"].agg(frozenset).to_dict()

    inst = job_inst.merge(sample[["job_name", "task_name"]], on=["job_name", "task_name"], how="inner")
    # "First" instance = earliest start_time (never-started instances last),
    # ties broken by file order.
    inst = inst.sort_values(["job_name", "task_name", "inst_start_time", "file_order"], na_position="last",
                            kind="mergesort")
    per_task = inst.groupby(["job_name", "task_name"]).size()
    first = inst.drop_duplicates(["job_name", "task_name"], keep="first").drop(columns="file_order")

    df = sample.merge(first, on=["job_name", "task_name"], how="inner")
    log(f"\n[Step 2] pai_instance_table streamed: {n_scanned:,} rows scanned in chunks of {chunksize:,}")
    log(f"  {len(inst):,} instance rows belong to the {len(sample):,} sampled tasks; "
        f"{(per_task > 1).sum():,} tasks have >1 instance (kept only the earliest-starting one)")
    log(f"  dropped {len(sample) - len(df):,} sampled tasks with zero matching instances; {len(df):,} remain "
        f"(failure rate {df['label'].mean():.2%})")
    record("joined to first instance", df["label"], "no matching instance")

    # ------------------------------------------------------------------ Step 3
    machines = df["machine"].dropna().unique()
    metric_cols = list(WINDOW_METRICS)
    parts, n_scanned = [], 0
    for chunk in pd.read_csv(os.path.join(raw_dir, "pai_machine_metric.csv"),
                             usecols=["worker_name", "machine", "end_time"] + metric_cols, chunksize=chunksize):
        n_scanned += len(chunk)
        parts.append(chunk[chunk["machine"].isin(machines)])
    mm = pd.concat(parts, ignore_index=True)

    means, n_rows, n_own_job, reasons = _lookback_average(df, mm, job_workers, lookback_hours * 3600, metric_cols)
    for j, col in enumerate(metric_cols):
        df[WINDOW_METRICS[col]] = means[:, j]

    matched = n_rows > 0
    log(f"\n[Step 3] pai_machine_metric streamed: {n_scanned:,} rows scanned; {len(mm):,} rows are on the "
        f"{len(machines):,} assigned machines")
    log(f"  telemetry window: other jobs' workers on the assigned machine that ended in the "
        f"{lookback_hours:g}h before the task's first instance started")
    log(f"  leakage guard: excluded {n_own_job.sum():,} in-window rows that belong to the task's own job")
    for reason, n in pd.Series(reasons[~matched]).value_counts().items():
        log(f"  drop {n:,} tasks: {reason}")
    log(f"  dropped {(~matched).sum():,} tasks with zero machine_metric rows in the window; "
        f"{matched.sum():,} remain (failure rate {df.loc[matched, 'label'].mean():.2%})")
    rows_used = pd.Series(n_rows[matched])
    log(f"  machine_metric rows averaged per task: median {rows_used.median():.0f}, "
        f"90th pct {rows_used.quantile(0.9):.0f}, max {rows_used.max()}")
    df = df[matched].copy()
    record(f"telemetry in {lookback_hours:g}h lookback", df["label"], "no machine_metric row in lookback window")

    # ------------------------------------------------------------------ Step 4
    spec = pd.read_csv(os.path.join(raw_dir, "pai_machine_spec.csv"), usecols=["machine", "cap_cpu", "cap_mem", "cap_gpu"])
    df = df.merge(spec, on="machine", how="left", validate="many_to_one", indicator=True)
    in_spec = df["_merge"] == "both"
    log(f"\n[Step 4] pai_machine_spec: {len(spec):,} machines")
    log(f"  dropped {(~in_spec).sum():,} tasks whose machine is missing from pai_machine_spec; "
        f"{in_spec.sum():,} remain (failure rate {df.loc[in_spec, 'label'].mean():.2%})")
    df = df[in_spec].drop(columns="_merge")
    record("machine found in machine_spec", df["label"], "machine missing from machine_spec")

    # ------------------------------------------------------------------ Step 5
    features = df[FEATURE_NAMES].astype("float64")
    features.index = pd.MultiIndex.from_frame(df[["job_name", "task_name"]])
    label = pd.Series(df["label"].to_numpy(), index=features.index, name="label")
    log(f"\n[Step 5] engineered {features.shape[1]} features for {len(features):,} tasks: {', '.join(FEATURE_NAMES)}")

    # ------------------------------------------------------------------ Step 6
    nan_mask = features.isna()
    has_nan = nan_mask.any(axis=1) | label.isna()
    only_nan = nan_mask[nan_mask.sum(axis=1) == 1].sum()
    log(f"\n[Step 6] NaN audit over the 10 features + label ({len(features):,} tasks):")
    for col in FEATURE_NAMES:
        log(f"  {col:<16s} NaN in {nan_mask[col].sum():>6,} tasks  (sole NaN feature in {only_nan[col]:>5,})")
    kept_rate = label[~has_nan].mean()
    dropped_rate = label[has_nan].mean() if has_nan.any() else float("nan")
    log(f"  dropped {has_nan.sum():,} tasks with any NaN (failure rate among dropped {dropped_rate:.2%}); "
        f"{(~has_nan).sum():,} remain (failure rate {kept_rate:.2%})")
    X = features[~has_nan].copy()
    y = label[~has_nan].astype("int64")
    record("no NaN in the 10 features", y, "NaN in at least one feature")

    log("  clipping to 1st/99th percentile:")
    for col in FEATURE_NAMES:
        lo, hi = X[col].quantile([0.01, 0.99])
        n_lo, n_hi = int((X[col] < lo).sum()), int((X[col] > hi).sum())
        X[col] = X[col].clip(lo, hi)
        log(f"  {col:<16s} [{lo:,.4g}, {hi:,.4g}]  clipped {n_lo:>3,} low + {n_hi:>3,} high "
            f"= {(n_lo + n_hi) / len(X):.2%} of values")

    feature_ranges = {col: (float(X[col].min()), float(X[col].max())) for col in FEATURE_NAMES}
    constant = [col for col, (lo, hi) in feature_ranges.items() if hi <= lo]
    if constant:
        log(f"  WARNING: constant features {constant}: KAAM cannot place a spline grid on a zero-width range")
    log(f"  final: {len(X):,} tasks, {int(y.sum()):,} Failed, failure rate {y.mean():.2%}")

    # ------------------------------------------------------------------ Step 7
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X.assign(label=y).to_csv(out_path)
    log(f"\n[Step 7] saved {len(X):,} rows x ({X.shape[1]} features + label) -> {out_path}")

    log("\nSurvival funnel (raw pai_task_table -> final feature matrix):")
    log(pd.DataFrame(funnel).to_string(index=False, na_rep="-", formatters={
        "tasks": "{:,}".format,
        "failure_rate": "{:.2%}".format,
        "dropped": "{:,.0f}".format,
    }))

    return X, y, feature_ranges


def scale_features(X, feature_ranges):
    """Min-max scale each feature to [0, 1] using only its own (min, max).

    KAAM's base term ``base_weight * silu(x)`` grows linearly with the raw
    input, so raw trace units (network receive rates around 1e8) saturate
    the sigmoid. Each column is shifted and rescaled by constants of its own,
    so every feature still reaches the model independently and each learned
    curve maps back to raw units with the inverse transform.

    Args:
        X: DataFrame whose columns are keys of ``feature_ranges``.
        feature_ranges: Dict of feature name -> (min, max) in raw units.

    Returns:
        DataFrame of the same shape with every column scaled to [0, 1]
        (values outside the range land outside [0, 1]).
    """
    lo = pd.Series({name: feature_ranges[name][0] for name in X.columns})
    hi = pd.Series({name: feature_ranges[name][1] for name in X.columns})
    return (X - lo) / (hi - lo)


def _stratified_sample(df, n, random_state):
    """Sample ``n`` rows so the label ratio matches ``df``'s own ratio."""
    if n >= len(df):
        return df.copy()
    n_pos = int(round(n * df["label"].mean()))
    pos = df[df["label"] == 1].sample(n=n_pos, random_state=random_state)
    neg = df[df["label"] == 0].sample(n=n - n_pos, random_state=random_state)
    return pd.concat([pos, neg]).sample(frac=1, random_state=random_state)


def _lookback_average(tasks, mm, job_workers, lookback_seconds, metric_cols):
    """Average the machine_metric rows describing each task's machine just before it started.

    For each task, the rows used are those on its assigned machine whose
    worker ended in ``[t0 - lookback_seconds, t0)``, where ``t0`` is the
    start time of the task's first instance, excluding every worker of the
    task's own job. Each row used therefore describes the machine strictly
    before the task started, and never the task or its sibling tasks.

    machine_metric is sorted by (machine, end_time) once, so each machine's
    rows form one contiguous, time-ordered block; each task then needs only
    two binary searches (``searchsorted``) inside its own machine's block,
    never a scan of the metric table.

    Args:
        tasks: DataFrame with job_name, machine, inst_start_time columns.
        mm: machine_metric rows with worker_name, machine, end_time and
            ``metric_cols``.
        job_workers: Dict of job_name -> set of that job's worker_names.
        lookback_seconds: Window length before ``t0``.
        metric_cols: machine_metric columns to average.

    Returns:
        Tuple ``(means, n_rows, n_own_job, reasons)``: window averages of
        shape (len(tasks), len(metric_cols)) (NaN where nothing matched),
        the number of rows averaged per task, the number of in-window rows
        excluded for belonging to the task's own job, and the reason each
        unmatched task found no rows ("" when matched).
    """
    mm = mm.sort_values(["machine", "end_time"], kind="mergesort").reset_index(drop=True)
    ends = mm["end_time"].to_numpy(dtype="float64")
    workers = mm["worker_name"].to_numpy(dtype=object)
    values = mm[metric_cols].to_numpy(dtype="float64")
    blocks = mm.groupby("machine", sort=False).indices

    means = np.full((len(tasks), len(metric_cols)), np.nan)
    n_rows = np.zeros(len(tasks), dtype="int64")
    n_own_job = np.zeros(len(tasks), dtype="int64")
    reasons = np.full(len(tasks), "", dtype=object)
    for i, (job, machine, t0) in enumerate(zip(tasks["job_name"], tasks["machine"], tasks["inst_start_time"])):
        if pd.isna(machine):
            reasons[i] = "first instance has no machine"
            continue
        if pd.isna(t0):
            reasons[i] = "first instance never started (NaN start_time)"
            continue
        block = blocks.get(machine)
        if block is None:
            reasons[i] = "assigned machine has no machine_metric rows"
            continue
        lo, hi = block[0], block[-1] + 1
        a = lo + np.searchsorted(ends[lo:hi], t0 - lookback_seconds, side="left")
        b = lo + np.searchsorted(ends[lo:hi], t0, side="left")
        own = job_workers.get(job, frozenset())
        rows = [r for r in range(a, b) if workers[r] not in own]
        n_own_job[i] = (b - a) - len(rows)
        if not rows:
            reasons[i] = "no other job's worker ended on the machine in the lookback window"
            continue
        n_rows[i] = len(rows)
        with warnings.catch_warnings():
            # An all-NaN column gives a NaN mean, which Step 6 drops and reports.
            warnings.simplefilter("ignore", RuntimeWarning)
            means[i] = np.nanmean(values[rows], axis=0)
    return means, n_rows, n_own_job, reasons
