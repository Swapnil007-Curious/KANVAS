"""The vectorised full-trace window must compute exactly what the v1 per-task loop computed."""

import numpy as np
import pandas as pd
import pytest

from kanvas.data import _lookback_average
from kanvas.fulltrace import lookback_window_means


def _random_trace(rng, n_tasks=400, n_rows=3000, n_machines=12, n_jobs=60):
    mm = pd.DataFrame({
        "worker_name": [f"w{i}" for i in range(n_rows)],
        "machine": rng.integers(0, n_machines, n_rows),
        "end_time": np.where(rng.random(n_rows) < 0.03, np.nan, rng.uniform(0, 50_000, n_rows).round()),
        "m1": np.where(rng.random(n_rows) < 0.2, np.nan, rng.normal(10, 3, n_rows)),
        "m2": rng.lognormal(18, 1, n_rows),
    })
    worker_job = rng.integers(0, n_jobs, n_rows)
    tasks = pd.DataFrame({
        "job": rng.integers(0, n_jobs, n_tasks),
        "machine": np.where(rng.random(n_tasks) < 0.05, -1, rng.integers(0, n_machines + 2, n_tasks)),
        "t0": np.where(rng.random(n_tasks) < 0.05, np.nan, rng.uniform(0, 55_000, n_tasks).round()),
    })
    return mm, worker_job, tasks


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("lookback", [1800.0, 3600.0, 21600.0])
def test_vectorised_window_equals_v1_loop(seed, lookback):
    rng = np.random.default_rng(seed)
    mm, worker_job, tasks = _random_trace(rng)
    metric_cols = ["m1", "m2"]

    # v1 form: string ids, own-job guard as a set of the job's worker names
    job_workers = {}
    for w, j in zip(mm["worker_name"], worker_job):
        job_workers.setdefault(f"j{j}", set()).add(w)
    job_workers = {k: frozenset(v) for k, v in job_workers.items()}
    v1_tasks = pd.DataFrame({"job_name": [f"j{j}" for j in tasks["job"]],
                             "machine": [np.nan if m < 0 else f"m{m}" for m in tasks["machine"]],
                             "inst_start_time": tasks["t0"]})
    v1_mm = mm.assign(machine=[f"m{m}" for m in mm["machine"]])
    means1, rows1, own1, reasons1 = _lookback_average(v1_tasks, v1_mm, job_workers, lookback, metric_cols)

    means2, rows2, own2, reason2 = lookback_window_means(
        tasks["machine"], tasks["t0"], tasks["job"], mm["machine"], mm["end_time"], worker_job,
        mm[metric_cols].to_numpy(), lookback)

    assert np.array_equal(rows1, rows2)
    assert np.array_equal(own1, own2)
    assert np.array_equal(np.isnan(means1), np.isnan(means2))
    finite = np.isfinite(means1)
    assert np.allclose(means1[finite], means2[finite], rtol=1e-12, atol=0)
    # a task is kept by both forms exactly when it has at least one row
    assert np.array_equal(reasons1 == "", reason2 == 0)
    assert (rows2 > 0).sum() > 50  # the synthetic trace exercises the matched path


def test_window_is_half_open_and_excludes_own_job():
    # machine 0: rows ending at 100, 3600 (edge, in), 5000 (in), 6000 (own job), 7200 (at start, out)
    means, n_rows, n_own, reason = lookback_window_means(
        task_machine=[0], task_t0=[7200.0], task_job=[7],
        mm_machine=[0, 0, 0, 0, 0], mm_end=[100.0, 3600.0, 5000.0, 6000.0, 7200.0],
        mm_job=[1, 2, 3, 7, 4], mm_values=np.array([[999.0], [10.0], [30.0], [999.0], [999.0]]),
        lookback_seconds=3600.0)
    assert n_rows.tolist() == [2] and n_own.tolist() == [1] and reason.tolist() == [0]
    assert means[0, 0] == pytest.approx(20.0)
