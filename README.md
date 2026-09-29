# KANVAS

KANVAS (Kolmogorov-Arnold Additive Model for interpretable clinical risk prediction) predicts binary risk — for example, ICU mortality — as `Sigmoid(sum of per-feature contributions + bias)`, where every input feature is passed through its own independent, learned nonlinear function (a B-spline "edge", implemented in `kanvas/bspline.py` as `BSplineEdge`, following the base-activation-plus-spline-residual parameterization from the KAN paper) and no feature's function is ever allowed to see another feature's value. This strict per-feature decoupling, enforced structurally in `kanvas/model.py` by the `KAAM` model (one `BSplineEdge` per feature, stored in an `nn.ModuleDict`), is what makes every prediction decomposable into a clinician-readable list of per-feature contributions via `KAAM.explain()`, rather than an opaque black-box score.

This repository currently contains architecture and validation only, trained on synthetic toy data — no clinical data is used. To sanity-check the architecture, install dependencies with `pip install -r requirements.txt`, then open `notebooks/01_toy_demo.ipynb` and run all cells; it generates a synthetic dataset with known ground-truth per-feature functions (sinusoidal, U-shaped, and linear), trains a `KAAM` model on it, and plots the learned curves against the true functions so you can visually confirm the model recovers them. To run the unit tests (including the test that verifies features cannot interact), run `pytest tests/` from the repository root.

## Real data: Alibaba Cluster-Trace-GPU-2020

`kanvas/data.py` turns the Alibaba PAI GPU cluster trace into a failure-prediction dataset:

- The label is task `status`: 1 = `Failed`, 0 = `Terminated`.
- There are 10 per-task features, each traced to one raw column.
- Node telemetry comes from other jobs' workers that finished on the task's machine in the hour before it started. This design is leak-free, and it keeps failed tasks, which a join on the task's own worker window would lose.

To run it, place `pai_job_table.csv`, `pai_task_table.csv`, `pai_instance_table.csv`, `pai_machine_metric.csv` and `pai_machine_spec.csv` in `data/raw/`, then open `notebooks/02_alibaba_pipeline_eda.ipynb`. The notebook calls `build_kanvas_dataset("../data/raw")`, which prints a row count and drop reason at every step and writes `data/processed/kanvas_features.csv`.

`pytest tests/` includes the pipeline tests. They take about a minute because they stream the 2 GB instance table, and they skip if the raw files are missing.
