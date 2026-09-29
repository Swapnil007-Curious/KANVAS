# KANVAS

**Kolmogorov-Arnold Network Verification and Analysis Suite.** An interpretable additive model (KAAM) that predicts GPU cluster task failure from scheduling and machine-telemetry features, benchmarked against logistic regression and a parameter-matched MLP on the Alibaba Cluster Trace GPU 2020 dataset.

Author: Swapnil Shukla, Sri Ma Vidyalaya, India.

## What it is

KAAM gives each of 10 input features its own learnable B-spline edge and sums the results:

    risk = sigmoid( sum_i phi_i(x_i) + bias ),   phi_i(x) = w_i * SiLU(x) + sum_j c_ij * B_j(x)

Features never interact. A prediction is fully decomposed into 10 per-feature contributions, and each phi_i can be plotted as a curve. The model has 151 parameters (140 spline coefficients, 10 SiLU weights, 1 bias). Feature decoupling is enforced structurally with a PyTorch `ModuleDict` and checked by unit test.

## Results (18,890 tasks, 1-hour leak-free telemetry lookback)

| Model | Parameters | Test AUC |
|---|---|---|
| Logistic regression | 11 | 0.7346 |
| KAAM | 151 | 0.7520 |
| MLP (8,6) | 149 | 0.7558 |

- KAAM vs MLP: -0.0038 AUC, paired bootstrap 95% CI includes 0. The interpretability cost is not measurable at this sample size.
- KAAM vs logistic regression: +0.0174 AUC, CI excludes 0.
- KAAM per-feature importance agrees with exact SHAP on the MLP: Spearman rho = 0.745 (n = 2,834).
- Initialization variance (std 0.0001) is far smaller than split variance (std 0.0104).

## Limitations (read this)

**The result does not generalize across users.** One user accounts for 31% of tasks and 54% of failures. When the split is grouped by user, all three models fall to chance: KAAM 0.4535 +/- 0.0418, logistic regression 0.4193, MLP 0.4025. The learned signal partly reflects who submitted the job, not just what the job requests. The random-split AUCs above should be read with that in mind. Details in `notebooks/07_user_grouped_robustness.ipynb`.

Other constraints: one trace, one cluster, binary label (Failed vs Terminated), two originally planned features replaced because of data problems (see the paper).

## Repository layout

    kanvas/        model, data pipeline, training, baselines, SHAP, robustness code
    notebooks/     01 toy demo -> 07 user-grouped robustness, run in order
    tests/         47 pytest tests (decoupling, leak checks, split integrity, export audit)
    figures/       all figures, by phase
    models/        kaam_canonical.pt (split seed 42, init seed 0, test AUC 0.7520)
    data/processed/publication/   identifier-stripped feature matrix, phi curves, benchmark table, data dictionary

## Reproduce

    pip install -r requirements.txt
    pytest

The raw trace is not included (about 1.4 GB). Download `cluster-trace-gpu-v2020` from https://github.com/alibaba/clusterdata, extract it to `ALIBABA-CLUSTER-TRACE-GPU-v2020/`, then run notebooks 02 onward. The published feature matrix in `data/processed/publication/` lets you skip the pipeline and go straight to modeling.

## Data

Derived from the Alibaba Cluster Trace GPU 2020 dataset (Weng et al., "MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters", NSDI 2022). User and job identifiers are removed from every exported file, and a test enforces it. Follow the original dataset's terms of use.

## References

- Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024. arXiv:2404.19756
- Weng et al., NSDI 2022 (dataset)

## License

MIT. See `LICENSE`.