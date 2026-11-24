# Kaggle notebook series: KANVAS

Five public-ready Kaggle notebooks built from this repository, plus the Kaggle dataset they read. Nothing here has
been uploaded; every notebook is private by default (`"is_private": true`).

| # | Folder | Kaggle title | What it shows |
|---|---|---|---|
| 1 | `notebooks/kanvas/01-pai-trace-eda` | Alibaba PAI GPU trace 2020: a complete EDA | tables, GPU supply, users, task requests, arrivals, heavy-tailed durations, queueing delay, failure rates |
| 2 | `notebooks/kanvas/02-idle-gpus` | Where cluster GPUs sit idle | requested vs used GPU compute, GPU memory, hourly allocation, fragmentation on 8-GPU machines |
| 3 | `notebooks/kanvas/03-failure-lightgbm-shap` | Predicting GPU task failure: LightGBM and SHAP | LightGBM failure baseline, TreeSHAP, exact duplicates across a random split, user-grouped evaluation |
| 4 | `notebooks/kanvas/04-kaam-from-scratch` | Kolmogorov-Arnold additive models from scratch | Cox-de Boor B-splines, the 151-parameter KAAM, synthetic check, training on the trace, user-grouped evaluation |
| 5 | `notebooks/kanvas/05-splines-vs-shap` | Reading the splines: KAAM curves vs SHAP | exact Shapley check for additive models, KAAM vs LightGBM SHAP, the one-core rule of user 1 |

Each folder holds `notebook.ipynb` (executed, outputs included), `kernel-metadata.json` and `cover.png`
(1280x720, a plot from that notebook). `notebooks/kanvas/run_log.json` records the local run time of each notebook.

## Data

The trace is CC BY 4.0 (the `LICENSE` file in `alibaba/clusterdata/cluster-trace-gpu-v2020`), which allows
redistribution with attribution. A copy already exists on Kaggle (`derrickmwiti/cluster-trace-gpu-v2020`, all seven
CSVs), but it is labelled CC0 and carries no licence text or attribution, so the series uses its own dataset:

- `datasets/alibaba-gpu-trace-2020/`: `README.md` (attribution, citation, changes, columns), `LICENSE.txt` (the CC BY
  4.0 text shipped with the trace), `dataset-metadata.json` (written by `make_metadata.py`), `stage.py`.
- `stage.py` assembles the upload folder: the seven CSVs (each verified against the official sha256 after its header
  line), `SHA256SUMS.txt` and the derived table `kanvas_tasks_1h.parquet` (539,816 labelled tasks, user hash replaced
  by an integer rank).

The local raw copy in `data/raw` came from the Kaggle mirror; two of its tables (`pai_machine_metric`,
`pai_sensor_table`) print some floats with fewer digits than the official files, so `stage.py` is pointed at
header-prefixed official copies of those two first (see the commands below).

## Rebuild and run locally

    # 1. official copies of the two tables that differ (header line + official file)
    mkdir -p kaggle/_stage/official_dl kaggle/_stage/official_hdr
    for t in pai_machine_metric pai_sensor_table; do
      curl -L -o kaggle/_stage/official_dl/$t.tar.gz https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/$t.tar.gz
      tar -xzf kaggle/_stage/official_dl/$t.tar.gz -C kaggle/_stage/official_dl
    done
    head -1 data/raw/pai_machine_metric.csv > kaggle/_stage/official_hdr/pai_machine_metric.csv
    cat kaggle/_stage/official_dl/pai_machine_metric.csv >> kaggle/_stage/official_hdr/pai_machine_metric.csv
    head -1 ALIBABA-CLUSTER-TRACE-GPU-v2020/pai_sensor_table.csv > kaggle/_stage/official_hdr/pai_sensor_table.csv
    cat kaggle/_stage/official_dl/pai_sensor_table.csv >> kaggle/_stage/official_hdr/pai_sensor_table.csv

    # 2. stage the dataset (checks every CSV against the official checksums)
    python kaggle/datasets/alibaba-gpu-trace-2020/stage.py \
        --raw-dir kaggle/_stage/official_hdr --raw-dir data/raw --raw-dir ALIBABA-CLUSTER-TRACE-GPU-v2020 \
        --derived data/processed/v2/full_1h.parquet --out kaggle/_stage/input/alibaba-gpu-trace-2020

    # 3. build, execute and check the notebooks (Kaggle paths are shimmed to kaggle/_stage)
    python kaggle/tools/nbbuild.py run 01 02 03 04 05
    python kaggle/tools/nbbuild.py check

Notebook sources live in `tools/src/*.nb` (plain text with `#%% md` / `#%% code` markers; shared setup cells are
included with `#%% include`). Edit the source, not the `.ipynb`. `kaggle/_stage/` is git-ignored.

## Publishing (nothing has been pushed)

The dataset must exist before any notebook is pushed, because each notebook lists it in `dataset_sources`.
Pushing a notebook starts a Kaggle run of it (the same as Save & Run All).

    # 1. create the dataset (private; uploads about 3.8 GB)
    kaggle datasets create -p kaggle/_stage/input/alibaba-gpu-trace-2020

    # 2. push the notebooks (private, CPU; 03 to 05 have internet on for a pinned lightgbm fallback install)
    kaggle kernels push -p kaggle/notebooks/kanvas/01-pai-trace-eda
    kaggle kernels push -p kaggle/notebooks/kanvas/02-idle-gpus
    kaggle kernels push -p kaggle/notebooks/kanvas/03-failure-lightgbm-shap
    kaggle kernels push -p kaggle/notebooks/kanvas/04-kaam-from-scratch
    kaggle kernels push -p kaggle/notebooks/kanvas/05-splines-vs-shap

    # 3. check each run
    kaggle kernels status swapnilshukla007/<slug>

Make the dataset public before the first notebook, then publish one notebook every one or two days (set
`"is_private": false` and push again, or use the notebook's settings page). Upload each `cover.png` as the
notebook's cover image on Kaggle. The series links inside each notebook point to the final slugs and work once
the linked notebook is public.
