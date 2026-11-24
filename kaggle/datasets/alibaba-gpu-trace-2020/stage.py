"""Assemble the upload folder for the Kaggle dataset ``alibaba-gpu-trace-2020``.

What it does, in order:

1. Finds the seven trace tables in the ``--raw-dir`` folders. Each local CSV is
   the official headerless file with its header line (from the official
   ``.header`` file) prepended. The script checks that the bytes after the first
   line match the sha256 published in the official README, so the dataset holds
   the official data unchanged apart from that header line.
2. Writes ``kanvas_tasks_1h.parquet``: the KANVAS task table (539,816 labelled
   tasks, ten features, label, start time) made by ``kanvas/fulltrace.py`` with a
   one-hour lookback (``data/processed/v2/full_1h.parquet``). The user hash is
   replaced by ``user_rank`` (1 = the user with the most tasks in this table).
3. Copies README.md, LICENSE.txt, SHA256SUMS.txt and dataset-metadata.json next to
   the data. CSVs are hard-linked when possible, so staging needs no extra disk.

Usage (repository root):

    python kaggle/datasets/alibaba-gpu-trace-2020/stage.py \
        --raw-dir data/raw --raw-dir ALIBABA-CLUSTER-TRACE-GPU-v2020 \
        --derived data/processed/v2/full_1h.parquet \
        --out kaggle/_stage/input/alibaba-gpu-trace-2020

Then: ``kaggle datasets create -p kaggle/_stage/input/alibaba-gpu-trace-2020``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# sha256 of the official extracted (headerless) CSVs, from
# https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md
OFFICIAL_SHA256 = {
    "pai_group_tag_table.csv": "d23fcb56d88b2976fda4a62708d77b2a8c98635c4b443f100ed5553d4cd07fbe",
    "pai_instance_table.csv": "b179f7d7e0927a6663d719b728c92640447c5a0fc3c6e4edff31e7207ab9dd17",
    "pai_job_table.csv": "379ecb3becaba347f44a53bf7eb53e54b185221b2a0338a3f828828d269ba96c",
    "pai_machine_metric.csv": "8b6592daab0373ef7fec730198b4df5891b1722c2ab4c7016249166e131402d3",
    "pai_machine_spec.csv": "251e9e33649155a6a935fdc1dc0e90cd4698206936b60bd8fce60ff2ccd51439",
    "pai_sensor_table.csv": "12dd9929b70f3efe18d1279d9873e3a519c64f431613b23b059ed8c46a376dd7",
    "pai_task_table.csv": "6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de",
}

FEATURES = ["plan_cpu", "plan_mem", "plan_gpu", "inst_num", "avg_cpu_usr", "avg_gpu_util", "avg_cpu_kernel",
            "avg_load_1", "cap_gpu", "avg_net_receive"]
DERIVED_NAME = "kanvas_tasks_1h.parquet"
DOCS = ["README.md", "LICENSE.txt", "dataset-metadata.json"]


def sha256_after_header(path: Path, chunk=1 << 22):
    """Return (sha256 of the file without its first line, sha256 of the whole file)."""
    body, whole = hashlib.sha256(), hashlib.sha256()
    with open(path, "rb") as f:
        first = f.readline()
        whole.update(first)
        while True:
            block = f.read(chunk)
            if not block:
                break
            body.update(block)
            whole.update(block)
    return body.hexdigest(), whole.hexdigest()


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_derived(src: Path, dst: Path) -> pd.DataFrame:
    """Write the KANVAS task table with an integer user rank instead of the user hash."""
    d = pd.read_parquet(src).reset_index()
    counts = d.groupby("user").size().rename("n").reset_index()
    counts = counts.sort_values(["n", "user"], ascending=[False, True], kind="mergesort")
    rank = pd.Series(range(1, len(counts) + 1), index=counts["user"].to_numpy())
    d["user_rank"] = rank.reindex(d["user"].to_numpy()).to_numpy().astype("int32")
    out = d[["job_name", "task_name"] + FEATURES + ["label", "start_time", "user_rank", "machine", "n_metric_rows"]]
    out = out.astype({"job_name": "string", "task_name": "string", "machine": "string", "label": "int8",
                      "n_metric_rows": "int32"})
    out.to_parquet(dst, index=False, compression="zstd")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", action="append", required=True)
    ap.add_argument("--derived", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-hash", action="store_true", help="skip the sha256 check (faster restaging)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, official in OFFICIAL_SHA256.items():
        src = next((Path(r) / name for r in args.raw_dir if (Path(r) / name).exists()), None)
        if src is None:
            raise FileNotFoundError(f"{name} not found in {args.raw_dir}")
        if not args.skip_hash:
            body, whole = sha256_after_header(src)
            if body != official:
                raise ValueError(f"{name}: content after the header line does not match the official sha256")
            lines.append(f"{whole}  {name}  (this file)\n{official}  {name}  (official headerless file = this "
                         f"file without its first line)")
            print(f"ok  {name}")
        link_or_copy(src, out / name)

    d = make_derived(Path(args.derived), out / DERIVED_NAME)
    print(f"ok  {DERIVED_NAME}: {len(d):,} tasks, failure rate {d['label'].mean():.4f}, "
          f"{d['user_rank'].nunique():,} users, user 1 has {int((d['user_rank'] == 1).sum()):,} tasks")

    for doc in DOCS:
        shutil.copy2(HERE / doc, out / doc)
    if lines:
        (HERE / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    if (HERE / "SHA256SUMS.txt").exists():
        shutil.copy2(HERE / "SHA256SUMS.txt", out / "SHA256SUMS.txt")
    print(f"staged {len(list(out.iterdir()))} files")


if __name__ == "__main__":
    main()
