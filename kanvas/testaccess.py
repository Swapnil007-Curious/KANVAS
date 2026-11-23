"""The only route to test labels in v2; every call is logged.

The experiment runner fits models and writes test-set *predictions* without
reading a test label. Scoring code obtains labels through ``test_labels``, which
appends one JSON line per call to ``results/test_access_log.jsonl`` with the
time, the dataset, protocol, split, the purpose of the access and the SHA-256 of
``paper/PROTOCOL.md`` at that moment.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "results" / "test_access_log.jsonl"
PROTOCOL_PATH = ROOT / "paper" / "PROTOCOL.md"


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def test_labels(y_all, test_idx, *, dataset: str, protocol: str, split: str, purpose: str, log_path=None):
    """Return ``y_all[test_idx]`` and log the access."""
    entry = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset": dataset,
             "protocol": protocol, "split": str(split), "n_test": int(len(test_idx)), "purpose": purpose,
             "protocol_sha256": protocol_sha256()}
    path = Path(log_path) if log_path is not None else LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return np.asarray(y_all)[np.asarray(test_idx)]


def read_log(log_path=None):
    path = Path(log_path) if log_path is not None else LOG_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
