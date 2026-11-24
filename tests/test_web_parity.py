"""The static demo (docs/) computes the same failure probability as the canonical PyTorch KAAM.

The same 1,000 test tasks of the canonical split (full trace, random split 42) go through
  * the PyTorch model (kanvas.models_v2.load_kaam on kaam_states.pt[0], with the split's preprocessing),
  * the page's arithmetic in numpy (scripts/export_web.py: web_predict, piecewise-linear interpolation on the
    curves in docs/model.json, exact values at discrete levels), and
  * docs/kaam.js itself, run with node.
Each must agree with PyTorch to 1e-3 in probability. The AUCs on these rows are printed; the labels are read
through kanvas.testaccess, so each run adds one line to results/test_access_log.jsonl.

    pytest tests/test_web_parity.py -s
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_web as W  # noqa: E402

DOCS = ROOT / "docs"
N_ROWS = 1000
TOL = 1e-3
FORBIDDEN = {"user", "user_id", "job_name", "task_name", "worker_name", "machine", "inst_id", "group", "task_id"}

NODE_RUNNER = r"""
const fs = require("fs");
const [kaamPath, modelPath, rowsPath] = process.argv.slice(2);
const KAAM = require(kaamPath);
const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const rows = JSON.parse(fs.readFileSync(rowsPath, "utf8"));
const out = rows.map((r) => {
  const p = KAAM.predict(model, r);
  const sum = model.baseline_logit + model.features.reduce((s, n) => s + p.contrib[n], 0);
  return [p.risk, p.logit, sum];
});
process.stdout.write(JSON.stringify(out));
"""


def _model_json():
    path = DOCS / "model.json"
    if not path.exists():
        pytest.skip("docs/model.json not written yet (python scripts/export_web.py)")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def parity():
    model = _model_json()
    if not W.MATRIX.exists() or not (W.RUN / "kaam_states.pt").exists():
        pytest.skip("published feature matrix or canonical run missing")
    torch_model, pp, split = W.load_canonical()
    X = W.load_matrix()
    rows = np.sort(np.random.default_rng(0).choice(split["test"], N_ROWS, replace=False))
    X_rows = X.iloc[rows][W.FEATURES].reset_index(drop=True)
    return {"model": model, "rows": rows, "X": X_rows,
            "p_torch": W.torch_predict(torch_model, pp, X_rows), "p_web": W.web_predict(model, X_rows)}


def test_python_page_math_matches_pytorch(parity):
    from sklearn.metrics import roc_auc_score

    from kanvas.testaccess import test_labels

    gap = np.abs(parity["p_web"] - parity["p_torch"])
    y_all = pd.read_csv(W.MATRIX, usecols=["label"])["label"].to_numpy()
    y = test_labels(y_all, parity["rows"], dataset="full", protocol="random", split="s42",
                    purpose="web demo parity check on 1,000 test tasks (tests/test_web_parity.py)")
    auc_t, auc_w = roc_auc_score(y, parity["p_torch"]), roc_auc_score(y, parity["p_web"])
    print(f"\n[web parity] {N_ROWS} test tasks: python page math vs PyTorch max |dp| {gap.max():.2e}, "
          f"mean {gap.mean():.2e}; ROC-AUC PyTorch {auc_t:.6f}, page math {auc_w:.6f} "
          f"(difference {auc_w - auc_t:+.2e})")
    assert gap.max() < TOL
    assert abs(auc_t - auc_w) < 1e-3


def test_node_kaam_js_matches_pytorch(parity, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(parity["X"].to_dict(orient="records")), encoding="utf-8")
    runner = tmp_path / "runner.js"
    runner.write_text(NODE_RUNNER, encoding="utf-8")
    res = subprocess.run([node, str(runner), str(DOCS / "kaam.js"), str(DOCS / "model.json"), str(rows_path)],
                         capture_output=True, text=True, timeout=120, check=True)
    out = np.asarray(json.loads(res.stdout), dtype=np.float64)
    p_js, logit_js, sum_js = out[:, 0], out[:, 1], out[:, 2]
    gap_t = np.abs(p_js - parity["p_torch"])
    gap_w = np.abs(p_js - parity["p_web"])
    print(f"\n[web parity] {N_ROWS} test tasks: node kaam.js vs PyTorch max |dp| {gap_t.max():.2e}; "
          f"vs python page math {gap_w.max():.1e}; contributions sum to the logit within "
          f"{np.abs(sum_js - logit_js).max():.1e}")
    assert gap_t.max() < TOL
    assert gap_w.max() < 1e-12  # same arithmetic in double precision
    assert np.abs(sum_js - logit_js).max() < 1e-9


def test_model_json_is_small_and_complete():
    model = _model_json()
    size = (DOCS / "model.json").stat().st_size
    assert size < 300 * 1024, f"model.json is {size / 1024:.0f} KB"
    assert model["features"] == W.FEATURES
    for n in W.FEATURES:
        c = model["curves"][n]
        assert len(c["x"]) == len(c["phi"]) == W.N_GRID
        assert c["x"][0] == model["preprocess"]["clip"][n][0] and c["x"][-1] == model["preprocess"]["clip"][n][1]
        assert np.all(np.diff(c["x"]) > 0)
        assert {"label", "unit", "scale"} <= set(model["meta"][n])
    assert len(model["paper"]["rows"]) == 4 and all(len(r["cells"]) == 7 for r in model["paper"]["rows"])


def test_audit_table_shows_the_papers_numbers():
    """The page prints three decimals; each must equal the paper's (no double rounding)."""
    model = _model_json()
    ci = pd.read_csv(W.ANALYSIS / "model_ci.csv")
    for (key, label), row in zip(W.MODELS, model["paper"]["rows"]):
        assert row["model"] == label
        for (proto, subset, _), cell in zip(W.COLUMNS, row["cells"]):
            ref = ci[(ci.protocol == proto) & (ci.subset == subset) & (ci.model == key)].iloc[0]
            for k, col in (("auc", "auc"), ("lo", "ci_low"), ("hi", "ci_high")):
                assert f"{cell[k]:.3f}" == f"{ref[col]:.3f}", (label, proto, subset, k)


def test_examples_are_real_test_tasks_without_identifiers():
    path = DOCS / "examples.json"
    if not path.exists():
        pytest.skip("docs/examples.json not written yet")
    examples = json.loads(path.read_text(encoding="utf-8"))["examples"]
    assert len(examples) == 30
    assert {e["label"] for e in examples} == {0, 1}
    for e in examples:
        assert set(e) <= {"id", "label", "x", "paper_case"}
        assert list(e["x"]) == W.FEATURES
        assert not {k.lower() for k in e["x"]} & FORBIDDEN
    if W.MATRIX.exists() and (W.RUN / "split.npz").exists():
        X = W.load_matrix()
        test = X.iloc[np.load(W.RUN / "split.npz")["test"]][W.FEATURES].to_numpy()
        keys = {tuple(r) for r in test}
        assert all(tuple(e["x"][n] for n in W.FEATURES) in keys for e in examples)
    # failures and successes, caught and missed, at the paper's threshold
    model = _model_json()
    risk = W.web_predict(model, pd.DataFrame([e["x"] for e in examples]))
    flagged = risk >= model["model"]["threshold"]
    label = np.array([e["label"] for e in examples])
    for want_label in (0, 1):
        for want_flag in (False, True):
            assert ((label == want_label) & (flagged == want_flag)).sum() >= 5


def test_site_uses_relative_paths_only():
    for name in ("index.html", "app.js", "style.css"):
        path = DOCS / name
        if not path.exists():
            pytest.skip(f"docs/{name} missing")
        text = path.read_text(encoding="utf-8")
        bad = re.findall(r"""(?:src|href)\s*=\s*["']/(?!/)[^"']*["']|url\(\s*["']?/(?!/)|fetch\(\s*["']/(?!/)""", text)
        assert not bad, f"{name} has root-absolute paths: {bad}"
