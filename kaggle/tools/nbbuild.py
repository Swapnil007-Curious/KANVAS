"""Build, execute and check the Kaggle notebooks of the KANVAS series.

Each notebook is written as a plain-text source file in ``kaggle/tools/src`` with
cell markers, so it can be diffed and reviewed like code:

    #%% md            a markdown cell
    #%% code          a code cell
    #%% code hide     a code cell whose input Kaggle hides (``_kg_hide-input``)
    #%% include NAME  the cells of ``src/NAME`` inlined here

Usage (from the repository root, with the project's virtual environment):

    python kaggle/tools/nbbuild.py build 01            # write notebook.ipynb only
    python kaggle/tools/nbbuild.py run 01              # build, execute, scrub, copy cover.png
    python kaggle/tools/nbbuild.py run 03 --fast       # execute with KANVAS_FAST=1
    python kaggle/tools/nbbuild.py check               # nbformat, leak and style checks on all notebooks

Execution uses a Kaggle path shim: the notebook reads ``KAGGLE_INPUT_ROOT`` and
``KAGGLE_WORKING`` when they are set and falls back to ``/kaggle/input`` and
``/kaggle/working``. ``run`` points them at ``kaggle/_stage/input`` (made by
``kaggle/datasets/alibaba-gpu-trace-2020/stage.py``) and a scratch working folder.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
KAGGLE = ROOT / "kaggle"
SRC = KAGGLE / "tools" / "src"
NB_ROOT = KAGGLE / "notebooks" / "kanvas"
STAGE_INPUT = KAGGLE / "_stage" / "input"
WORK_ROOT = KAGGLE / "_stage" / "working"
RUN_LOG = NB_ROOT / "run_log.json"

# notebook number -> (source file, folder)
NOTEBOOKS = {
    "01": ("k1_trace_eda.nb", "01-pai-trace-eda"),
    "02": ("k2_idle_gpus.nb", "02-idle-gpus"),
    "03": ("k3_failure_lgbm_shap.nb", "03-failure-lightgbm-shap"),
    "04": ("k4_kaam_from_scratch.nb", "04-kaam-from-scratch"),
    "05": ("k5_splines_vs_shap.nb", "05-splines-vs-shap"),
}

MARKER = re.compile(r"^#%%\s*(md|code|include)\s*(.*)$")

# Style rules for markdown (and rendered markdown output) from the series brief.
BANNED_WORDS = ["delve", "crucial", "pivotal", "robust", "showcase", "leverage", "utilize", "utilise",
                "landscape", "tapestry", "testament", "seamless", "unlock", "game-changer", "game changer",
                "meticulous", "furthermore", "additionally"]
DASHES = ["—", "–"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


# ----------------------------------------------------------------------------- parsing
def parse_source(path: Path) -> list:
    """Return a list of nbformat cells from a ``.nb`` source file (includes resolved)."""
    cells, kind, opts, buf = [], None, "", []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "code":
            cell = nbformat.v4.new_code_cell(text)
            if "hide" in opts.split():
                cell.metadata["_kg_hide-input"] = True
                cell.metadata["jupyter"] = {"source_hidden": True}
            cells.append(cell)

    for line in path.read_text(encoding="utf-8").splitlines():
        m = MARKER.match(line)
        if m:
            flush()
            kind, opts, buf = m.group(1), m.group(2).strip(), []
            if kind == "include":
                cells.extend(parse_source(SRC / opts))
                kind = None
            continue
        buf.append(line)
    flush()
    return cells


def build(num: str) -> Path:
    """Write ``notebook.ipynb`` (unexecuted) for notebook ``num``; return its path."""
    src_name, folder = NOTEBOOKS[num]
    nb = nbformat.v4.new_notebook()
    nb.cells = parse_source(SRC / src_name)
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    out = NB_ROOT / folder / "notebook.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


# ----------------------------------------------------------------------------- execution
def scrub_strings() -> list:
    """Absolute paths and account names that must never appear in a committed output."""
    user = getpass.getuser()
    home = str(Path.home())
    cands = {str(ROOT), str(STAGE_INPUT), str(WORK_ROOT), home, home.replace("\\", "/"), str(ROOT).replace("\\", "/")}
    cands |= {c.replace("\\", "\\\\") for c in list(cands)}
    return sorted([c for c in cands if c], key=len, reverse=True), user


def scrub_text(text: str) -> str:
    paths, user = scrub_strings()
    for p in paths:
        text = text.replace(p, "<path>")
    if user and len(user) >= 3:
        text = re.sub(rf"\b{re.escape(user)}\b", "<user>", text)
    return text


def scrub_outputs(nb) -> None:
    """Replace local paths and the account name in every output; drop timing metadata."""
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = scrub_text(out["text"])
            for key, val in list(out.get("data", {}).items()):
                if key.startswith("text/") or key == "application/json":
                    out["data"][key] = scrub_text(val) if isinstance(val, str) else val
            if out.get("output_type") == "error":
                out["traceback"] = [scrub_text(t) for t in out.get("traceback", [])]


def run(num: str, fast: bool, timeout: int) -> dict:
    """Build and execute notebook ``num`` with the Kaggle path shim; return the run record."""
    from nbclient import NotebookClient

    path = build(num)
    nb = nbformat.read(path, as_version=4)
    work = WORK_ROOT / NOTEBOOKS[num][1]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.environ["KAGGLE_INPUT_ROOT"] = str(STAGE_INPUT)
    os.environ["KAGGLE_WORKING"] = str(work)
    os.environ["KANVAS_FAST"] = "1" if fast else "0"
    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    start = time.perf_counter()
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(work)}})
    try:
        client.execute()
    except Exception as exc:  # report the failing cell briefly instead of a coloured traceback
        name = type(exc).__name__
        msg = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
        tail = "\n".join(msg.strip().splitlines()[-6:])
        raise SystemExit(f"notebook {num} failed: {name}\n{tail}") from None
    seconds = time.perf_counter() - start
    scrub_outputs(nb)
    nbformat.validate(nb)
    nbformat.write(nb, path)

    cover = work / "cover.png"
    if not cover.exists():
        raise FileNotFoundError(f"notebook {num} did not write cover.png")
    from PIL import Image

    with Image.open(cover) as im:
        if im.size != (1280, 720):
            raise ValueError(f"cover.png is {im.size}, expected (1280, 720)")
    shutil.copy2(cover, path.parent / "cover.png")

    record = {"notebook": NOTEBOOKS[num][1], "seconds": round(seconds, 1), "fast": fast,
              "python": platform.python_version(), "cpu_threads": os.cpu_count(),
              "date": time.strftime("%Y-%m-%d")}
    log = json.loads(RUN_LOG.read_text()) if RUN_LOG.exists() else {}
    log[num] = record
    RUN_LOG.write_text(json.dumps(log, indent=2) + "\n")
    return record


# ----------------------------------------------------------------------------- checks
def _markdown_texts(nb):
    """Markdown cell sources plus rendered markdown outputs (the 'What we found' cells)."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            yield i, cell.source
        elif cell.cell_type == "code":
            for out in cell.get("outputs", []):
                md = out.get("data", {}).get("text/markdown")
                if md:
                    yield i, md if isinstance(md, str) else "".join(md)


def check(nums) -> int:
    """Validate notebooks and metadata; print every problem; return the number of problems."""
    problems = 0
    paths, user = scrub_strings()
    leak_patterns = [re.compile(re.escape(p), re.I) for p in paths if len(p) > 3]
    leak_patterns += [re.compile(r"[A-Za-z]:\\\\?Users", re.I), re.compile(r"/home/\w+"),
                      re.compile(r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}", re.I),
                      re.compile(r"KGAT_[A-Za-z0-9]+"), re.compile(r"gsk_[A-Za-z0-9]{10,}"),
                      re.compile(r"[\w.+-]+@(gmail|yahoo|outlook|hotmail)\.com", re.I)]
    if user and len(user) >= 3:
        leak_patterns.append(re.compile(rf"\b{re.escape(user)}\b"))
    for num in nums:
        folder = NB_ROOT / NOTEBOOKS[num][1]
        path = folder / "notebook.ipynb"
        if not path.exists():
            print(f"[{num}] missing notebook.ipynb"); problems += 1; continue
        nb = nbformat.read(path, as_version=4)
        try:
            nbformat.validate(nb)
        except Exception as exc:  # noqa: BLE001
            print(f"[{num}] nbformat: {exc}"); problems += 1
        raw = path.read_text(encoding="utf-8")
        for pat in leak_patterns:
            for m in pat.finditer(raw):
                ctx = raw[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                print(f"[{num}] leak {pat.pattern!r}: ...{ctx}..."); problems += 1
        for cell in nb.cells:
            if cell.cell_type == "code":
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "error":
                        print(f"[{num}] error output: {out.get('ename')}: {out.get('evalue')}"); problems += 1
                    if out.get("name") == "stderr" and "Warning" in out.get("text", ""):
                        print(f"[{num}] warning in stderr: {out['text'][:160]!r}")
        for i, text in _markdown_texts(nb):
            low = text.lower()
            for w in BANNED_WORDS:
                if re.search(rf"\b{re.escape(w)}", low):
                    print(f"[{num}] cell {i}: banned word {w!r}"); problems += 1
            for d in DASHES:
                if d in text:
                    print(f"[{num}] cell {i}: em/en dash"); problems += 1
            if EMOJI.search(text):
                print(f"[{num}] cell {i}: emoji"); problems += 1
            for line in text.splitlines():
                if line.startswith("#"):
                    words = line.lstrip("#").strip().split()
                    if words and re.fullmatch(r"\d+\.", words[0]):
                        words = words[1:]  # numbered heading: the first real word may be capitalised
                    caps = [w for w in words[1:] if w[:1].isupper() and not _allowed_cap(w)]
                    if caps:
                        print(f"[{num}] cell {i}: heading not sentence case? {line!r}")
        meta_path = folder / "kernel-metadata.json"
        if not meta_path.exists():
            print(f"[{num}] missing kernel-metadata.json"); problems += 1
        else:
            meta = json.loads(meta_path.read_text())
            if meta.get("is_private") is not True:
                print(f"[{num}] kernel-metadata is not private"); problems += 1
            slug = meta["id"].split("/")[1]
            if slug != _slugify(meta["title"]):
                print(f"[{num}] slug {slug!r} does not match title {meta['title']!r}"); problems += 1
            if not 5 <= len(meta["title"]) <= 50:
                print(f"[{num}] title length {len(meta['title'])} outside 5..50"); problems += 1
        if not (folder / "cover.png").exists():
            print(f"[{num}] missing cover.png"); problems += 1
    for path in sorted(list((KAGGLE / "datasets").rglob("*")) + [KAGGLE / "README.md"]):
        if path.is_file() and path.suffix in {".md", ".json", ".txt", ".py"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pat in leak_patterns:
                for m in pat.finditer(text):
                    print(f"[{path.relative_to(ROOT)}] leak {pat.pattern!r}: {text[max(0, m.start() - 30):m.end() + 30]!r}")
                    problems += 1
    print(f"{problems} problem(s)")
    return problems


_ALLOWED_CAPS = {"Alibaba", "PAI", "GPU", "GPUs", "SHAP", "KAAM", "KANVAS", "LightGBM", "PyTorch", "B-spline",
                 "B-splines", "Kolmogorov-Arnold", "ROC-AUC", "AUC", "CPU", "CPUs", "MLP", "EDA", "T4", "P100",
                 "V100", "V100M32", "MISC", "I", "User", "SiLU", "Cox-de", "Boor", "NSDI", "CC", "BY", "4.0"}


def _allowed_cap(word):
    w = word.strip("():,.?'\"")
    return w in _ALLOWED_CAPS or w.isupper() or any(ch.isdigit() for ch in w)


def _slugify(title):
    s = re.sub(r"[^a-z0-9\s-]", "", title.lower())
    return re.sub(r"[\s-]+", "-", s).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "check"])
    ap.add_argument("nums", nargs="*")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    nums = args.nums or sorted(NOTEBOOKS)
    if args.cmd == "build":
        for n in nums:
            print(build(n).relative_to(ROOT))
    elif args.cmd == "run":
        for n in nums:
            print(json.dumps(run(n, args.fast, args.timeout)))
    else:
        sys.exit(1 if check(nums) else 0)


if __name__ == "__main__":
    main()
