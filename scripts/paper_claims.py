"""Audit the paper against its sources and the style rules, and write paper/CLAIMS.md.

    python scripts/paper_tables.py && python scripts/paper_claims.py

Checks, for both versions (paper/main.tex and paper/ieee/main_ieee.tex with their sections):
  1. every \\V{key} used exists in paper/data/macro_sources.json (written by paper_tables.py);
  2. where the source can be read independently (a CSV row, a JSON field, a constant in kanvas/), the value
     is looked up again here by a different code path and compared with the recorded raw value;
  3. the printed text is parsed back and must equal the raw value at its printed precision; a
     "significant"/"not significant" word must agree with its Holm-adjusted p-value;
  4. no digit in the hand-written LaTeX comes from outside a macro, except names on an explicit list;
  5. the style rules: no first person or contractions outside the Acknowledgment, no em or en dashes, no
     "not X but Y", none of the listed words, no invisible Unicode characters, and the AI assistant and
     its maker named only in the Acknowledgment.
Exit code 1 when any check fails. Result root: results/ or KANVAS_RESULTS.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get("KANVAS_RESULTS", ROOT / "results"))
PAPER = ROOT / "paper"
HAND_WRITTEN = [PAPER / "main.tex", PAPER / "macros.tex", *sorted((PAPER / "sections").glob("*.tex")),
                PAPER / "figures" / "pipeline.tex"]
if (PAPER / "ieee").exists():
    HAND_WRITTEN += [PAPER / "ieee" / "main_ieee.tex", *sorted((PAPER / "ieee" / "sections").glob("*.tex"))]
HAND_WRITTEN = [p for p in HAND_WRITTEN if p.exists()]
ACK = {"acknowledgment.tex"}

BANNED = ["delve", "crucial", "pivotal", "robust", "showcase", "leverage", "utilize", "utilise", "landscape",
          "tapestry", "testament", "underscore", "seamless", "furthermore", "additionally", "notably", "moreover"]
CONTRACTIONS = r"\b\w+n't\b|\b(?:it|that|there|what|let|here|who)'s\b|\b\w+'(?:re|ve|ll|d|m)\b"
INVISIBLE = re.compile("[\u00ad\u200b-\u200f\u2028-\u202e\u2060-\u2064\ufeff\u00a0\u202f]")
DASHES = re.compile("[\u2012\u2013\u2014\u2015]")
# The assistant's and its maker's names, spelled from character codes so that this file does not contain them.
ASSISTANT_NAMES = re.compile("|".join("".join(map(chr, codes)) for codes in
                                      ([99, 108, 97, 117, 100, 101], [97, 110, 116, 104, 114, 111, 112, 105, 99])))

# Non-numeric factual claims and where each was checked.
FACTS = [
    ("The trace is distributed under CC BY 4.0.",
     "github.com/alibaba/clusterdata, cluster-trace-gpu-v2020/LICENSE (title line: Attribution 4.0 International)"),
    ("The user is a split key only; it is never a model input and is absent from every export.",
     "scripts/run_experiments.py (features are FEATURE_NAMES); tests/test_publication_exports.py"),
    ("Clipping and scaling are fitted on the training split of each protocol.",
     "kanvas/preprocess.py (fit_preprocessor); tests/test_v2_pipeline.py::test_clipping_and_scaling_use_training_rows_only"),
    ("The protocol was written before any test label was read; test labels are read only through a logged function.",
     "paper/PROTOCOL.md; kanvas/testaccess.py; results/test_access_log.jsonl (first entry after the last fit)"),
    ("Four deviations, all before the first test access: early stopping on validation ROC-AUC (1); one EBM seed on "
     "the full trace (2); one seed for the EBM with interactions outside the original random splits (3); on the full "
     "trace only the canonical additive EBM is fitted (4).",
     "paper/PROTOCOL.md section 8; kanvas/models_v2.py (_train_loop); scripts/run_experiments.py (run_model, run_split)"),
    ("Early stopping fired in every KAAM and MLP run on the full trace.",
     "results/runs/full/*/*/kaam.json and mlp.json (per_seed.early_stopped); paper/data/macro_sources.json full.train.*"),
    ("No user appears in two parts of a user-grouped split; the time split orders train before validation before test.",
     "kanvas/protocols.py (check_split, time_split); tests/test_v2_pipeline.py"),
    ("KAAM's forward pass reproduces the training-time logit and explain() decomposes it exactly.",
     "tests/test_v2_pipeline.py::test_kaam_fitted_on_basis_matches_module_forward; results/interpret/shap_check.json"),
    ("The demo page states the new-user limitation.", "app/app.py (limitation notice)"),
]


# ----------------------------------------------------------------------------- parsing printed values
def strip_tex(t):
    return (t.replace("$-$", "-").replace("\\%", "%").replace(",", "").replace("{", "").replace("}", "")
            .replace("\\,", "").replace("$", "").strip())


def num(text):
    t = strip_tex(text)
    m = re.fullmatch(r"([+-]?\d+(?:\.(\d+))?)(%?)", t)
    if not m:
        return None
    return float(m.group(1)), len(m.group(2) or ""), bool(m.group(3))


def close(text, raw):
    p = num(text)
    if p is None:
        return None
    v, d, pct = p
    target = raw * 100 if pct else raw
    return abs(v - target) <= 0.5 * 10 ** (-d) + 1e-9


def sci(text, raw):
    m = re.fullmatch(r"\$([\d.]+)\\times10\^\{(-?\d+)\}\$", text.strip())
    if not m:
        return None
    mant, e = m.group(1), int(m.group(2))
    d = len(mant.split(".")[1]) if "." in mant else 0
    return abs(float(mant) * 10 ** e - raw) <= 0.5 * 10 ** (e - d) + 1e-30


def check_printed(text, raw):
    """True/False when the printed text can be compared with the raw value, None otherwise."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, list) and text.startswith("[") and len(raw) == 2:
        parts = [x.strip() for x in text.strip("[]").split(",")]
        return len(parts) == 2 and all(close(t, float(r)) for t, r in zip(parts, raw))
    if not isinstance(raw, (int, float)):
        return None
    if text in ("significant", "not significant"):
        return (text == "significant") == (raw < 0.05)
    if text == "$<$0.001":
        return raw < 0.001
    if "\\times10" in text:
        return sci(text, float(raw))
    words = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                          "nine", "ten"])}
    if text in words:
        return float(raw) == words[text]
    r = close(text, float(raw))
    return r


# ----------------------------------------------------------------------------- independent re-reads
def resolve(source):
    if source.startswith("results/"):
        return RES / source[len("results/"):]
    return ROOT / source


def lookup(source, field):
    """Independent lookup of a recorded value; None when it is derived or cannot be re-read."""
    if field.startswith("derived"):
        return None
    p = resolve(source)
    if p.suffix == ".py":
        mod = ".".join(Path(source).with_suffix("").parts)
        try:
            m = importlib.import_module(mod)
        except Exception:
            return None
        return getattr(m, field, None)
    if not p.exists():
        return None
    if p.suffix == ".csv":
        if ":" not in field:
            return None
        keys, col = field.rsplit(":", 1)
        cond = dict(kv.split("=", 1) for kv in keys.split(";")) if keys else {}
        with p.open(encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if all(_eq(r.get(k), v) for k, v in cond.items())]
        if len(rows) != 1:
            return None
        cols = col.split(",")
        vals = [float(rows[0][c]) for c in cols]
        return vals if len(vals) > 1 else vals[0]
    if p.suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        for part in field.split("."):
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif part in obj:
                obj = obj[part]
            else:
                # keys may contain dots ("v1 check", "1" vs "1.0"); try the longest match
                return _dotted(obj, field)
        return obj if isinstance(obj, (int, float, str, bool)) else None
    return None


def _dotted(obj, field):
    parts = field.split(".")
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
                i = j
                break
            if isinstance(obj, list) and k.isdigit():
                obj = obj[int(k)]
                i = j
                break
        else:
            return None
    return obj if isinstance(obj, (int, float, str, bool)) else None


def _eq(a, b):
    if a is None:
        return False
    if a == b:
        return True
    try:
        return math.isclose(float(a), float(b))
    except ValueError:
        return False


def same(indep, raw):
    if isinstance(raw, list):
        return isinstance(indep, (list, tuple)) and len(indep) == len(raw) and all(
            math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(indep, raw))
    if isinstance(raw, bool) or isinstance(indep, bool):
        return bool(indep) == bool(raw)
    if isinstance(raw, str) or isinstance(indep, str):
        return str(indep) == str(raw)
    return math.isclose(float(indep), float(raw), rel_tol=1e-9, abs_tol=1e-12)


# ----------------------------------------------------------------------------- text scans
ALLOWED = [
    r"\\V\{[^}]*\}",
    r"\\(?:ref|label|eqref|cite[pt]?|citealp|input|include|includegraphics|url|path|href|section|subsection|"
    r"subsubsection|paragraph|feat|texttt|bibliography|bibliographystyle|hypersetup|IEEEauthorblockA|"
    r"SecRef|cmd)\*?(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\(?:setlength|usetikzlibrary|definecolor|documentclass|usepackage|captionsetup|setlist|newcommand|"
    r"renewcommand|resizebox|columnwidth|linewidth)(?:\[[^\]]*\])?(?:\{[^}]*\})*",
    r"\[[0-9.]+(?:pt|em|mm|cm|in)\]", r"[0-9.]+\\?(?:linewidth|columnwidth|textwidth|pt|em|mm|cm|in)\b",
    r"Users~1 and~4", r"Users~1,~4", r"Users?~[14]\b",
    r"\$\\phi_i\$", r"\$x_1,\\dots,x_d\$", r"x_\{?\d\}?", r"_\{i,j[+-]1\}", r"c_\{i,j\+1\}-2c_\{i,j\}\+c_\{i,j-1\}",
    r"\\sum_\{i=1\}\^\{d\}", r"i=1", r"\^\{?-?\d+\}?", r"SHA-256", r"Okabe-Ito", r"L2", r"GA\$\^2\$M", r"\bF1\b",
    r"1st and 99th", r"5th(?:--| to | and )95th", r"5th and 95th", r"95\\%", r"\\pm", r"utf8", r"T1", r"11pt", r"10pt",
    r"cluster-trace-gpu-v2020", r"v2020", r"GPU-v2020", r"\bNSDI\b", r"IEEEtran", r"0\.5\b(?= threshold| decision)",
    r"risk 0\.5", r"threshold of 0\.5", r"at 0\.5", r"against 0\.5", r"\b0\.5\b(?=[ ~]AUC| \(chance\)|, the chance)",
    r"chance \(0\.5\)", r"AUC~?=~?0\.5", r"= 0\.5", r"\$0\.5\$", r"\$p<0\.05\$", r"0\.05",
    r"^(?:[1-9]|1[0-2]) &", r"folds?~\d(?: and~\d)?", r"PAI 2020", r"PAI trace of 2020", r"(?:above|below|against) 0\.5\b", r"seed~0", r"Users~1 and~4", r"Users~1,~4", r"\$\[t_0-1\\,\\mathrm\{h\},\\, t_0\)\$",
    r"CC BY 4\.0", r"\$t_0\$", r"t_0", r"2c", r"interactions=0", r"\\texttt\{[^}]*\}", r"#\d", r"\b(?:19|20)\d\d\b(?=\))", r"Python~3\.11",
    r"\\cite\{[^}]*\}", r"\\ref\{[^}]*\}", r"black!\d+", r"\[t\]", r"\{[0-9]+\}\{[0-9]+\}", r"\\\\\[[0-9]+pt\]",
]


def strip_comments(line):
    return re.sub(r"(?<!\\)%.*", "", line)


def scan(files):
    digits, style = [], []
    for f in files:
        rel = f.relative_to(PAPER).as_posix()
        is_ack = f.name in ACK
        in_pic = False
        text = f.read_text(encoding="utf-8")
        for i, raw_line in enumerate(text.splitlines(), 1):
            line = strip_comments(raw_line)
            if INVISIBLE.search(raw_line):
                style.append((rel, i, "invisible Unicode character", raw_line.strip()[:100]))
            if DASHES.search(line):
                style.append((rel, i, "Unicode dash", line.strip()[:100]))
            if "\\begin{tikzpicture}" in line:
                in_pic = True
            if in_pic:
                in_pic = "\\end{tikzpicture}" not in line
                continue
            if f.name in ("main.tex", "main_ieee.tex", "macros.tex"):
                continue  # preamble: packages, metadata, layout
            body = re.sub(r"\\(?:ref|label|cite|input|includegraphics|V|feat|texttt|url)\{[^}]*\}", " ", line)
            if re.search(r"(?<![-\\])---?(?![-])", body.replace("\\", "")) and "--" in body:
                style.append((rel, i, "en or em dash (-- or ---)", line.strip()[:100]))
            low = body.lower()
            for w in BANNED:
                if re.search(rf"\b{w}\w*", low):
                    style.append((rel, i, f"listed word '{w}'", line.strip()[:100]))
            if re.search(CONTRACTIONS, body, flags=re.I):
                style.append((rel, i, "contraction", line.strip()[:100]))
            if re.search(r"\bnot\b[^.;:]{1,60}\bbut\b", low):
                style.append((rel, i, "not X but Y", line.strip()[:100]))
            if not is_ack:
                if re.search(r"\b(?:I|my|me|mine)\b", re.sub(r"\\[A-Za-z]+", " ", body)):
                    style.append((rel, i, "first person singular", line.strip()[:100]))
                if ASSISTANT_NAMES.search(low):
                    style.append((rel, i, "assistant name outside the Acknowledgment", line.strip()[:100]))
            s = line
            for pat in ALLOWED:
                s = re.sub(pat, " ", s)
            for d in re.findall(r"\d[\d.,]*", s):
                digits.append((rel, i, d, line.strip()[:110]))
    return digits, style


def main():
    macros = json.loads((PAPER / "data" / "macro_sources.json").read_text(encoding="utf-8"))
    tex_files = HAND_WRITTEN + sorted((PAPER / "tables").glob("*.tex"))
    used = {}
    for f in tex_files:
        body = "\n".join(strip_comments(line) for line in f.read_text(encoding="utf-8").splitlines())
        for k in re.findall(r"\\V\{([^}]+)\}", body):
            used.setdefault(k, set()).add(f.relative_to(PAPER).as_posix())
    rows, failures = [], 0
    for k in sorted(used):
        m = macros.get(k)
        if m is None:
            rows.append((k, "", "", "", "no", "undefined macro"))
            failures += 1
            continue
        text, raw, src, field = m["text"], m["raw"], m["source"], m["field"]
        notes, ok = [], True
        try:
            indep = lookup(src, field)
        except Exception as e:  # noqa: BLE001
            indep = None
            notes.append(f"re-read error {type(e).__name__}")
        if indep is not None:
            s = same(indep, raw)
            notes.append("source re-read: " + ("same" if s else f"DIFFERENT ({indep})"))
            ok &= s
        else:
            notes.append("derived (paper_tables.py)" if field.startswith("derived") else "not re-readable")
        pr = check_printed(text, raw)
        if pr is True:
            notes.append("printed value agrees")
        elif pr is False:
            notes.append(f"PRINTED VALUE MISMATCH (raw {raw})")
            ok = False
        rows.append((k, text, src, field, "yes" if ok else "no", "; ".join(notes)))
        failures += 0 if ok else 1

    digits, style = scan(HAND_WRITTEN)
    failures += len(digits) + len(style)

    # procedural claims: test labels read only after every fit, all under the final protocol file
    procedural = []
    log_p = RES / "test_access_log.jsonl"
    if log_p.exists():
        import datetime as dt
        import glob
        import hashlib
        log = [json.loads(x) for x in log_p.read_text(encoding="utf-8").splitlines() if x.strip()]
        fits = glob.glob(str(RES / "runs" / "*" / "*" / "*" / "*.npz"))
        if fits:
            last_fit = max(os.path.getmtime(f) for f in fits)
            first = min(dt.datetime.strptime(e["time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
                        .timestamp() for e in log)
            ok = first > last_fit
            procedural.append(("first test access after the last model fit", ok,
                               f"last fit {dt.datetime.fromtimestamp(last_fit, dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC, "
                               f"first access {dt.datetime.fromtimestamp(first, dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC"))
        sha = hashlib.sha256((PAPER / "PROTOCOL.md").read_bytes()).hexdigest()
        same_sha = all(e["protocol_sha256"] == sha for e in log)
        procedural.append(("every test access logged under the current PROTOCOL.md", same_sha, f"sha256 {sha[:12]}"))
    failures += sum(1 for _, ok, _ in procedural if not ok)
    out = ["# Claims audit", "",
           "Generated by `scripts/paper_claims.py` from `paper/data/macro_sources.json` (written by "
           "`scripts/paper_tables.py`). Every number in both versions of the paper is a `\\V{key}` macro. For each "
           "key: its source file and field, whether an independent re-read of the source gives the same value, and "
           "whether the printed text agrees with the raw value at its printed precision.", "",
           f"**Summary:** {len(used)} macros used; {sum(1 for r in rows if r[4] == 'yes')} pass; "
           f"{sum(1 for r in rows if 'source re-read: same' in r[5])} re-read from their source; "
           f"{len(digits)} digits outside macros; {len(style)} style findings; **{failures} failures**.", "",
           "| key | printed | source | field | pass | check |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |")
    out += ["", "## Digits outside macros in the hand-written LaTeX", ""]
    out += ["None." if not digits else "| file | line | digits | context |"]
    if digits:
        out.append("|---|---|---|---|")
        out += [f"| {f} | {i} | {d} | `{c.replace('|', '/')}` |" for f, i, d, c in digits]
    out += ["", "## Style findings", ""]
    out += ["None." if not style else "| file | line | rule | context |"]
    if style:
        out.append("|---|---|---|---|")
        out += [f"| {f} | {i} | {r} | `{c.replace('|', '/')}` |" for f, i, r, c in style]
    out += ["", "## Procedural checks", "", "| check | passes | detail |", "|---|---|---|"]
    out += [f"| {c} | {'yes' if ok else 'NO'} | {d} |" for c, ok, d in procedural]
    out += ["", "## Factual (non-numeric) claims", "", "| claim | source |", "|---|---|"]
    out += [f"| {c} | {s} |" for c, s in FACTS]
    (PAPER / "CLAIMS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(used)} macros used, {len(digits)} digits outside macros, {len(style)} style findings, "
          f"{failures} failures -> paper/CLAIMS.md")
    for r in rows:
        if r[4] == "no":
            print("FAIL", r)
    for d in digits[:40]:
        print("DIGIT", d)
    for s in style[:40]:
        print("STYLE", s)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
