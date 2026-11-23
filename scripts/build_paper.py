"""Build both versions of the paper with tectonic and check the logs and the PDFs.

    python scripts/build_paper.py            # full version and IEEE version
    python scripts/build_paper.py --only ieee

For each version: compile with tectonic (which reruns BibTeX and LaTeX as needed), then fail on any
overfull box, undefined reference or citation, or undefined number macro; rewrite the PDF metadata so it
holds the title and the author only; and report the page count. Tectonic lives in
C:/Users/Admin/.local/tectonic/ unless TECTONIC points elsewhere.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TECTONIC = os.environ.get("TECTONIC", r"C:\Users\Admin\.local\tectonic\tectonic.exe")
TITLE = ("KANVAS: Auditing Shortcut Learning in GPU Job Failure Prediction with Additive Kolmogorov-Arnold "
         "Models")
AUTHOR = "Swapnil Shukla"
VERSIONS = {"full": (PAPER, "main.tex"), "ieee": (PAPER / "ieee", "main_ieee.tex")}


def title_from_tex(tex):
    m = re.search(r"pdftitle=\{([^}]*)\}", tex.read_text(encoding="utf-8"))
    return m.group(1) if m else TITLE


def build(name):
    cwd, main = VERSIONS[name]
    if name == "ieee":
        bib = (PAPER / "references.bib").read_text(encoding="utf-8")
        # IEEE style prints no url for entries that have a DOI or venue; keep the bib identical otherwise
        (cwd / "references_ieee.bib").write_text(re.sub(r"(?m)^\s*url\s*=\s*\{[^}]*\},?\s*\n", "", bib),
                                                 encoding="utf-8")
    cmd = [TECTONIC, "-X", "compile", main, "--keep-logs", "--keep-intermediates"]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path = cwd / main.replace(".tex", ".log")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    problems = []
    if r.returncode != 0:
        problems.append("tectonic failed:\n" + (r.stderr or r.stdout)[-3000:])
    for pat, what in [(r"Overfull \\[hv]box[^\n]*", "overfull box"),
                      (r"LaTeX Warning: Reference [^\n]*undefined[^\n]*", "undefined reference"),
                      (r"LaTeX Warning: Citation [^\n]*undefined[^\n]*", "undefined citation"),
                      (r"There were undefined references", "undefined references"),
                      (r"Package numbers Error[^\n]*", "undefined number macro"),
                      (r"LaTeX Warning: Label [^\n]*multiply defined[^\n]*", "duplicate label"),
                      (r"Missing character: [^\n]*", "missing glyph")]:
        for m in re.finditer(pat, log):
            problems.append(f"{what}: {m.group(0)[:160]}")
    blg = cwd / main.replace(".tex", ".blg")
    if blg.exists():
        for m in re.finditer(r"Warning--[^\n]*", blg.read_text(encoding="utf-8", errors="replace")):
            problems.append(f"bibtex: {m.group(0)[:160]}")
    pdf = cwd / main.replace(".tex", ".pdf")
    pages = None
    if pdf.exists():
        pages = clean_metadata(pdf, title_from_tex(cwd / main))
    return pages, problems


def clean_metadata(pdf, title):
    """Keep only /Title and /Author in the document information; drop XMP metadata. Returns the page count."""
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, TextStringObject

    reader = PdfReader(str(pdf))
    writer = PdfWriter(clone_from=reader)
    info = DictionaryObject({NameObject("/Title"): TextStringObject(title),
                             NameObject("/Author"): TextStringObject(AUTHOR)})
    writer._info = writer._add_object(info)  # replaces the Producer/Creator/dates that tectonic and pypdf add
    root = writer._root_object
    if "/Metadata" in root:
        del root["/Metadata"]
    tmp = pdf.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as fh:
        writer.write(fh)
    shutil.move(tmp, pdf)
    check = PdfReader(str(pdf))
    keys = sorted(check.metadata.keys()) if check.metadata else []
    if keys != ["/Author", "/Title"]:
        raise SystemExit(f"{pdf.name}: metadata keys {keys}")
    return len(check.pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(VERSIONS))
    args = ap.parse_args()
    failed = False
    for name in ([args.only] if args.only else list(VERSIONS)):
        pages, problems = build(name)
        print(f"{name}: {pages} pages, {len(problems)} problems")
        for p in problems:
            print("  ", p)
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
