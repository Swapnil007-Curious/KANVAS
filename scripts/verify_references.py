"""Check every entry of paper/references.bib against an authoritative record.

    python scripts/verify_references.py

DOI -> Crossref (api.crossref.org); arXiv eprint -> arXiv API; otherwise the url (proceedings page)
is fetched and the title looked for in it, with a Crossref bibliographic search as a fallback. An
entry passes when the title matches (token overlap >= 0.8) and, where the record has them, the
year and the first author's family name agree. Writes paper/data/reference_verification.json and
exits with code 1 if any entry fails.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "references.bib"
OUT = ROOT / "paper" / "data" / "reference_verification.json"
UA = {"User-Agent": "reference-check/1.0 (bibliography verification script)"}


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", m.group(3), re.S):
            fields[f.group(1).lower()] = " ".join(f.group(2).split())
        entries.append({"type": m.group(1), "key": m.group(2).strip(), **fields})
    return entries


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s or "")
    s = re.sub(r"[{}\\'`\"~^]", "", s)
    return [t for t in re.findall(r"[a-z0-9]+", html.unescape(s).lower()) if len(t) > 2]


def overlap(a, b):
    ta, tb = set(norm(a)), set(norm(b))
    return len(ta & tb) / max(1, len(ta))


def fam_norm(s):
    s = re.sub(r"\[a-zA-Z]+\s*", "", s or "")
    s = re.sub(r"[{}\'`\"~^]", "", s)
    toks = re.findall(r"[a-z]+", html.unescape(s).lower())
    return toks[-1] if toks else ""


def first_family(entry):
    a = entry.get("author", "").split(" and ")[0]
    fam = a.split(",")[0] if "," in a else a.split()[-1]
    return fam_norm(fam)


def get(url, accept=None, tries=5):
    headers = dict(UA)
    if accept:
        headers["Accept"] = accept
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def check_doi(e):
    data = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(e["doi"], safe="/()<>;:.")))["message"]
    title = " ".join(data.get("title", [""]) + data.get("subtitle", []))
    year = (data.get("issued", {}).get("date-parts") or [[None]])[0][0]
    fam = (data.get("author") or [{}])[0].get("family", "")
    return {"source": "crossref", "record_title": title, "record_year": year, "record_first_author": fam,
            "title_overlap": overlap(e["title"], title),
            "year_ok": year is None or abs(int(year) - int(e["year"])) <= 1,
            "author_ok": not fam or fam_norm(fam) == first_family(e)}


def check_arxiv(e):
    try:
        xml = get("http://export.arxiv.org/api/query?id_list=" + e["eprint"])
    except Exception:
        page = get("https://arxiv.org/abs/" + e["eprint"])
        m = re.search(r'<meta name="citation_title" content="([^"]+)"', page)
        a = re.search(r'<meta name="citation_author" content="([^"]+)"', page)
        title = html.unescape(m.group(1)) if m else ""
        fam = a.group(1).split(",")[0] if a else ""
        return {"source": "arxiv abstract page", "record_title": title, "record_year": None,
                "record_first_author": fam, "title_overlap": overlap(e["title"], title), "year_ok": True,
                "author_ok": not fam or fam_norm(fam) == first_family(e)}
    m = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    title = " ".join(m.group(1).split()) if m else ""
    a = re.search(r"<entry>.*?<author>\s*<name>(.*?)</name>", xml, re.S)
    fam = a.group(1).split()[-1] if a else ""
    y = re.search(r"<entry>.*?<published>(\d{4})", xml, re.S)
    return {"source": "arxiv", "record_title": title, "record_year": int(y.group(1)) if y else None,
            "record_first_author": fam, "title_overlap": overlap(e["title"], title),
            "year_ok": True, "author_ok": not fam or fam_norm(fam) == first_family(e)}


def check_url(e):
    try:
        page = get(e["url"])
        text = re.sub(r"<[^>]+>", " ", page)
        ov = overlap(e["title"], " ".join(norm(text)))
        res = {"source": "proceedings page", "url": e["url"], "title_overlap": ov, "year_ok": True,
               "author_ok": first_family(e) in re.findall(r"[a-z]+", html.unescape(text).lower())}
        if ov >= 0.8 and res["author_ok"]:
            return res
    except Exception as exc:  # fall back to a Crossref search
        res = {"source": "proceedings page", "url": e["url"], "error": str(exc)[:120]}
    q = urllib.parse.quote(e["title"])
    works = json.loads(get(f"https://api.openalex.org/works?per-page=5&search={q}"))["results"]
    for w in works:
        title = w.get("title") or ""
        fam = ((w.get("authorships") or [{}])[0].get("author") or {}).get("display_name", "")
        if overlap(e["title"], title) >= 0.8 and fam_norm(fam) == first_family(e)                 and abs(int(w.get("publication_year") or e["year"]) - int(e["year"])) <= 1:
            return {"source": "openalex", "record_title": title, "record_year": w.get("publication_year"),
                    "record_first_author": fam, "record_id": w.get("id"), "record_venue":
                    ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "title_overlap": overlap(e["title"], title), "year_ok": True, "author_ok": True,
                    "page_check": res}
    q = urllib.parse.quote(f"{e['title']} {e.get('author', '').split(' and ')[0]}")
    items = json.loads(get(f"https://api.crossref.org/works?rows=5&query.bibliographic={q}"))["message"]["items"]
    best = max(items, key=lambda it: overlap(e["title"], " ".join(it.get("title", [""]))), default=None)
    if best is None:
        return res
    title = " ".join(best.get("title", [""]))
    year = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
    fam = (best.get("author") or [{}])[0].get("family", "")
    return {"source": "crossref search", "record_title": title, "record_year": year, "record_first_author": fam,
            "record_doi": best.get("DOI"), "title_overlap": overlap(e["title"], title),
            "year_ok": year is None or abs(int(year) - int(e["year"])) <= 1,
            "author_ok": not fam or fam_norm(fam) == first_family(e), "page_check": res}


def main():
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    results, failures = [], 0
    for e in entries:
        try:
            if e.get("doi"):
                r = check_doi(e)
            elif e.get("eprint"):
                r = check_arxiv(e)
            else:
                r = check_url(e)
        except Exception as exc:
            r = {"source": "error", "error": str(exc)[:200], "title_overlap": 0, "year_ok": False, "author_ok": False}
        r["verified"] = bool(r.get("title_overlap", 0) >= 0.8 and r.get("year_ok") and r.get("author_ok"))
        failures += not r["verified"]
        results.append({"key": e["key"], "title": e.get("title"), **r})
        print(f"{'ok  ' if r['verified'] else 'FAIL'} {e['key']:<28s} {r['source']:<17s} overlap "
              f"{r.get('title_overlap', 0):.2f}", flush=True)
        time.sleep(1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"entries": len(entries), "verified": len(entries) - failures, "results": results},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(entries) - failures} of {len(entries)} verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
