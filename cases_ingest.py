#!/usr/bin/env python3
"""
cases_ingest.py — open-access GI/hepatology CASE REPORTS (PMC, CC-licensed)
for the EGT-KG pipeline.

Why: case reports carry the clinical-reasoning chain (presentation -> workup ->
diagnosis -> management) that abstracts compress away — the densest legitimate
training signal after textbooks. Only OPEN-ACCESS (CC-BY/CC-BY-SA) articles
from PubMed Central are used; no paywalled or copyrighted-only content.

Output schema matches pubmed_ingest/textbook_ingest stores (pmid/title/
abstract/journal/year/mesh/oa_fulltext) so egtkg_build.py consumes it
unchanged. Keys are CR_<pmid>.

Usage:
    python cases_ingest.py --store /content/cr_store.jsonl --days 21 --max 40
Rebuilds the file from the window each run (VM disk is ephemeral; dedupe
against the literature store via --merge-store when persisting longer).
"""
import argparse
import datetime
import json
import pathlib
import re
import time
import urllib.parse

from pubmed_ingest import EPMC, http_get, epmc_oa_fulltext  # reuse verified helpers

EPMC_QUERY = (
    '((TITLE:"gastroenterolog*" OR TITLE:"hepatolog*" OR TITLE:liver OR '
    'TITLE:endoscop* OR TITLE:colon* OR TITLE:gastric OR TITLE:"inflammatory bowel" '
    'OR TITLE:pancrea* OR TITLE:esophag* OR TITLE:hepatic OR TITLE:biliary OR '
    'TITLE:cholestasis OR TITLE:cirrhosis OR TITLE:hepatitis OR TITLE:celiac OR '
    'TITLE:"peptic ulcer" OR TITLE:varices OR TITLE:ascites) AND "case report") AND '
    '(OPEN_ACCESS:y OR IN_EPMC:y) AND HAS_FT:y AND SRC:MED AND '
    'FIRST_PDATE:[{start} TO {end}]'
)


def load_seen(path):
    seen = set()
    p = pathlib.Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["pmid"])
            except Exception:
                continue
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="case-report store jsonl (rewritten each run)")
    ap.add_argument("--merge-store", default=None,
                    help="knowledge_store.jsonl — skip keys already present there")
    ap.add_argument("--days", type=int, default=180,
                    help="case-report window (PMC fulltext indexing lags ~2mo; "
                         "180d keeps the pipeline supplied)")
    ap.add_argument("--max", type=int, default=40, help="max case reports per run")
    args = ap.parse_args()

    seen = load_seen(args.store)
    if args.merge_store:
        seen |= load_seen(args.merge_store)
    pre = len(seen)

    start = (datetime.date.today() - datetime.timedelta(days=args.days)).isoformat()
    end = datetime.date.today().isoformat()
    q = EPMC_QUERY.format(start=start, end=end)
    url = EPMC + "search?" + urllib.parse.urlencode(
        {"query": q, "format": "json", "pageSize": str(args.max * 2),
         "resultType": "core", "sort": "FIRST_PDATE_D desc"})
    try:
        data = json.loads(http_get(url, timeout=60))
    except Exception as e:
        print(f"[cases] EPMC query failed: {e}", flush=True)
        return
    hits = data.get("resultList", {}).get("result", [])
    print(f"[cases] {data.get('hitCount', len(hits))} OA case reports in "
          f"{args.days}d window; fetching up to {args.max}", flush=True)

    rows = []
    fetched = 0
    for h in hits:
        if fetched >= args.max:
            break
        pmid = str(h.get("pmid") or "")
        if not pmid or f"CR_{pmid}" in seen:
            continue
        abst = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h.get("abstractText") or "")).strip()
        title = re.sub(r"<[^>]+>", "", h.get("title") or "").strip()
        if len(abst) < 200:
            continue
        art = {"pmid": pmid}
        ft = epmc_oa_fulltext(art)
        row = {
            "pmid": f"CR_{pmid}",
            "title": f"{title} (case report)",
            "abstract": abst,
            "journal": ((h.get("journalInfo") or {}).get("journal") or {}).get("title", ""),
            "year": h.get("pubYear") or "",
            "doi": h.get("doi") or "",
            "mesh": ["case-report"],
            "oa_fulltext": ft,
        }
        rows.append(row)
        seen.add(f"CR_{pmid}")
        if ft:
            fetched += 1
        time.sleep(0.4)

    pathlib.Path(args.store).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.store).write_text("\n".join(json.dumps(r) for r in rows))
    with_ft = sum(1 for r in rows if r["oa_fulltext"])
    print(f"[cases] STATS " + json.dumps({
        "window_days": args.days, "new": len(rows), "with_fulltext": with_ft,
        "store_keys_seen": len(seen) - pre + len(rows)}), flush=True)


if __name__ == "__main__":
    main()
