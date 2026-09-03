#!/usr/bin/env python3
"""
textbook_ingest.py — static-knowledge source ingest for the EGT-KG GI/hepatology
pipeline. Complements the daily PubMed/EuropePMC literature ingest (pubmed_ingest.py)
with DENSE, STABLE reference knowledge:

  1. StatPearls (NCBI Bookshelf) — 151 GI/hepatology chapters (~25-30k chars each,
     updated annually, keyless HTML). Section headers (Etiology/Evaluation/
     Treatment/Pearls...) make excellent typed-KG extraction targets.
  2. NIDDK digestive + liver disease pages (~56 patient/professional topics,
     keyless HTML, stable URLs).
  3. WGO global guidelines (~30 guideline pages, keyless HTML summaries).

All keyless, no auth, polite rate-limited. Output: reference_store.jsonl
reusing the same schema as the literature store (pubmed_ingest.py) so
egtkg_build.py consumes both unchanged:
  {"pmid": "SP_<uid>", "title": ..., "abstract": <full text>, "journal":
   "StatPearls"|"NIDDK"|"WGO", "year": ..., "mesh": ["reference", <topic>],
   "oa_fulltext": ""}

Dedupes against a literature store when --merge-store is given.
"""
import argparse
import datetime
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
EUTILS_UA = {"User-Agent": "egtkg-textbook/1.0 (research ingest)"}

# StatPearls GI/hep chapter discovery: title-term sweep (db=books has no MeSH
# field; [MeSH Terms] returns 0 there — verified).
SP_TERMS = [
    "gastrointestinal", "gastroenterology", "hepatology", "liver", "colitis",
    "crohn", "cirrhosis", "hepatitis", "pancreatitis", "endoscopy",
    "colonoscopy", "celiac", "dyspepsia", "GERD", "barrett", "esophag",
    "gastric", "biliary", "cholangitis", "ascites", "varices", "steatosis",
    "MASLD", "IBS", "constipation", "diarrhea", "peptic ulcer", "gi bleed",
    "portal hypertension", "hepatocellular", "cholestasis", "proctitis",
    "anal fissure", "hemorrhoid", "malabsorption", "polyp", "achalasia",
    "gastroparesis", "bile duct", "peritonitis", "hepatic encephalopathy",
    "gilbert", "pancreatic", "appendicitis", "diverticulitis", "gastroenteritis",
]

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<script.*?</script>|<style.*?</style>", re.S)
WS_RE = re.compile(r"\s+")


def http_get(url, timeout=45, ua=BROWSER_UA, retries=3):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"GET failed {url[:100]}: {last}")


def html_to_text(html):
    t = SCRIPT_RE.sub(" ", html)
    t = TAG_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


# ---------------------------------------------------------------------------
# StatPearls
# ---------------------------------------------------------------------------

def sp_chapter_uids():
    def e(tool, **p):
        u = EUTILS + tool + "?" + urllib.parse.urlencode(p)
        return urllib.request.urlopen(
            urllib.request.Request(u, headers=EUTILS_UA), timeout=40).read()
    uids = set()
    for t in SP_TERMS:
        try:
            r = json.loads(e("esearch.fcgi", db="books",
                             term=f'"StatPearls"[Book] AND {t}[Title]',
                             retmax="100", retmode="json"))["esearchresult"]
            uids.update(r["idlist"])
        except Exception:
            continue
        time.sleep(0.4)
    return sorted(uids)


def sp_fetch(uids, out, seen):
    """Download StatPearls chapters -> reference_store rows."""
    def e(tool, **p):
        u = EUTILS + tool + "?" + urllib.parse.urlencode(p)
        return urllib.request.urlopen(
            urllib.request.Request(u, headers=EUTILS_UA), timeout=40).read()
    n = 0
    B = 40
    for i in range(0, len(uids), B):
        batch = uids[i:i + B]
        try:
            s = json.loads(e("esummary.fcgi", db="books",
                             id=",".join(batch), retmode="json"))
        except Exception as ex:
            print(f"[textbook] esummary batch {i} failed: {ex}", flush=True)
            continue
        for uid in batch:
            d = s.get("result", {}).get(uid, {})
            title = (d.get("title") or "").strip()
            if not title:
                continue
            year = (d.get("pubdate") or "")[:4]
            try:
                html = http_get(f"https://www.ncbi.nlm.nih.gov/books/{uid}/"
                                if False else
                                f"https://www.ncbi.nlm.nih.gov/books/{uid}/")
            except Exception:
                continue
            text = html_to_text(html)
            # Bookshelf chrome: 'Continuing Education'/'Disclosure' blocks sit
            # BOTH before (sidebar, ~pos 1.7k) and after the real content.
            # Keep the span from 'Introduction'/'Abstract' to 'Disclosure'
            # (falls back to raw text when markers are missing).
            start = min((p for p in (text.find("Introduction"),
                                     text.find("Abstract"),
                                     text.find("Etiology")) if p >= 0),
                        default=-1)
            end = text.find("Disclosure")
            if start >= 0 and end > start:
                text = text[start:end]
            elif start >= 0:
                text = text[start:]
            if len(text) < 2000:
                continue
            row = {
                "pmid": f"SP_{uid}",
                "title": title,
                "abstract": text[:24000],
                "journal": "StatPearls",
                "year": year,
                "doi": "",
                "mesh": ["reference", "statpearls"],
                "oa_fulltext": "",
            }
            key = row["pmid"]
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps(row) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"[textbook] statpearls {n} chapters", flush=True)
            time.sleep(1.2)  # polite: Bookshelf HTML is heavier than E-utilities
    return n


# ---------------------------------------------------------------------------
# NIDDK
# ---------------------------------------------------------------------------

def niddk_fetch(out, seen):
    n = 0
    for section, label in [("digestive-diseases", "niddk-gi"),
                           ("liver-disease", "niddk-liver")]:
        try:
            html = http_get(f"https://www.niddk.nih.gov/health-information/{section}")
        except Exception as e:
            print(f"[textbook] niddk {section} listing failed: {e}", flush=True)
            continue
        links = sorted(set(re.findall(
            rf'href="(/health-information/{section}/[a-z0-9\-]+)"', html)))
        for path in links:
            key = f"NIDDK_{path.rsplit('/', 1)[-1]}"
            if key in seen:
                continue
            try:
                h2 = http_get("https://www.niddk.nih.gov" + path)
            except Exception:
                continue
            text = html_to_text(h2)
            # trim site chrome
            i = text.find("On this page")
            if i > 0:
                text = text[i:]
            if len(text) < 1500:
                continue
            title = path.rsplit("/", 1)[-1].replace("-", " ").title()
            out.write(json.dumps({
                "pmid": key,
                "title": f"{title} (NIDDK)",
                "abstract": text[:16000],
                "journal": "NIDDK",
                "year": str(datetime.date.today().year),
                "doi": "",
                "mesh": ["reference", label],
                "oa_fulltext": "",
            }) + "\n")
            seen.add(key)
            n += 1
            time.sleep(1.0)
    return n


# ---------------------------------------------------------------------------
# WGO guidelines
# ---------------------------------------------------------------------------

def wgo_fetch(out, seen):
    n = 0
    try:
        html = http_get("https://www.worldgastroenterology.org/guidelines")
    except Exception as e:
        print(f"[textbook] wgo listing failed: {e}", flush=True)
        return 0
    paths = sorted(set(re.findall(r'href="(/guidelines/global-guidelines/[^"]+)"', html)))
    for path in paths:
        slug = path.rsplit("/", 1)[-1]
        key = f"WGO_{slug}"
        if key in seen:
            continue
        try:
            h2 = http_get("https://www.worldgastroenterology.org" + path)
        except Exception:
            continue
        text = html_to_text(h2)
        if len(text) < 1200:
            continue
        title = slug.replace("-", " ").title()
        out.write(json.dumps({
            "pmid": key,
            "title": f"{title} (WGO Global Guideline)",
            "abstract": text[:16000],
            "journal": "WGO",
            "year": str(datetime.date.today().year),
            "doi": "",
            "mesh": ["reference", "wgo-guideline"],
            "oa_fulltext": "",
        }) + "\n")
        seen.add(key)
        n += 1
        time.sleep(1.0)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="reference_store.jsonl to write")
    ap.add_argument("--merge-store", default=None,
                    help="existing knowledge_store.jsonl — skip keys already present")
    ap.add_argument("--sources", default="statpearls,niddk,wgo")
    ap.add_argument("--max-statpearls", type=int, default=0, help="cap chapters (0=all)")
    args = ap.parse_args()

    seen = set()
    merge_path = pathlib.Path(args.merge_store) if args.merge_store else None
    if merge_path and merge_path.exists():
        for line in merge_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["pmid"])
            except Exception:
                continue
    print(f"[textbook] merge-store preloaded: {len(seen)} keys", flush=True)

    srcs = set(args.sources.split(","))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        if "statpearls" in srcs:
            uids = sp_chapter_uids()
            if args.max_statpearls:
                uids = uids[:args.max_statpearls]
            print(f"[textbook] StatPearls chapters discovered: {len(uids)}", flush=True)
            n = sp_fetch(uids, f, seen)
            print(f"[textbook] StatPearls harvested: {n}", flush=True)
        if "niddk" in srcs:
            n = niddk_fetch(f, seen)
            print(f"[textbook] NIDDK topics: {n}", flush=True)
        if "wgo" in srcs:
            n = wgo_fetch(f, seen)
            print(f"[textbook] WGO guidelines: {n}", flush=True)
    print(f"[textbook] DONE -> {out} ({len(seen)} total keys)", flush=True)


if __name__ == "__main__":
    main()
