#!/usr/bin/env python3
"""
pubmed_ingest.py — daily GI/hepatology literature ingest for the EGT-KG pipeline.

Runner-side stage (CPU-only, pure stdlib, no API keys):
  1. PubMed esearch: GI/liver MeSH + gastroenterology/hepatology terms, window
     = last N days (edat). Keyless E-utilities (3 req/s politeness delay).
  2. PubMed efetch: titles + abstracts + MeSH + DOI per PMID (batches of 100).
  3. Europe PMC: OA full texts (JATS XML) for OA articles -> richer evidence.
  4. Merge into the persistent knowledge store (dedupe by PMID; cap size).

Usage:
    python pubmed_ingest.py --days 2 --store knowledge_store.jsonl --out new_articles.jsonl
"""
import argparse
import datetime
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
POLITE_S = 0.45  # keyless NCBI limit is 3 req/s; stay well under

QUERY = (
    '("Digestive System Diseases"[Mesh] OR "Liver Diseases"[Mesh] '
    'OR gastroenterolog*[tiab] OR hepatolog*[tiab] '
    'OR "gastroenterology"[Journal] OR "hepatology"[Journal] '
    'OR "Am J Gastroenterol"[Journal] OR "Gut"[Journal] '
    'OR "Aliment Pharmacol Ther"[Journal] OR "J Hepatol"[Journal] '
    'OR "Hepatology"[Journal] OR "Clin Gastroenterol Hepatol"[Journal] '
    'OR "Gastrointest Endosc"[Journal] OR "Inflamm Bowel Dis"[Journal])'
)

MAX_STORE = 6000  # trim oldest beyond this


def http_get(url, timeout=45, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "egtkg-gi/1.0 (research ingest)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # transient NCBI/EPMC hiccups
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url[:120]}: {last}")


def load_store(path):
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    store = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            a = json.loads(line)
            store[a["pmid"]] = a
        except Exception:
            continue
    return store


def esearch_pmids(days):
    import datetime
    now = datetime.date.today()
    start = (now - datetime.timedelta(days=int(days))).isoformat()
    end = now.isoformat()
    # NCBI `N:days[edat]` returned 0 on this backend (dateline lag); explicit
    # date-range works and is robust to their indexer.
    dateq = f"{start}:{end}[edat]"
    params = {
        "db": "pubmed",
        "term": f"{QUERY} AND ({dateq})",
        "retmode": "json",
        # newest-2000 slice of the window (date-sorted). OA/PMC articles
        # typically carry June-July epub dates while the newest 400 are
        # in-press Sept items — a 400 cap left the OA cohort permanently
        # outside the fetch head (0 OA hits in the smoke run).
        "retmax": "2000",
        "sort": "date",
    }
    url = EUTILS + "esearch.fcgi?" + urllib.parse.urlencode(params)
    data = json.loads(http_get(url))
    ids = data.get("esearchresult", {}).get("idlist", [])
    return [i for i in ids if i.strip()]


def _txt(node):
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip() if node is not None else ""


def efetch_articles(pmids):
    """Return list of article dicts from PubMed efetch XML."""
    out = []
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i + 100]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }
        url = EUTILS + "efetch.fcgi?" + urllib.parse.urlencode(params)
        try:
            root = ET.fromstring(http_get(url))
        except Exception as e:
            print(f"[ingest] efetch batch {i} failed: {e}", flush=True)
            continue
        for art in root.findall(".//PubmedArticle"):
            pmid = _txt(art.find(".//MedlineCitation/PMID"))
            if not pmid:
                continue
            title = _txt(art.find(".//Article/ArticleTitle"))
            abst_parts = []
            for ab in art.findall(".//Abstract/AbstractText"):
                label = ab.get("Label")
                t = _txt(ab)
                if t:
                    abst_parts.append(f"{label}: {t}" if label else t)
            abstract = "\n".join(abst_parts)
            journal = _txt(art.find(".//Journal/Title"))
            year = _txt(art.find(".//JournalIssue/PubDate/Year")) or _txt(art.find(".//JournalIssue/PubDate/MedlineDate"))
            doi = ""
            for aid in art.findall(".//ArticleIdList/ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = _txt(aid)
                    break
            mesh = [_txt(m.find("DescriptorName")) for m in art.findall(".//MeshHeadingList/MeshHeading")]
            mesh = [m for m in mesh if m]
            if not abstract:
                continue
            out.append({
                "pmid": pmid, "title": title, "abstract": abstract,
                "journal": journal, "year": year, "doi": doi, "mesh": mesh,
                "oa_fulltext": "",
            })
        time.sleep(POLITE_S)
    return out


def epmc_oa_fulltext(article):
    """Fetch OA full text (JATS XML body) from Europe PMC when available."""
    try:
        q = urllib.parse.quote(f'EXT_ID:{article["pmid"]} AND SRC:MED')
        url = EPMC + f"search?query={q}&resultType=core&format=json"
        data = json.loads(http_get(url))
        hits = data.get("resultList", {}).get("result", [])
        if not hits:
            return ""
        hit = hits[0]
        if str(hit.get("isOpenAccess", "N")).lower() != "y":
            return ""
        pmcid = hit.get("pmcid", "")
        if not pmcid:
            return ""
        xml = http_get(EPMC + f"{pmcid}/fullTextXML", timeout=60).decode("utf-8", "ignore")
        root = ET.fromstring(xml)
        paras = []
        for sec in root.findall(".//body//sec"):
            head = _txt(sec.find("title"))
            for p in sec.findall("p"):
                t = _txt(p)
                if len(t) > 120:
                    paras.append((f"{head}: {t}" if head and not t.startswith(head) else t))
        if not paras:
            for p in root.findall(".//body//p"):
                t = _txt(p)
                if len(t) > 120:
                    paras.append(t)
        # cap: keep evidence-dense body text under ~18k chars per article
        text = "\n".join(paras)[:18000]
        return text
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2, help="ingest window in days (edat)")
    ap.add_argument("--store", required=True, help="persistent knowledge store jsonl (read+write)")
    ap.add_argument("--out", required=True, help="where to write this run's new articles jsonl")
    ap.add_argument("--max-oa", type=int, default=16, help="max OA full texts to pull per run")
    ap.add_argument("--oa-probe", type=int, default=60,
                    help="probe at most this many NEW articles for EPMC OA before giving up")
    ap.add_argument("--backfill-days", type=int, default=120)
    args = ap.parse_args()

    store = load_store(args.store)
    days = args.days
    if not store:
        days = max(days, args.backfill_days)
        print(f"[ingest] empty store — backfilling {days} days", flush=True)

    pmids = esearch_pmids(days)
    print(f"[ingest] esearch: {len(pmids)} PMIDs in last {days}d", flush=True)
    articles = efetch_articles(pmids)
    print(f"[ingest] efetch: {len(articles)} articles with abstracts", flush=True)

    new = [a for a in articles if a["pmid"] not in store]
    print(f"[ingest] {len(new)} new (not in store of {len(store)})", flush=True)

    oa_done = 0
    # OA-targeted probe: instead of blind-sampling the newest N (mostly
    # non-OA, 0/40 hit in the smoke run), query Europe PMC once for OA articles
    # with full text in the window. NOTE: EPMC needs ITS OWN query syntax
    # (MESH:"...", FIRST_PDATE range, HAS_FT) — PubMed [Journal]/[tiab] tags
    # return 0 there. PMC full-text indexing lags PubMed by ~2 months, so the
    # lookback is 90d regardless of the PubMed ingest window.
    oa_pmids = set()
    try:
        start = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
        end = datetime.date.today().isoformat()
        oq = ('(MESH:"Digestive System Diseases" OR MESH:"Liver Diseases") AND '
              f'(OPEN_ACCESS:y OR IN_EPMC:y) AND HAS_FT:y AND SRC:MED AND '
              f'FIRST_PDATE:[{start} TO {end}]')
        ourl = EPMC + "search?" + urllib.parse.urlencode(
            {"query": oq, "format": "json", "pageSize": str(args.max_oa * 3)})
        odata = json.loads(http_get(ourl, timeout=60))
        for h in odata.get("resultList", {}).get("result", []):
            if h.get("pmid"):
                oa_pmids.add(str(h["pmid"]))
        print(f"[ingest] EPMC OA-targeted (90d): {len(oa_pmids)} OA full-text articles", flush=True)
    except Exception as e:
        print(f"[ingest] OA targeted query failed (fallback to blind probe): {e}", flush=True)
    # order: OA-first, then the rest (still bounded by oa_probe)
    ordered = ([a for a in new if a["pmid"] in oa_pmids]
               + [a for a in new if a["pmid"] not in oa_pmids])
    for a in ordered[:max(args.max_oa * 3, args.oa_probe)]:
        if oa_done >= args.max_oa:
            break
        a["oa_fulltext"] = epmc_oa_fulltext(a)
        if a["oa_fulltext"]:
            oa_done += 1
            if oa_done % 5 == 0:
                print(f"[ingest] OA progress {oa_done}", flush=True)
        time.sleep(0.3)
    # Merge OA articles that the PubMed date-sorted head MISSED (OA/PMC
    # cohort carries June-July epub dates; the newest-2000 slice of a
    # 15430-hit window never reaches them — verified live). Their metadata
    # comes from the EPMC core hit; full text is fetched below.
    merged_oa = 0
    have = {a["pmid"] for a in new}
    if oa_pmids - have:
        try:
            oq2 = ('(MESH:"Digestive System Diseases" OR MESH:"Liver Diseases") AND '
                   '(OPEN_ACCESS:y OR IN_EPMC:y) AND HAS_FT:y AND SRC:MED AND '
                   f'FIRST_PDATE:[{(datetime.date.today() - datetime.timedelta(days=90)).isoformat()} TO {datetime.date.today().isoformat()}]')
            ourl2 = EPMC + "search?" + urllib.parse.urlencode(
                {"query": oq2, "format": "json", "pageSize": str(args.max_oa * 3),
                 "resultType": "core"})
            odata2 = json.loads(http_get(ourl2, timeout=60))
            for h in odata2.get("resultList", {}).get("result", []):
                p = str(h.get("pmid") or "")
                if not p or p in have or p in store:
                    continue
                if len(new) - len(have) >= args.max_oa:  # bound the merge
                    break
                abst = re.sub(r"\s+", " ", h.get("abstractText") or "").strip()
                a = {
                    "pmid": p,
                    "title": re.sub(r"<[^>]+>", "", h.get("title") or "").strip(),
                    "abstract": abst,
                    "journal": (h.get("journalInfo", {}) or {}).get("journal", {}).get("title", ""),
                    "year": (h.get("pubYear") or ""),
                    "doi": h.get("doi") or "",
                    "mesh": [m.get("descriptorName", "") for m in (h.get("meshHeadingList", {}) or {}).get("meshHeading", [])],
                    "oa_fulltext": "",
                }
                if not a["abstract"]:
                    continue
                new.append(a)
                have.add(p)
                merged_oa += 1
        except Exception as e:
            print(f"[ingest] OA merge failed (non-fatal): {e}", flush=True)
        # now fetch full texts for the merged OA cohort
        for a in [x for x in new if x["pmid"] in oa_pmids and not x["oa_fulltext"]]:
            if oa_done >= args.max_oa + merged_oa:
                break
            a["oa_fulltext"] = epmc_oa_fulltext(a)
            if a["oa_fulltext"]:
                oa_done += 1
            time.sleep(0.3)
    print(f"[ingest] Europe PMC OA full texts: {oa_done} (merged {merged_oa} OA-only articles)", flush=True)

    for a in new:
        store[a["pmid"]] = a

    # cap store size (drop oldest by pmid order == oldest numeric)
    if len(store) > MAX_STORE:
        keep = sorted(store.keys(), key=lambda x: int(x))[-MAX_STORE:]
        store = {k: store[k] for k in keep}

    pathlib.Path(args.store).write_text(
        "\n".join(json.dumps(store[k]) for k in sorted(store.keys(), key=lambda x: int(x)))
    )
    pathlib.Path(args.out).write_text("\n".join(json.dumps(a) for a in new))

    stats = {
        "window_days": days, "searched": len(pmids), "with_abstract": len(articles),
        "new": len(new), "oa_fulltexts": oa_done, "store_size": len(store),
    }
    print(f"[ingest] STATS {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
