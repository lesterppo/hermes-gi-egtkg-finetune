#!/usr/bin/env python3
"""
eval_build.py — build a HELD-OUT evaluation set for the GI/hep EGT-KG adapter.

Why this exists: the daily loop trains on rows derived from an accumulating
knowledge store, so evaluating on that same store would be contaminated. This
script takes a FRESH PubMed ingest and keeps only articles whose PMID is absent
from the training store — the model has never seen them — then builds QA rows
from that held-out slice only.

Usage:
    # 1. fresh ingest into a throwaway copy of the training store
    python3 pubmed_ingest.py --days 60 --store /tmp/eval_store.jsonl \
        --out /tmp/eval_new.jsonl --max-oa 0 --oa-probe 0
    # 2. keep the articles the training store does NOT have, build eval rows
    python3 eval_build.py --train-store /tmp/train_store.jsonl \
        --fresh /tmp/eval_new.jsonl --out /tmp/eval_rows.jsonl --limit 80

Output rows keep the SAME message shape the trainer uses (see egtkg_build) plus
audit fields (pmid/journal/year/kind/e) so scoring can check grounding and
provenance without re-reading the store.
"""
import argparse
import json
import pathlib
import random
import re
import subprocess
import sys


def load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-store", required=True,
                    help="knowledge_store.jsonl the adapter was trained on")
    ap.add_argument("--fresh", required=True,
                    help="jsonl of freshly ingested articles (pubmed_ingest --out)")
    ap.add_argument("--out", required=True, help="eval rows jsonl")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--build-py", default=str(pathlib.Path(__file__).with_name("egtkg_build.py")))
    args = ap.parse_args()

    train_pmids = {str(a.get("pmid")) for a in load_jsonl(args.train_store) if a.get("pmid")}
    fresh = load_jsonl(args.fresh)
    heldout = [a for a in fresh if a.get("pmid") and str(a["pmid"]) not in train_pmids]
    print(f"[eval] training store PMIDs: {len(train_pmids)}")
    print(f"[eval] freshly ingested:    {len(fresh)}")
    print(f"[eval] HELD OUT (unseen):   {len(heldout)}")
    if not heldout:
        print("[eval] nothing held out — widen --days on the ingest", file=sys.stderr)
        return 1

    tmp = pathlib.Path("/tmp/eval_heldout_store.jsonl")
    with open(tmp, "w") as fh:
        for a in heldout:
            fh.write(json.dumps(a) + "\n")

    rows_path = pathlib.Path("/tmp/eval_heldout_rows.jsonl")
    r = subprocess.run([sys.executable, args.build_py, "--store", str(tmp),
                        "--out", str(rows_path), "--seed", str(args.seed),
                        "--limit", str(args.limit * 3)])
    if r.returncode != 0:
        print("[eval] egtkg_build failed", file=sys.stderr)
        return 1
    rows = load_jsonl(rows_path)

    # one row per PMID max, mixed kinds, deterministic order
    by_pmid = {}
    for row in rows:
        pmid = (row.get("pmid") or (row.get("e") or "").split("PMID:")[-1] or "").strip()
        by_pmid.setdefault(pmid or f"x{len(by_pmid)}", []).append(row)
    rng = random.Random(args.seed)
    picked = []
    for pmid, group in by_pmid.items():
        picked.append(rng.choice(group))
    rng.shuffle(picked)
    picked = picked[:args.limit]

    # provenance audit fields for scoring
    meta = {str(a["pmid"]): a for a in heldout}
    with open(args.out, "w") as fh:
        for i, row in enumerate(picked):
            pmid = (row.get("pmid") or "").strip()
            art = meta.get(pmid, {})
            fh.write(json.dumps({
                "id": f"eval-{i:03d}",
                "messages": row["messages"],
                "kind": row.get("kind"),
                "pmid": pmid,
                "journal": art.get("journal"),
                "year": art.get("year"),
                "title": art.get("title"),
                "evidence": row.get("e"),
            }) + "\n")
    kinds = {}
    for row in picked:
        kinds[row.get("kind")] = kinds.get(row.get("kind"), 0) + 1
    print(f"[eval] wrote {len(picked)} held-out rows -> {args.out}  kinds={kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
