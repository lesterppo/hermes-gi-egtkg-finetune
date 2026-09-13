#!/usr/bin/env python3
"""Regression tests for the knowledge-store key handling + merge.

Background (why these exist): the store holds BOTH PubMed records (numeric pmid
keys) and reference corpora (textbook 'SP_...', case reports 'CR_...'). A bare
int(k) sort in pubmed_ingest therefore raised

    ValueError: invalid literal for int() with base 10: 'SP_4405244'

once the textbook corpus landed in the store — every daily ingest died there,
nobody saw it (the runner does not mirror VM stdout), and the loop trained on a
FROZEN corpus for days while the row count sat at ~1490.

Runs offline (no network, no GPU).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubmed_ingest as pi  # noqa: E402
from daily_finetune import merge_stores  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def rec(key, title="t"):
    d = {"pmid": key, "title": title, "abstract": "a", "journal": "J", "year": "2026"}
    return d


def test_sort_key_mixed():
    keys = ["42222916", "SP_4405244", "39123456", "CR_99881"]
    try:
        ordered = sorted(keys, key=pi._sort_key)
    except Exception as e:
        check("mixed keys sort without raising", False, repr(e))
        return
    check("mixed keys sort without raising", True)
    check("pubmed keys numeric-ordered first",
          ordered[:2] == ["39123456", "42222916"], str(ordered))
    check("non-pubmed keys last", ordered[2:] == ["CR_99881", "SP_4405244"], str(ordered))


def test_write_store_with_reference_corpus():
    """The exact crash case: store contains SP_ keys, ingest writes it back."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "store.jsonl")
    store = {str(1000 + i): rec(str(1000 + i)) for i in range(5)}
    store["SP_4405244"] = rec("SP_4405244", "StatPearls chapter")
    store["CR_99881"] = rec("CR_99881", "Case report")
    try:
        pi.write_store(store, path)
        ok = True
    except Exception as e:
        ok = False
        check("write_store handles SP_/CR_ keys", False, repr(e))
    if ok:
        lines = [json.loads(l) for l in open(path) if l.strip()]
        check("write_store handles SP_/CR_ keys", True, f"{len(lines)} lines")
        check("all records preserved", len(lines) == 7, str(len(lines)))
        keys = [str(r["pmid"]) for r in lines]
        check("non-pubmed entries written last",
              keys[-2:] == ["CR_99881", "SP_4405244"], str(keys))


def test_write_store_caps_pubmed_only():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "store.jsonl")
    store = {str(i): rec(str(i)) for i in range(10)}
    store["SP_1"] = rec("SP_1")
    kept = pi.write_store(store, path, max_pub=3)
    check("cap keeps only the newest pubmed keys",
          sorted(k for k in kept if str(k).isdigit()) == ["7", "8", "9"], str(sorted(kept)))
    check("cap keeps the reference corpus regardless", "SP_1" in kept)


def test_merge_stores_dedupes():
    """`cat >>` used to duplicate the whole textbook corpus every run."""
    tmp = tempfile.mkdtemp()
    store_path = os.path.join(tmp, "store.jsonl")
    ref_path = os.path.join(tmp, "ref_store.jsonl")
    with open(store_path, "w") as fh:
        for i in range(3):
            fh.write(json.dumps(rec(str(100 + i))) + "\n")
    with open(ref_path, "w") as fh:
        for i in range(4):
            fh.write(json.dumps(rec(f"SP_{i}")) + "\n")
    n1 = merge_stores(store_path, ref_path, label="textbook")
    n2 = merge_stores(store_path, ref_path, label="textbook")  # second run: no growth
    check("merge adds new keys", n1 == 7, str(n1))
    check("merge is idempotent (no duplicate growth)", n2 == 7, str(n2))
    lines = [l for l in open(store_path) if l.strip()]
    check("store has no duplicate keys", len(lines) == 7, str(len(lines)))


if __name__ == "__main__":
    test_sort_key_mixed()
    test_write_store_with_reference_corpus()
    test_write_store_caps_pubmed_only()
    test_merge_stores_dedupes()
    print()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
