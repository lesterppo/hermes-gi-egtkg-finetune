#!/usr/bin/env python3
"""
ab_score.py — score the base-vs-adapter A/B output (offline, deterministic).

Input:  ab_results.jsonl from eval_ab.py (each row has adapter_out + base_out)
Output: JSON summary + markdown table on stdout

Metrics per answer (no LLM needed, so the numbers are reproducible):
  quote       token recall of the reference evidence span inside the answer
              (the EGT-KG promise: the answer IS the verbatim evidence)
  prov        answer carries provenance (Source:/PMID: + journal string)
  fmt         answer follows the trained shape ("The article states:"/"states:")
  extra       fraction of answer content words NOT in (evidence span + prompt)
              -> proxy for invention/hallucination
  trunc       answer looks cut off (no terminal punctuation)
"""
import argparse
import json
import re
import sys

STOP = set("""a an the of in on for to and or with without by from as at is are was were be been being
this that these those it its their there here we our they them he she his her not no than then thus
using only retrieved medical article evidence answer what does report states state results result
according literature how relate relation between source reply precisely exact stated""".split())


def words(s):
    return [w for w in re.findall(r"[a-z0-9%.\-]+", (s or "").lower()) if w not in STOP and len(w) > 1]


def ref_span(row):
    """The evidence span the trained answer quotes.

    Falls back to row["evidence"]: the scored/summary files are post-processed
    and no longer carry `messages`, so a messages-only lookup returns "" — and an
    empty ground truth silently turns every downstream judgement into noise.
    """
    a = ""
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            a = m.get("content", "")
            break
    a = re.sub(r"\s*\(Source:.*?\)\s*$", "", a).strip()
    for pat in (r"^The article states:\s*", r"^The evidence\s*\([^)]*\)\s*states:\s*",
                r"^.*?\bstates:\s*"):
        m = re.match(pat, a, re.S)
        if m:
            return a[m.end():].strip()
    return a or (row.get("evidence") or "").strip()


def contains_provenance(ans, row):
    if re.search(r"\(Source:|PMID:?\s*\d{6,}", ans or ""):
        return 1
    j = (row.get("journal") or "").strip()
    if j and len(j) > 6 and j.lower()[:20] in (ans or "").lower():
        return 1
    return 0


def score_one(ans, row, evidence):
    ans = ans or ""
    ev_w = words(evidence)
    an_w = words(ans)
    if not an_w:
        return {"quote": 0.0, "prov": 0, "fmt": 0, "extra": 1.0, "trunc": 1, "len": 0, "empty": 1}
    ev_set = set(ev_w)
    hit = sum(1 for w in ev_w if w in set(an_w))
    quote = hit / max(1, len(ev_w))
    allowed = ev_set | set(words(" ".join(m.get("content", "") for m in row.get("messages", []))))
    extra = sum(1 for w in an_w if w not in allowed) / max(1, len(an_w))
    fmt = 1 if re.search(r"\bstates:|\bThe article states:|\bThe evidence\b", ans) else 0
    trunc = 1 if ans.strip() and ans.strip()[-1] not in ".!?)\"'" else 0
    return {"quote": round(quote, 3), "prov": contains_provenance(ans, row), "fmt": fmt,
            "extra": round(extra, 3), "trunc": trunc, "len": len(ans), "empty": 0}


def agg(rows, key):
    n = len(rows)
    if not n:
        return {}
    def mean(f):
        return round(sum(f(r[key]) for r in rows) / n, 3)
    return {"n": n, "quote": mean(lambda s: s["quote"]), "prov": mean(lambda s: s["prov"]),
            "fmt": mean(lambda s: s["fmt"]), "extra": mean(lambda s: s["extra"]),
            "trunc": mean(lambda s: s["trunc"]), "empty": mean(lambda s: s["empty"]),
            "len": int(mean(lambda s: s["len"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None, help="write the per-row scored jsonl here")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    raw = [json.loads(l) for l in open(args.results) if l.strip()]
    scored = []
    for r in raw:
        ev = ref_span(r)
        s = {"id": r.get("id"), "kind": r.get("kind"), "evidence": ev,
             "adapter": score_one(r.get("adapter_out"), r, ev),
             "base": score_one(r.get("base_out"), r, ev),
             "adapter_out": r.get("adapter_out"), "base_out": r.get("base_out"),
             "t_adapter_s": r.get("t_adapter_s"), "t_base_s": r.get("t_base_s")}
        scored.append(s)

    a = agg(scored, "adapter")
    b = agg(scored, "base")
    wins = sum(1 for s in scored if s["adapter"]["quote"] > s["base"]["quote"])
    losses = sum(1 for s in scored if s["adapter"]["quote"] < s["base"]["quote"])
    ties = len(scored) - wins - losses
    prov_a = sum(1 for s in scored if s["adapter"]["prov"])
    prov_b = sum(1 for s in scored if s["base"]["prov"])

    print("| metric | adapter ON | base (LoRA off) | delta |")
    print("|---|---|---|---|")
    for k, label in [("quote", "verbatim evidence recall"), ("prov", "carries provenance"),
                     ("fmt", "trained answer shape"), ("extra", "out-of-evidence words"),
                     ("trunc", "truncated answers"), ("empty", "empty answers"),
                     ("len", "mean answer chars")]:
        print(f"| {label} | {a[k]} | {b[k]} | {round(a[k] - b[k], 3):+} |")
    print()
    print(f"rows={len(scored)}  evidence-recall wins: adapter {wins} / base {losses} / tie {ties}")
    print(f"provenance: adapter {prov_a}/{len(scored)}  base {prov_b}/{len(scored)}")

    if args.examples:
        print("\n--- examples (adapter vs base) ---")
        for s in scored[:args.examples]:
            print(f"\n[{s['id']} / {s['kind']}] evidence: {s['evidence'][:150]}")
            print(f"  ADAPTER: {(s['adapter_out'] or '')[:220]}")
            print(f"  BASE   : {(s['base_out'] or '')[:220]}")

    summary = {"n": len(scored), "adapter": a, "base": b,
               "quote_wins": {"adapter": wins, "base": losses, "tie": ties},
               "provenance": {"adapter": prov_a, "base": prov_b}}
    if args.out:
        with open(args.out, "w") as fh:
            for s in scored:
                fh.write(json.dumps(s) + "\n")
        with open(args.out.replace(".jsonl", "_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n[score] wrote {args.out} + summary json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
