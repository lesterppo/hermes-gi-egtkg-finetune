#!/usr/bin/env python3
"""
judge_ab.py — blinded third-party scoring of the base-vs-adapter A/B sample.

Sends each comparison to Gemini Pro (webapi via gemini.py) with the answers
labelled A/B in a randomized order, so the judge cannot know which side is the
fine-tune. Scores 1-5 on: grounded (no invented content), provenance, clinical
usefulness. Results are mapped back to adapter/base locally.

Usage:
    python3 judge_ab.py --scored /tmp/abeval/scored.jsonl --out /tmp/abeval/judge.jsonl \
        --n 14 --per-kind kg=7,evidence=4,fact=3
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys

GEMINI = os.path.expanduser("~/gemini-cli/gemini.py")

PROMPT = """You are a strict clinical NLP reviewer. Below is a QUESTION, the
GROUND-TRUTH EVIDENCE it must be answered from, and TWO candidate answers (A and
B) produced by different systems.

Score each answer 1-5 on:
- grounded: is every claim supported by the ground-truth evidence? (5 = all
  claims traced to the evidence, 1 = invented/unsupported content, wrong study)
- provenance: does it attribute the evidence (source/PMID/journal)? (5 = explicit
  and correct, 1 = none)
- useful: would a gastroenterologist find this answer usable as written?

Ignore style and length. Penalise hallucination hard.

QUESTION:
{q}

GROUND-TRUTH CONTEXT (everything the answering system was given — treat any
source metadata appearing here as legitimate, NOT as hallucination):
{ev}

ANSWER A:
{a}

ANSWER B:
{b}

Reply with ONLY a JSON object, no prose, no markdown fence:
{{"A": {{"grounded": n, "provenance": n, "useful": n, "note": "<=15 words"}},
  "B": {{"grounded": n, "provenance": n, "useful": n, "note": "<=15 words"}},
  "pick": "A" or "B"}}"""


def call_gemini(prompt, timeout=240):
    """gemini.py --raw prints a POINTER json ({"ok":true,"f":"<path>","s":N}); the
    model text lives in that file, so read it before parsing anything."""
    r = subprocess.run([sys.executable, GEMINI, "-m", "pro", "--raw", "-t", str(timeout), "-p", prompt],
                       capture_output=True, text=True, timeout=timeout + 60)
    raw = (r.stdout or "").strip()
    out = raw
    try:
        ptr = json.loads(raw)
        if isinstance(ptr, dict) and ptr.get("f") and os.path.exists(ptr["f"]):
            out = open(ptr["f"]).read()
    except Exception:
        pass

    def norm(obj):
        """Accept {A,B} / {a,b} / {scores:{A,B}} / {answers:[{label:..}...]}."""
        if not isinstance(obj, dict):
            return None
        if "A" in obj and "B" in obj:
            return obj
        low = {str(k).lower(): v for k, v in obj.items()}
        if "a" in low and "b" in low:
            return {"A": low["a"], "B": low["b"], "pick": str(low.get("pick", "")).upper()}
        for v in obj.values():
            if isinstance(v, dict) and ({"A", "B"} <= set(v)):
                return v
            if isinstance(v, list):
                got = {}
                for it in v:
                    if isinstance(it, dict):
                        lbl = str(it.get("label") or it.get("answer") or "").strip().upper()[:1]
                        if lbl in ("A", "B"):
                            got[lbl] = it
                if {"A", "B"} <= set(got):
                    return got
        return None

    # scan every {...} block from the END: the verdict is the model's final object
    for m in reversed(list(re.finditer(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", out, re.S)) +
                      list(re.finditer(r"\{.*\}", out, re.S))):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        good = norm(obj)
        if good:
            return good, None
    return None, out[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="ab_results.jsonl from eval_ab.py (has the prompts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--per-kind", default="kg=7,evidence=4,fact=3")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    from ab_score import ref_span
    rows = [json.loads(l) for l in open(args.results) if l.strip()]
    for r in rows:
        r["evidence"] = ref_span(r)
        # The judge must see the SAME material the model saw. The user prompt
        # carries the source line (title/journal/year/PMID) plus the span; the
        # bare span does not, so judging against the span alone marks a correct
        # citation copied from the prompt as a fabricated one.
        q = ""
        for m in r.get("messages", []):
            if m.get("role") == "user":
                q = m.get("content", "")
                break
        r["context"] = q
    want = {}
    for part in args.per_kind.split(","):
        k, v = part.split("=")
        want[k.strip()] = int(v)
    rng = random.Random(args.seed)
    picked = []
    for kind, n in want.items():
        pool = [r for r in rows if r.get("kind") == kind]
        rng.shuffle(pool)
        picked.extend(pool[:n])
    picked = picked[:args.n]
    print(f"[judge] {len(picked)} comparisons -> Gemini Pro (blinded A/B)", flush=True)

    results = []
    skipped = 0
    for i, row in enumerate(picked):
        flip = rng.random() < 0.5
        a = row["base_out"] if flip else row["adapter_out"]
        b = row["adapter_out"] if flip else row["base_out"]
        q = ""
        for m in row.get("messages", []):
            if m.get("role") == "user":
                q = m.get("content", "")
                break
        ev = (row.get("context") or row.get("evidence") or "").strip()
        if not ev:
            # Scoring "is every claim supported by the evidence" against an EMPTY
            # evidence block floors both sides and looks like a result. Refuse.
            skipped += 1
            print(f"[judge] {i+1}/{len(picked)} {row['id']} SKIPPED — no context",
                  flush=True)
            continue
        prompt = PROMPT.format(q=q[:900], ev=ev[:1800],
                              a=(a or "")[:1200], b=(b or "")[:1200])
        verdict, err = call_gemini(prompt)
        if verdict is None:
            print(f"[judge] {i+1}/{len(picked)} {row['id']} FAILED: {err}", flush=True)
            continue
        def side(key):
            return verdict["A"] if (key == "adapter") == (not flip) else verdict["B"]
        rec = {"id": row["id"], "kind": row["kind"], "flip": flip,
               "adapter": side("adapter"), "base": side("base"),
               "pick": verdict.get("pick"),
               "pick_is_adapter": (verdict.get("pick") == "A") == (not flip)}
        results.append(rec)
        with open(args.out, "w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        print(f"[judge] {i+1}/{len(picked)} {row['id']} ({row['kind']}) done", flush=True)

    if skipped == len(picked):
        print("[judge] every row had empty evidence — refusing to emit scores. "
              "Check ref_span()/the input file before trusting a judge run.",
              file=sys.stderr)
        return 2

    if results:
        n = len(results)
        def mean(f):
            return round(sum(f(r) for r in results) / n, 2)
        print("\n[judge] mean scores (n=%d)" % n)
        for k in ("grounded", "provenance", "useful"):
            print(f"  {k:11} adapter {mean(lambda r: r['adapter'][k])}  "
                  f"base {mean(lambda r: r['base'][k])}")
        print(f"  judge preferred the adapter in "
              f"{sum(1 for r in results if r['pick_is_adapter'])}/{n} comparisons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
