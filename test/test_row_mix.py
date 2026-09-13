#!/usr/bin/env python3
"""Regression tests for the training row mix (egtkg_build.build_rows).

Why: the held-out A/B (2026-09-13) showed `kg` relation rows — which did NOT
carry the evidence span in the prompt — teaching the model to fabricate studies,
PMIDs and disease context (blinded judge: grounded 1.29 vs the base model 1.57).
The mix also had ZERO abstention examples, so the model could not decline an
unanswerable question. These tests pin the corrected contract.

Offline: no network, no model.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import egtkg_build as eb  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


ART_A = {
    "pmid": "42000001", "journal": "Gut", "year": "2026",
    "title": "Rifaximin for hepatic encephalopathy",
    "abstract": ("Rifaximin reduced the risk of overt hepatic encephalopathy recurrence "
                 "compared with placebo (HR 0.42, 95% CI 0.28-0.63) over 6 months. "
                 "Treatment was well tolerated in 91% of patients."),
}
ART_B = {
    "pmid": "42000002", "journal": "Hepatology", "year": "2026",
    "title": "Portal vein thrombosis in cirrhosis",
    "abstract": ("Portal vein thrombosis was present in 17% of screened patients. "
                 "Anticoagulation was associated with recanalization in 46% of them."),
}


def build(n_articles=2, seed=5):
    store = {"42000001": ART_A, "42000002": ART_B} if n_articles == 2 else {"42000001": ART_A}
    return eb.build_rows(store, random.Random(seed))


def test_kg_rows_carry_the_evidence():
    rows = build()
    kg = [r for r in rows if r.get("kind") == "kg"]
    check("kg rows are produced", len(kg) > 0, f"{len(kg)}")
    ok = all('Retrieved evidence from' in r["messages"][0]["content"] and '"' in r["messages"][0]["content"]
             for r in kg)
    check("every kg prompt carries a quoted evidence span", ok)
    ev_ok = all(r.get("e") and r["e"] in r["messages"][0]["content"] for r in kg)
    check("kg evidence is the exact span stored in the row (auditable)", ev_ok)
    arcs = (eb.source_node(ART_A), eb.source_node(ART_B))
    ans_ok = all(any(a in r["messages"][1]["content"] for a in arcs) for r in kg)
    check("kg answers cite the source node (journal/year/PMID)", ans_ok)


def test_abstention_rows_exist_and_are_unrelated():
    rows = build()
    ab = [r for r in rows if r.get("kind") == "abstain"]
    check("abstention rows are produced", len(ab) > 0, f"{len(ab)}")
    check("abstention answer is the refusal string",
          all(r["messages"][1]["content"] == "Not answerable from the retrieved evidence." for r in ab))
    # the cited evidence must NOT contain the asked-about head term
    bad = 0
    for r in ab:
        q = r["messages"][0]["content"]
        span = q.split('"')[1] if '"' in q else ""
        head = q.rsplit("how does ", 1)[-1].split(" relate to ")[0]
        if head and head.split()[0].lower() in span.lower():
            bad += 1
    check("abstention evidence is unrelated to the asked relation", bad == 0, f"{bad} leaks")


def test_no_abstention_without_a_second_article():
    rows = build(n_articles=1)
    check("single-article store yields no abstention rows",
          not any(r.get("kind") == "abstain" for r in rows))


if __name__ == "__main__":
    test_kg_rows_carry_the_evidence()
    test_abstention_rows_exist_and_are_unrelated()
    test_no_abstention_without_a_second_article()
    print()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
