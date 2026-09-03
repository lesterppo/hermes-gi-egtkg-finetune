#!/usr/bin/env python3
"""
egtkg_build.py — Evidence-Grounded Typed KG + QA training rows for GI/hepatology.

Adaptation of EGT-KG (arXiv:2609.00479) applied to continuous fine-tuning of a
local small LLM on gastroenterology/hepatology literature. Runs fully on the
Colab VM, CPU-only, pure stdlib.

Per the paper:
  1. triples are extracted from evidence chunks, then
  2. free-form relations are typed into a schema (our AS variant), and
  3. a reified store keeps every triple glued to its verbatim evidence span + a
     provenance source node (PMID + title + position), so the graph guides
     retrieval but generation grounds solely on original evidence text — never
     on an LLM-compressed summary the paper warns degrades answers.

Training rows (ChatML messages) are generated for a local SLM fine-tuner
(same pipeline as the proven soup-daily-finetune pattern, but now evidence-grounded
medical QA instead of finance alpaca).
"""
import argparse
import json
import pathlib
import random
import re

# ---------------------------------------------------------------------------
# Typed schema (AS-style, generated for the GI+hepatology domain)
# ---------------------------------------------------------------------------
# relation -> typed category used to constrain traversal (paper AS granularity)
REL_TYPE = {
    "TREATS": "causes_or_influences",
    "EFFECTIVE_FOR": "causes_or_influences",
    "ASSOCIATED_WITH": "causes_or_influences",
    "RISK_FACTOR_FOR": "causes_or_influences",
    "CAUSED_BY": "causes_or_influences",
    "CAUSES": "causes_or_influences",
    "CONTRAINDICATED_IN": "causes_or_influences",
    "PREDICTS": "research_or_observation",
    "DIAGNOSTIC_OF": "definition",
    "MARKER_OF": "definition",
    "OCCURS_IN": "research_or_observation",
    "METHOD_FOR": "method",
}

# Relation patterns. Each has named groups H (head entity) and T (tail).
# Conservative: only clear clinical phrasings with capitalized (start-of-
# sentence / proper-name) noun phrases. First match wins per sentence.
_PATTERNS = [
    # --- TREATS / EFFECTIVE_FOR ---
    ("TREATS", r"(?P<H>[A-Z][A-Za-z\- ]{2,36}?)\s+is\s+(?:a\s+)?(?:first[-\s]line\s+|standard\s+|recommended\s+|current\s+|potent\s+|effective\s+)?treatment\s+for\s+(?P<T>[A-Z][A-Za-z\- ]{2,42})"),
    ("TREATS", r"(?P<H>[A-Z][A-Za-z\- ]{2,34})\s+is\s+indicated\s+for\s+(?P<T>[A-Z][A-Za-z\- ]{2,40})"),
    ("TREATS", r"(?P<T>[A-Z][A-Za-z\- ]{2,40})\s+(?:is|was)\s+treated\s+(?:with|using)\s+(?P<H>[A-Z][A-Za-z\- ]{2,38})"),
    ("EFFECTIVE_FOR", r"(?P<H>[A-Z][A-Za-z\- ]{2,36})\s+(?:was|were|is)\s+(?:highly|significantly)?\s*effective\s+for\s+the\s+treatment\s+of\s+(?P<T>[A-Z][A-Za-z\- ]{2,38})"),
    ("EFFECTIVE_FOR", r"(?P<T>[A-Z][A-Za-z\- ]{2,36})\s+was\s+significantly\s+improved\s+by\s+(?P<H>[A-Z][A-Za-z\- ]{2,32})"),
    # --- ASSOCIATED_WITH ---
    ("ASSOCIATED_WITH", r"(?P<H>[A-Za-z][A-Za-z\- ]{2,38}?)\s+(?:was|were|is|are)\s+(?:independently|significantly)?\s*(?:associated|linked)\s+with\s+(?:an?\s+)?(?P<T>[A-Za-z][A-Za-z\- ]{2,42})"),
    # --- RISK_FACTOR_FOR ---
    ("RISK_FACTOR_FOR", r"(?P<H>[A-Za-z][A-Za-z\- ]{2,36})\s+(?:is|was|were)\s+(?:a|an|the)\s+(?:independent|significant|strong|established)\s+risk\s+factor\s+for\s+(?:the\s+development\s+of\s+)?(?P<T>[A-Za-z][A-Za-z\- ]{2,42})"),
    # --- CAUSED_BY / CAUSES ---
    ("CAUSED_BY", r"(?P<T>[A-Za-z][A-Za-z\- ]{2,40})\s+(?:is|was)\s+(?:most\s+)?commonly\s+caused\s+by\s+(?P<H>[A-Za-z][A-Za-z\- ]{2,38})"),
    ("CAUSES", r"(?P<H>[A-Za-z][A-Za-z\- ]{2,36})\s+(?:is|are|was|were)\s+(?:the\s+)?(?:most\s+)?common\s+cause\s+(?:of|for)\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,42})"),
    ("CONTRAINDICATED_IN", r"(?P<H>[A-Za-z][A-Za-z\- ]{2,35})\s+(?:is|are|was|were)\s+contraindicated\s+in\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,38})"),
    # --- DIAGNOSTIC_OF ---
    ("DIAGNOSTIC_OF", r"(?P<H>[A-Za-z][A-Za-z\- ]{1,14}[\w\- ]{0,20}?)\s+(?:is|was)\s+(?:a|an)\s+(?:sensitive|specific|useful|reliable|non[-\s]invasive|accurate)\s+(?:marker|biomarker|test|finding|sign|tool|score)\s+(?:of|for)\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,42})"),
    ("DIAGNOSTIC_OF", r"(?P<T>[A-Za-z][A-Za-z\- ]{2,40})\s+(?:was|is)\s+(?:diagnos|detect)ed\s+(?:using|by|with)\s+(?P<H>[A-Za-z][A-Za-z\- ]{2,36})"),
    # --- MARKER_OF ---
    ("MARKER_OF", r"(?P<H>[A-Za-z][A-Za-z\- ]{1,14}[\w\- ]{0,22}?)\s+(?:is|are|was|were)\s+(?:a|an)\s+(?:useful|promising|reliable|non[-\s]invasive|serum|novel)\s+(?:marker|biomarker|indicator)\s+(?:of|for)\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,42})"),
    # --- PREDICTS ---
    ("PREDICTS", r"(?P<H>[A-Za-z][A-Za-z\- ]{1,14}[\w\-\. ]{0,26}?)\s+(?:is|was)\s+(?:independently\s+)?(?:predictive\s+of|able\s+to\s+predict)\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,42})"),
    # --- OCCURS_IN (percentage) ---
    ("OCCURS_IN", r"(?P<H>[A-Za-z][A-Za-z\- ]{2,34}?)\s+(?:occurs?|occurred|developed)\s+in\s+(?P<T>\d{1,3}(?:\.\d+)?)\s*%\s+of\s+(?:patients|cases)"),
    # --- METHOD_FOR ---
    ("METHOD_FOR", r"(?P<H>[A-Za-z][A-Za-z\- ]{1,14}[\w\- ]{0,14}?)\s+(?:is|are|was)\s+used\s+to\s+(?:diagnose|treat|screen|detect|assess|monitor|evaluate)\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,38})"),
    # --- extra clinical facts (comparative outcome / risk / survival) ---
    ("PREDICTS", r"(?P<H>[A-Za-z][A-Za-z\- ]{2,32})\s+was\s+(?:a\s+)?(?:significant|independent)\s+(?:predictor|prognostic\s+factor)\s+of\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,40})"),
    ("PREDICTS", r"(?P<H>[A-Za-z][A-Za-z\- ]{1,12}[\w\- ]{0,12}?)\s+(?:significantly|independently|strongly)\s+predict\w*\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,44})"),
    ("RISK_FACTOR_FOR", r"(?P<H>[A-Za-z][A-Za-z\- ]{2,32})\s+was\s+associated\s+with\s+(?:an?\s+)?(?:increased|reduced|higher|lower)\s+risk\s+of\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,40})"),
    ("OCCURS_IN", r"the\s+prevalence\s+of\s+(?P<H>[A-Za-z][A-Za-z\- ]{2,30})\s+was\s+(?P<T>\d{1,3}(?:\.\d+)?)\s*%"),
    # "higher/lower ... in patients with X"  -> associative occurrence
    ("ASSOCIATED_WITH", r"(?P<H>[A-Za-z][A-Za-z\- ]{2,26}?)\s+were\s+significantly\s+(?:\w+\s+){0,2}(?:higher|lower|greater)\s+in\s+patients\s+with\s+(?P<T>[A-Za-z][A-Za-z\- ]{2,36})"),
]
_COMPILED = [(rel, re.compile(pat)) for rel, pat in _PATTERNS]

# ---------------------------------------------------------------------------
# Question / answer templates. {H},{T} filled from the typed triple; {KM} is
# the verbatim evidence span (the reified witness text); {S} the source node
# ("Title (Journal, Year) PMID:xxxx"). Placeholders in assistant strings are
# the grounded answer form. Answers quote evidence + provenance.
# ---------------------------------------------------------------------------
KB_QA = [
    ("What is the effect of {H} on {T}?",
     "Evidence from gastroenterology/hepatology literature ({S}) reports: {KM}"),
    ("According to the literature, how does {H} relate to {T}?",
     "The evidence ({S}) states: {KM}"),
    ("Is {H} effective for {T}, and what does the article report?",
     "Evidence from {S}: {KM}"),
]
DEF_QA = [
    ("What does the literature say {H} is used for, or how is {T} diagnosed?",
     "From {S}: {KM}"),
]
MRK_QA = [
    ("Is {H} a clinically useful marker in {T}?",
     "Evidence from {S}: {KM}"),
]
# Mixed instruction-style factual QA that mirrors how the SLM is later asked. We
# make the "user" a factual question whose answer is an evidence sentence verbatim.
FACT_Q = [
    "What does this article report about the relationship between {H} and {T}?",
    "According to the evidence, {H} is {rel} [what]? Report exactly what the article says.",
    "Question based on the retrieved medical article: what is stated about {H} {T}?",
]
# Number-stat factual QA (the paper's "quantitative details that matter"):
NUM_FACT = [
    ("Based on the retrieved article, what fraction/rate was reported?",
     "{KM}")
]
# Diagnostic-inference QA keeps the clinical chain grounded.
DIAG_QA = [
    ("Given the evidence, what is the likely relationship between {H} and {T}?",
     "The article evidence ({S}) states: {KM}"),
]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip(".:,;\"'")


def sentences(text: str):
    """Split into>=2-word sentences; collapse whitespace; keep len info."""
    out = []
    for chunk in re.split(r"(?<=[.!?]) +", re.sub(r"\s+", " ", text or "")):
        chunk = chunk.strip()
        if len(chunk.split()) >= 4:
            out.append(chunk)
    return out


def load_store(store_path):
    """ingest store is a jsonl of dicts keyed by pmid (see pubmed_ingest)."""
    arts = {}
    for line in pathlib.Path(store_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            a = json.loads(line)
            arts[str(a.get("pmid", a.get("id", "")))] = a
        except Exception:
            continue
    return arts


def extract_triples(article):
    """Return list of {rel,type,H,T,e} typed triples with verbatim evidence."""
    text = (article.get("abstract") or "") + "\n" + (article.get("oa_fulltext") or "")
    triples = []
    for s in sentences(text):
        # guard: skip figure/ref/number-dense junk lines
        if len(s) > 600 or s.lower().startswith(("introduction", "method", "copyright")):
            continue
        for rel, rx in _COMPILED:
            m = rx.search(s)
            if not m:
                continue
            gd = m.groupdict()
            h = _clean(gd.get("H") or "")
            t = _clean(gd.get("T") or "")
            if not h or not t or h.lower() == t.lower():
                continue
            # keep evidence span bounded (~evidence window) — take a window
            win = 220
            center = (m.start() + m.end()) // 2
            ev = s[max(0, center - win // 2): center + win].strip()
            triples.append({
                "rel": rel,
                "type": REL_TYPE.get(rel, "research_or_observation"),
                "H": h, "T": t,
                "e": ev,
            })
            break
    # dedupe (rel,H,T) + drop subject-clause / phrase-swallow artifacts
    BAD_LEADS = {"analysis", "we", "patients", "this", "these", "treatment",
                 "the", "total", "median", "results", "study", "compared",
                 "remaining", "introduction", "conclusion", "background",
                 "aim", "methods", "purpose", "overall",
                 "expression", "development", "progression", "levels", "events",
                 "whereas", "higher", "lower", "and", "or", "but", "however",
                 "while", "although", "which", "among", "found"}
    JUNK_WORDS = {"expression", "development", "progression", "whereas", "events",
                  "levels", "indicating", "suggesting", "associated", "remained",
                  "remains", "significantly", "higher", "lower", "however",
                  "possible", "potential", "including", "compared"}
    dedup = set()
    out = []
    for tr in triples:
        k = (tr["rel"], tr["H"].lower(), tr["T"].lower())
        if k in dedup:
            continue
        Hw, Tw = tr["H"].split(), tr["T"].split()
        h0 = (Hw or [""])[0].lower().strip(".,;:-")
        t0 = (Tw or [""])[0].lower().strip(".,;:-")
        if (h0 in BAD_LEADS or t0 in BAD_LEADS):
            continue
        if len(tr["H"]) < 3 or len(tr["T"]) < 3:
            continue
        # entities should be short medical noun phrases; drop clause swallows
        if len(Hw) > 5 or len(Tw) > 6:
            continue
        if set(w.lower().strip(".,") for w in Hw) & JUNK_WORDS:
            continue
        if len(Tw) >= 2 and any(w.lower() in JUNK_WORDS for w in Tw):
            continue
        dedup.add(k)
        out.append(tr)
    # cap triples per article so a single noisy doc can't flood a run
    return out[:10]


def source_node(article):
    src = f"{article.get('title') or article.get('journal') or 'Article'}"
    src = _clean(src)
    jy = (article.get("journal") or "").strip()
    yr = (article.get("year") or "").strip()
    tail = []
    if jy and jy.lower().split()[0] not in (src.lower().split()[0] if src else ""):
        tail.append(jy)
    if yr:
        tail.append(yr)
    meta = " ".join(t for t in tail)
    pmid = article.get("pmid") or article.get("id")
    s = src
    if meta:
        s = f"{src} ({meta})"
    if pmid:
        s = f"{s} PMID:{pmid}"
    return s


def build_rows(store, rng):
    """Generate mixed ChatML training rows: {messages:[...]}."""
    rows = []
    for pmid, art in store.items():
        if not (art.get("abstract") or "").strip():
            continue
        src = source_node(art)
        tps = extract_triples(art)
        # EGT-KG grounding: any QA we emit must be answerable from evidence
        # held in the reified store. We do that by making the assistant answer
        # = the verbatim evidence sentence (never invented), plus provenance.
        if not tps:
            # Fallback: produce grounded comprehension rows only from sentences
            # carrying a quantified/clinical finding — teaches the model to
            # answer only what the evidence actually reports (with the number).
            ev = None
            for s in sentences(art["abstract"]):
                if re.search(r"\b\d{1,4}(?:\.\d+)?\s*(?:%|mg|days|weeks|months|years|vs\.|H?R\b|OR\b|CI)", s, re.I):
                    ev = s
                    break
            if ev:
                rows.append(
                    {"messages": [
                        {"role": "user",
                         "content": f"Using only the retrieved medical article "
                                    f"({src}), report the exact result stated:\n\"{ev}\"\n"
                                    f"Answer: what result/number does the article state? Reply precisely."},
                        {"role": "assistant",
                         "content": f"The article states: {ev} (Source: {src})"},
                    ], "kind": "evidence"}
                )
            continue
        # build typed-triple QA rows
        for tr in tps:
            H, T, KM, rel = tr["H"], tr["T"], tr["e"], tr["rel"]
            templates = KB_QA
            if rel in ("DIAGNOSTIC_OF",):
                templates = DEF_QA
            elif rel == "MARKER_OF":
                templates = MRK_QA
            u, a = rng.choice(templates)
            user_s = u.format(H=H, T=T)
            ans_s = a.format(H=H, T=T, KM=KM, S=src)
            rows.append(
                {"messages": [
                    {"role": "user", "content": user_s},
                    {"role": "assistant", "content": ans_s},
                ], "kind": "kg", "rel": rel,
                 "e": KM}  # keep evidence for provenance/audit, not for training
            )
        # a factual extractive row per article quoted verbatim from evidence
        s0 = rng.choice(sentences(art["abstract"]))
        rows.append(
            {"messages": [
                {"role": "user",
                 "content": f"Retrieved evidence from {src}: \"{s0}\"\n\n"
                            f"Using only the retrieved evidence, answer directly: "
                            f"what does this article report?"},
                {"role": "assistant",
                 "content": f"{s0} (Source: {src})"},
            ], "kind": "fact"}
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    store = load_store(args.store)
    if not store:
        print("[egtkg] empty store — nothing to build", flush=True)
        return
    rng = random.Random(args.seed)
    rows = build_rows(store, rng)
    # shuffle deterministic
    rng.shuffle(rows)
    if args.limit and len(rows) > args.limit:
        rows = rows[:args.limit]
    kinds = {}
    for r in rows:
        k = r.get("kind", "?")
        kinds[k] = kinds.get(k, 0) + 1
    pathlib.Path(args.out).write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    print(f"[egtkg] rows={len(rows)} keeps_kind={json.dumps(kinds)} store={len(store)}",
          flush=True)


if __name__ == "__main__":
    main()
