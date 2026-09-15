# EGT-KG: daily evidence-grounded GI/hepatology fine-tune on a free Colab T4

Continuous QLoRA fine-tuning of **Llama-3.1-8B-Instruct** on up-to-date
gastroenterology/hepatology literature, driven nightly from GitHub Actions and
persisted to **Google Drive**. Implements the retrieval/QA ideas of
[*EGT-KG: Evidence-Grounded Typed KG Retrieval for Practical Scientific QA
with Small Language Models*](https://arxiv.org/html/2609.00479v1) on medical
evidence:
answers are grounded on **verbatim evidence spans** (never LLM-compressed
summaries) and carry provenance (journal + year + PMID).

## How it works

```
GitHub Actions (cron 22:40 HKT)
   │  run_daily.py
   │    · mint Colab access token (server-side OAuth, no browser)
   │    · create free Colab T4 session, upload scripts + Drive ADC
   │    · poll VM log (heartbeat stall detection), keep-alive loop
   ▼
Colab VM  (daily_finetune.py)
   │  1. pubmed_ingest.py   fresh GI/hepatology evidence
   │       PubMed esearch/efetch  +  Europe PMC OA full texts (JATS XML)
   │       → accumulating  knowledge_store.jsonl  (dedup by PMID, capped)
   │  2. egtkg_build.py     typed reified KG + QA rows
   │       deterministic typed triples  +  verbatim evidence + provenance
   │       → grounded ChatML QA rows (kg / fact / evidence)
   │  3. QLoRA continue-train (NF4, r16/α32, 7 linears, bf16, batch 4)
   │  4. Drive checkpointing every N steps + date-archive + adapter pointer
   ▼
Google Drive  (the Run account)
   gi-egtkg/results/YYYY-MM-DD/…   (adapter + metrics + knowledge_store)
   gi-egtkg/adapter_in/…            (latest adapter — next run resumes)
```

No local machine and no API key needed to run: PubMed + Europe PMC are
keyless public endpoints, Colab is the free tier, Drive keeps state.

## Deploy

1. Create Drive folders under `gi-egtkg/` on the Run account's Drive:
   `adapter_in/` and `results/`. Record their folder ids
   (share `adapter_in` anyone-with-link as a `gdown` fallback, optional —
   the VM uses the Drive API so sharing is not required).
2. On GitHub, create repo, add **Actions secrets**:

   | secret | value |
   |---|---|
   | `COLAB_CLIENT_ID` / `COLAB_CLIENT_SECRET` | gcloud "Desktop" OAuth client id/secret |
   | `COLAB_REFRESH_TOKEN` | refresh token of the Run account (colaboratory+drive.file) |
   | `GDRIVE_ADC` | full `authorized_user` ADC JSON for that account (drive.file) |
   | `DRIVE_ADAPTER_IN` | folder id of `gi-egtkg/adapter_in` |
   | `DRIVE_RESULTS` | folder id of `gi-egtkg/results` |

3. Trigger the daily cron or a manual `workflow_dispatch` smoke
   (`rows: 300`) to verify.

No model weights or secret live in this repo; the runner installs nothing
heavy — the ML stack installs on the Colab VM each run.

## Evaluation gate (run before freezing or deploying an adapter)

The A/B gate answers "is this adapter actually better than the base model on
held-out literature?" — dispatch the `A/B eval gate` workflow:

```
Colab VM : pull training store -> fresh PubMed ingest -> keep only PMIDs ABSENT
           from the training store -> build QA rows -> generate every answer
           TWICE (adapter ON / model.disable_adapter() OFF, greedy, same prompt)
Runner   : download ab_results.jsonl -> ab_score.py (verbatim-evidence recall,
           provenance, trained shape, out-of-evidence words, truncation)
Drive    : results/ab-eval-<date>/{ab_results,eval_rows,eval_done}.json
```

The blinded Gemini Pro pass (`judge_ab.py`) is NOT in CI: Gemini web auth is
refused from runner IPs, so run it locally against the downloaded
`ab_results.jsonl`. Freeze rule (2026-09-13 baseline in
`docs/AB_EVAL_2026-09-13.md`): extractive rows (`evidence`/`fact`) stay 5/5
grounded AND the relation/abstention behaviour is removed or >= 4/5 grounded.

## Tuning knobs (workflow_dispatch)

- `rows` — evidence QA rows to train on (smoke **300**, daily **3000**).
- `ingest_days` — how far back to pull fresh literature (default 20).
- `max_minutes` — train budget (0 = until the ~2 h / ~12 h Colab recycle).
- `wall_minutes` — ABSOLUTE session wall (from session creation, default 120):
  the VM stops and finalizes there (`partial:true`, `stop_reason` in metrics),
  so a night can never be lost to a silent Colab recycle.
- `save_steps` — Drive checkpoint cadence.

## Why this design (lessons from the soup-daily-finetune pipeline)

Reused the battle-hardened Colab driver: heartbeat stall detection, its own
runner keep-alive loop (free-tier idle-prunes VMs after ~60 min), in-process
Drive uploads (subprocess spawns OOM the 12 GB RAM cgroup), model-only
checkpoints (`save_only_model`), bf16 (fp16 GradScaler has no sm_75 path),
`jupyter-kernel-client` compat patch, unique run ids, orphan-assignment
pre-clean, and "green log ≠ success" (only `[RESULT] ok=true` counts).
