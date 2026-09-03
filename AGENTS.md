# AGENTS.md — hermes-gi-egtkg-finetune

Guidance for AI agents (and humans) working in this repo.

## What this is
A headless, daily, evidence-grounded LoRA fine-tune of
Llama-3.1-8B-Instruct on up-to-date gastroenterology / hepatology literature,
orchestrated by GitHub Actions, executed on a **free Google Colab T4**, with
Google Drive as the persistent store and the deliverable adapter. No local
machine and no API keys needed to run.

## Architecture (data → model → Drive)
- `pubmed_ingest.py` — CPU/stdlib runner+VM ingest. PubMed esearch/efetch
  (keyless E-utilities) scoped to GI/hep via MeSH/journals/tiab, over an
  explicit date range (`N:days[edat]` returned 0 on our backend; use
  `DAY1:DAY2[edat]`), plus Europe PMC OA full text (JATS XML) for open papers.
  Writes/accumulates `knowledge_store.jsonl` (dedupe by PMID, capped).
- `egtkg_build.py` — builds the typed, **reified** knowledge graph and emits
  the QA rows the trainer learns from. Grounding rule (the point of the EGT-KG
  paper): answers must be constructible from **verbatim evidence spans** with
  provenance (journal + year + PMID) — never from an LLM-rephrased summary.
- `daily_finetune.py` — runs on the VM. Installs the known-good ML stack,
  calls the two stages above, then continues an NF4 QLoRA training run
  (r16/α32, all 7 linears, batch 4, seq ≤512, **bf16** — fp16 has no sm_75
  GradScaler path). Emits `[alive] step=N` heartbeats and `[RESULT] ok=true`
  only after everything succeeded.
- `run_daily.py` — GitHub Actions orchestrator. Mints Colab tokens, creates a
  T4 session, uploads the staged scripts (`daily_finetune.py`, `egtkg_build.py`,
  `pubmed_ingest.py`, vendored `gdrive.py`/`colab.py`) + the ADC, launches
  training detached, polls with stall detection + its own keep-alive loop, and
  recovers the newest Drive checkpoint on any failure.
- `.github/workflows/daily-egtkg.yml` — cron 14:40 UTC (22:40 HKT) + dispatch.
- `deploy_chat.py` — standalone: pull latest Drive adapter, load base in NF4 +
  LoRA, answer medical questions (single-file, runs on the T4).

## Hard rules
1. **Never commit credentials.** Everything is GitHub secrets / env vars:
   `COLAB_CLIENT_ID`, `COLAB_CLIENT_SECRET`, `COLAB_REFRESH_TOKEN`,
   `GDRIVE_ADC` (authorized_user JSON), `DRIVE_ADAPTER_IN`, `DRIVE_RESULTS`.
   Grep for `ya29.`, `1//0`, client_secret literals, gcloud client-id
   patterns (`NNNNNNNNNN-*.apps.googleusercontent.com`),
   email addresses, `/home/`, `.config/colab-cli` before any commit.
   `.gitignore` excludes `*.json` and runner output.
2. **Colab free tier is ephemeral** (~2 h kernel recycle / ~12 h cap, RAM
   cgroup ~12 GB). The persistent artifact is the adapter on Drive; /content
   is wiped each run. Nothing on the VM survives.
3. **`N:days[edat]` NCBI indexing is unreliable** here. Use explicit date
   ranges (see pubmed_ingest).
4. **Healthy VS green-log are different.** `[RESULT] ok=true` (printed after
   verify + Drive push) is the only success signal. `EXIT …` alone is failure.
5. **Relationship extraction is conservative and deterministic** (no 30B LLM
   like the paper's AutoSchemaKG). Entities are filtered for clause-swallow
   junk; rows are evidence-grounded, so a noisy head occasionally appears but
   the assistant text always quotes real evidence verbatim.

## Drive layout (the Run account's Google Drive)
- `gi-egtkg/adapter_in/` — latest adapter (continuity pointer; shared
  anyone-with-link so `deploy_chat.py` can `gdown --folder` it).
- `gi-egtkg/results/<date>/adapter_model.safetensors` + `metrics.json` +
  `loss_curve.json` + `knowledge_store.jsonl` — dated archives.
- `gi-egtkg/results/checkpoints/<run>/step-N-*.safetensors` — drive checkpoints.

## Pitfalls already learned (do not re-learn)
- Heartbeat stall + runner keep-alive, in-process Drive uploads, model-only
  checkpoints, bf16 not fp16, jupyter-kernel-client compat patch, unique
  run-ids, pre-clean orphaned assignments, gpu-unavailable retries.
