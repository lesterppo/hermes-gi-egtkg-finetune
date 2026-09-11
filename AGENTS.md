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
- **Liveness must not depend on the VM log alone.** `colab logs`/`colab exec`
  starts failing with rc=1 about 60 min into a session (colab-cli access-token
  lifetime; re-minting token.json does not revive it) while training keeps
  running normally. A log-only stall rule therefore kills healthy VMs: that is
  exactly what happened nightly — the runner declared HEARTBEAT_STALLED at
  ~75-82 min and ran `colab stop`, and Drive proves the VM was alive (2026-09-10:
  stop at 19:08, VM pushed step-100 at 19:16; 2026-09-04: stop then step-150).
  Current design: TWO channels — the log ([alive] heartbeats) and Drive
  (`checkpoints/<run_id>/heartbeat.json` every 10 steps + step-N adapters every
  `--save-steps`) — and the VM is only declared dead when BOTH are quiet for
  `RUN_STALL_MINUTES` (default 45, must exceed the 25-35 min checkpoint cadence).
- **Success must not depend on the VM log either.** The VM writes
  `result.json` (`{ok, run_id, steps, loss, partial}`) into its run folder after
  verify + archive, and the runner accepts `[RESULT] ok=true` from Drive when the
  log channel is dark; the colab download step is skipped in that case (same dead
  channel) and `out/metrics.json` is built from the Drive payload.
- **Never recover a checkpoint without a grace window.** The VM keeps pushing
  after the runner gives up, so recovery waits `RUN_RECOVER_WAIT_MINUTES`
  (default 12) for Drive activity to advance before copying into adapter_in;
  otherwise the next run resumes from an older step (Sep-10 lost 50 steps this
  way).
- Log BOTH stdout and stderr on a failed `colab` call: `colab.py die()` prints
  `{"ok":false,"err":...}` to STDOUT, so stderr-only logging hides the cause.
