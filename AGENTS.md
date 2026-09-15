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
  training detached, then polls TWO liveness channels (the VM log heartbeats and
  the run folder's Drive artifact mtimes), extends its deadline from the first
  heartbeat, and on any non-success exit settles on Drive's `result.json` before
  recovering the newest checkpoint.
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
   verify + Drive push) is the success signal, and the equivalent out-of-band
   signal is `result.json` `{ok:true}` in the run folder on Drive — the runner
   accepts either (the log channel is dead after ~60 min, so late runs only ever
   surface via `result.json`). `EXIT …` alone is NOT success.
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
- **The VM's life is NOT ours to control — budget against a WALL, not the
  epoch.** Free-tier sessions are recycled without warning and the assignment
  404s at the keep-alive endpoint the instant it happens. Measured: 2026-09-13
  created 17:38:54, first 404 20:00:19; 2026-09-14 19:36:01 → 21:56:13 — both
  **~2h20m after creation**, while 09-11/12 lived 3h+. Both of those nights were
  killed mid-epoch, so the VM never reached save + archive + `result.json`, the
  runner could only salvage the newest periodic checkpoint, and the night was
  reported as FAILURE (three failed nights; the A/B row-mix fix `8c0d2ca` was
  never once exercised in a completed run). Fix: `run_daily` stamps
  `RUN_WALL_MINUTES` (default 120) from **session creation** and passes the
  absolute epoch to the VM as `--deadline-epoch`; `TimeBudgetCallback` stops
  there (`stop_reason: session-deadline`, `partial: true`) and the run still
  finalizes — save, verify, archive, `adapter_in`, `result.json {ok:true}` — so
  a wall hit is a SUCCESS with the night's steps banked. The runner's poll
  deadline is anchored to the same wall (+ `WALL_FINALIZE_SLACK_MIN`), no longer
  `max_minutes + 20` past a VM that is already gone. Consequence to accept: an
  epoch longer than the wall is finished incrementally over several nights —
  that is what "continuous daily fine-tune" means. `metrics.json` carries
  `stop_reason` (`epoch-complete` | `budget` | `session-deadline`).
- **`gpu-unavailable` is per-account, not global.** When all four `colab new`
  retries return it, the account behind `COLAB_REFRESH_TOKEN` has burned its
  free-tier GPU quota for the day (~3-4 sessions) — verify by creating a T4
  session locally on a different account before blaming Colab. Rotate the
  secret (see the colab-cli rotation notes) or wait for the daily reset; the
  runtime account may differ from the Drive account (`GDRIVE_ADC`) safely.
- **The A/B eval is the gate — and it is now automated, but the JUDGE is not.**
  Dispatch `eval-ab.yml` (`run_eval.py`): free Colab T4 → pull the training
  store → fresh ingest → held-out rows (`eval_build.py`) → generate every answer
  twice, adapter ON vs `model.disable_adapter()` OFF → publish
  `ab_results.jsonl` + `eval_rows.jsonl` + `eval_done.json` to
  `results/ab-eval-<date>/` → the runner scores mechanically (`ab_score.py`) and
  uploads `out/`. `judge_ab.py` (blinded Gemini Pro) stays LOCAL: Gemini web auth
  is refused from runner IPs (consent page), so run it against the downloaded
  `ab_results.jsonl` from a machine with live browser cookies. Do not freeze or
  deploy an adapter without re-running the gate against the CURRENT `adapter_in`:
  the 2026-09-13 numbers (extractive rows 5/5 grounded, 7/7 preferred; `kg` rows
  actively harmful, base scored higher) were measured on `results-2026-09-12`,
  which is NOT what `adapter_in` holds now.
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
  That fix paid off immediately: the log channel's real failure is
  `{"err":"not-found"}` (colab-cli loses the session mapping), NOT an auth
  expiry — so token re-minting was never the fix. `colab recover` is retried
  twice when that error appears.
- **The poll deadline must start at TRAINING start, not at exec_detach.** VM
  setup (pip installs + PubMed ingest + model load) runs 29-49 min and does NOT
  count toward the VM's own `--max-minutes`, so a `max_minutes + 25` budget can
  expire while training is legitimately running: run 34624713317 timed out at
  18:36:29 and the VM pushed `result.json` at 18:40:11 — 4 minutes late.
  `poll_log(train_budget_s=...)` extends the deadline to
  `first_heartbeat + (max_minutes + 20) min`.
- **Settle before recovering.** Every stall/timeout/no-`[RESULT]` exit first
  waits `RUN_RECOVER_WAIT_MINUTES` for `result.json`; `ok=true` means the run
  COMPLETED (that file is written after verify + archive + adapter_in), so
  report SUCCESS and leave adapter_in alone instead of overwriting it with a
  mid-run checkpoint (the same run would otherwise lose 50+ steps).
- **Replace single-instance Drive files in place.** Same-name uploads create NEW
  files and Drive listings are eventually consistent, so delete-then-upload still
  duplicates: `heartbeat.json` / `loss_curve.json` / the latest-adapter pointer /
  the dated archive all use `upload_replace` (update the existing file id), with
  a list+delete fallback. `results-<date>/` is that date's FINAL STATE — a smoke
  run plus a real run on one day must not leave two 84 MB adapters side by side.
- Heartbeat polling is cheap insurance: the VM's `heartbeat.json` lands every
  10 steps (~3-4 min) while step checkpoints are ~20 min apart, so a stall
  window can be both tight and safe. Watch the ARTIFACT STORE for live progress
  when a GH step is still running — GitHub only publishes step logs on
  completion.

