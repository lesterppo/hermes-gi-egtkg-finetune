# CI blocked by GitHub billing — 2026-09-15

## Symptom

Every `workflow_dispatch` run on this **private** repo failed within ~2 seconds
with **zero recorded steps** and no logs:

```
$ gh api repos/lesterppo/hermes-gi-egtkg-finetune/actions/runs/<id>/jobs
JOB finetune completed failure 2026-09-15T20:36:32Z 2026-09-15T20:36:34Z
  steps: []
```

A scheduled run (35007645838, 18:27 UTC) had succeeded ~2 h earlier, which made
it look like a code/secret problem. It was not.

## Diagnosis

The check-run annotation names it:

```
$ gh api repos/lesterppo/hermes-gi-egtkg-finetune/check-runs/<job_id>/annotations
"The job was not started because recent account payments have failed or your
 spending limit needs to be increased. Please check the 'Billing & plans'
 section in your settings"
```

Account-wide Actions usage (`gh api /users/lesterppo/settings/billing/usage`):

| month | Linux minutes |
|---|---|
| 2026-06 | 1,489 |
| 2026-07 | 5,576 |
| 2026-08 | 5,880 |
| 2026-09 (to 09-15) | 2,697 |

Free private-repo allowance is 2,000 min/month. The daily fine-tune run is
~150 min (wall 120 + setup/finalize), i.e. ~4,500 min/month on its own, so the
account crosses the threshold mid-month and GitHub then refuses to start jobs
until the spending limit is raised or the cycle resets (1st of the month).

**Rule: a 2-second job with `steps: []` is billing. Check the annotation before
touching the workflow, the secrets or Colab.**

## What was NOT the cause

- Colab quota: `gpu-unavailable` is a *different*, per-account failure (see the
  rotation pool) and it happens inside the "Run daily EGT-KG fine-tune" step.
- The rotated `COLAB_REFRESH_TOKEN`: the token mints fine locally
  (`mint_access_token()` OK, `expires_in 3599`).
  (Separately: `gh secret set NAME -R repo -b -` writes the literal `"-"` — gh
  reads stdin only when `--body` is omitted. Fixed in `colab_rotate.py`.)

## Unblocking options (decide, then act)

| option | cost | notes |
|---|---|---|
| make this repo **public** | $0 | public repos get unlimited Actions minutes; repo is code-only (credentials are GitHub Secrets, no PHI) |
| raise the Actions **spending limit** | ~$25-35 / month | ~$0.008/min Linux overage for the current daily schedule |
| schedule 3 nights/week | $0 | fits ~2,000 min/month together with the other repos' jobs |
| wait for the **Oct 1** reset | $0 | ~2 weeks without training |

Until one is chosen, the daily loop and the A/B eval gate (`eval-ab.yml`) cannot
run, and every future scheduled run on this account will fail the same way.

## State when the block hit (all good — nothing lost)

- Scheduled run 35007645838 **succeeded**: 185 steps banked, `result.json`
  `{ok:true, steps:185, partial:true, ingest_ok:true}` — the session-wall budget
  stopping the run before Google recycled the VM.
- Adapter chain in `adapter_in`: 374 (old row mix) + 200 + 200 + 71 + 185 steps
  of the fixed mix (grounded `kg` prompts + abstention rows).
- Eval gate shipped (`eval-ab.yml`, `run_eval.py`, `eval_run.py`) but not yet
  executed: it needs one GPU session **and** Actions minutes.
