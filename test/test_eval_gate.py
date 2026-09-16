#!/usr/bin/env python3
"""Regression tests for the A/B eval gate (run_eval.py / eval_run.py / eval_ab.py).

Background: the 2026-09-13 A/B eval was driven by hand on a Colab session, so the
gate that decides "may this adapter be frozen / deployed?" was not repeatable and
had no workflow. These tests pin the automated contract:

  * `eval-ab.yml` is dispatch-only, runs `run_eval.py`, uploads `out/`;
  * the runner hands the VM an ABSOLUTE session wall and the eval finalizes a
    PARTIAL comparison set on it (same lesson as training: a silent recycle must
    not cost the whole session);
  * the VM publishes an `eval_done.json` marker to `results/ab-eval-<date>/`
    (success AND failure) because the log channel dies ~60 min in;
  * the runner scores mechanically only — the blinded Gemini judge stays local
    (Gemini web auth is refused from runner IPs).

Runs offline: no Colab, no Drive, no network.
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else ' — ' + str(extra)}")
    if not cond:
        FAILS.append(name)


def _read(p):
    return (REPO / p).read_text()


def test_workflow_is_dispatch_only_and_runs_the_runner():
    wf = _read(".github/workflows/eval-ab.yml")
    check("workflow exists and parses as YAML text", "workflow_dispatch:" in wf)
    check("dispatch-only (no cron: this costs a GPU session on purpose)",
          "schedule:" not in wf and "cron:" not in wf)
    check("runs run_eval.py", "run: python run_eval.py" in wf)
    check("uploads out/ artifacts", "path: out/" in wf)
    check("same secrets as the daily run", "COLAB_REFRESH_TOKEN" in wf and "GDRIVE_ADC" in wf)
    check("own concurrency group", "group: gi-egtkg-eval" in wf)
    check("wall input exposed", "wall_minutes:" in wf and "EVAL_WALL_MINUTES" in wf)
    check("freeze rule documented in the workflow", "freeze" in wf.lower() and "5/5 grounded" in wf)
    check("judge left to a local machine (documented)",
          "judge_ab.py" in wf and "consent page" in wf)


def test_runner_wall_and_session_discipline():
    src = _read("run_eval.py")
    check("wall computed from session create",
          "wall_epoch = int(session_created_at + WALL_MINUTES * 60)" in src)
    check("wall passed to the VM launcher", "'--deadline-epoch', '{wall_epoch}'" in src)
    check("poll deadline anchored to the wall + finalize slack",
          "deadline = (wall_epoch + FINALIZE_SLACK_MIN * 60)" in src)
    check("uses its OWN session name (never fights the training session)",
          'SESSION = os.environ.get("EVAL_SESSION", "gi-egtkg-eval")' in src)
    check("stop session on every exit path", src.count('rd.colab("stop", "-s", SESSION') >= 3)
    check("keep-alive loop started and stopped",
          "rd._start_keep_alive(SESSION)" in src and "rd._stop_keep_alive(keep_alive_stop)" in src)
    check("uploads the scripts the VM needs", all(
        f'"/content/{n}"' in src for n in
        ("eval_run.py", "eval_build.py", "eval_ab.py", "pubmed_ingest.py",
         "egtkg_build.py", "daily_finetune.py", "gdrive.py")))
    check("judge NOT invoked from the runner (only mentioned in a log hint)",
          "import judge_ab" not in src and 'judge_ab.py",' not in src)
    check("mechanical scoring on the runner", "ab_score.py" in src and "scored_summary.json" in src)


def test_vm_orchestrator_publishes_markers():
    src = _read("eval_run.py")
    check("held-out pool from a fresh ingest into a throwaway store",
          "eval_scratch_store.jsonl" in src and "--max-oa" in src)
    check("training store pulled from Drive defines held-out",
          "_latest_store_local" in src and "--train-store" in src)
    check("aborts when nothing is held-out",
          "nothing held-out to evaluate" in src)
    check("publishes eval_done.json on success", 'drive.upload_replace(tmp, folder, "eval_done.json")' in src)
    check("publishes a FAILURE marker too (runner must not wait for the deadline)",
          "[EVALRESULT] ok=false" in src)
    check("partial set accepted and reported", '"partial": partial' in src)
    check("drive folder name is the dated ab-eval-<date>", 'folder_name = f"ab-eval-{date}"' in src)
    check("adapter evaluated is the one in adapter_in (--adapter-folder)", "--adapter-folder" in src)


def test_failure_reporting_is_diagnosable():
    """A failed eval must surface its REASON, not 'artifact missing'."""
    runner = _read("run_eval.py")
    vm = _read("eval_run.py")
    j = _read("judge_ab.py")
    check("judge scores against the model's own context, not the bare span",
          'r["context"] = q' in j and 'row.get("context") or row.get("evidence")' in j)
    check("judge refuses to score with no context",
          "refusing to emit scores" in j)
    check("ref_span falls back to the evidence field (scored files have no messages)",
          'row.get("evidence")' in _read("ab_score.py"))
    ab = _read("eval_ab.py")
    daily = _read("run_daily.py")
    check("runner parses ok=false from [EVALRESULT]",
          'r"\\[EVALRESULT\\]\\s*ok=(true|false)' in runner)
    check("runner always resolves the Drive marker for the structured reason",
          "read_marker(" in runner and "eval failed on the VM" in runner)
    check("VM tees the child output to a file", "def run_tee(" in vm and '"eval_ab.log"' in vm)
    check("VM ships that log with the marker", "extra_files=[ab_log]" in vm)
    check("VM ships that log on success too", '(os.path.join(args.out_dir, "eval_ab.log"), "eval_ab.log")' in vm)
    check("adapter fetch tries the in-process Drive API first",
          "from daily_finetune import Drive" in ab and "adapter fetched via Drive API" in ab)
    check("adapter fetch reports every exhausted path", "all three fetch paths exhausted" in ab)
    check("logs redact the Drive account display name", "_ACCT_NAME_RE" in daily)
    j = _read("judge_ab.py")
    check("judge scores against the model's own context, not the bare span",
          'r["context"] = q' in j and 'row.get("context") or row.get("evidence")' in j)
    check("judge refuses to score with no context",
          "refusing to emit scores" in j)
    check("ref_span falls back to the evidence field (scored files have no messages)",
          'row.get("evidence")' in _read("ab_score.py"))
    ab = _read("eval_ab.py")
    check("the VM installs the quantized-load stack itself",
          "bitsandbytes>=0.46.1" in ab)
    check("deps are verified before the model loads",
          "verify_deps()" in ab and "def verify_deps" in ab)
    check("verify runs even with --skip-install",
          "if args.skip_install:\n        verify_deps()" in ab)
    check("the orchestrator no longer suppresses eval_ab's install",
          '"--skip-install"], ab_log' not in _read("eval_run.py"))
    check("runner ignores an eval_done.json from an earlier run (date-named folder)",
          "def read_marker(" in runner and 'dr["run_id"] != run_id' in runner)
    check("runner falls back to marker mtime when run_id is absent",
          "not_before_epoch" in runner and "before this run" in runner)


def test_eval_ab_wall_behaviour():
    src = _read("eval_ab.py")
    check("--deadline-epoch arg", '"--deadline-epoch"' in src)
    check("stops generating on the wall", "time.time() > args.deadline_epoch" in src)
    check("finalizes what it has", "finalizing the partial set" in src)
    check("marks the output partial", "partial: session wall" in src)
    check("results are flushed per row (survive a kill)",
          src.count("fh.flush()") >= 1 and '"\\n")' in src)


if __name__ == "__main__":
    test_workflow_is_dispatch_only_and_runs_the_runner()
    test_runner_wall_and_session_discipline()
    test_vm_orchestrator_publishes_markers()
    test_failure_reporting_is_diagnosable()
    test_eval_ab_wall_behaviour()
    print()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
