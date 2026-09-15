#!/usr/bin/env python3
"""Regression tests for the SESSION WALL budget (run_daily -> daily_finetune).

Background (the bug these lock in): Google recycles free-tier Colab sessions
without warning and the assignment then 404s at the keep-alive endpoint while
training is still mid-epoch. Observed 2026-09-13 (created 17:38:54, first 404
20:00:19) and 2026-09-14 (19:36:01 -> 21:56:13) — both ~2h20m, while 09-11/12
survived 3h+. Both failures were killed mid-epoch, so the VM never reached its
save + archive + result.json, the runner could only salvage the newest periodic
checkpoint, and the night was reported as FAILURE.

Contract:
  * run_daily stamps a session wall (RUN_WALL_MINUTES, default 120) measured
    from SESSION CREATION and passes it to the VM as --deadline-epoch.
  * daily_finetune stops training at that absolute instant and still has time to
    verify/archive/push result.json {ok:true, partial:true} -> the runner can
    report SUCCESS and the night's steps are banked.
  * the runner's poll deadline is anchored to the SAME wall (previously
    max_minutes + 20 after training start, i.e. far past the recycle).

Runs offline: no Colab, no Drive, no network.
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import run_daily as rd  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else ' — ' + str(extra)}")
    if not cond:
        FAILS.append(name)


def test_wall_defaults_under_the_observed_recycle():
    """120 min default must stay below the ~140 min observed recycle."""
    src = (REPO / "run_daily.py").read_text()
    m = re.search(r'RUN_WALL_MINUTES", "(\d+)"', src)
    check("RUN_WALL_MINUTES has a default", m is not None)
    wall = int(m.group(1))
    check("wall default < 140 min (observed recycle)", wall < 140, f"got {wall}")
    check("wall default leaves training time after ~40 min setup", wall - 40 >= 60, f"got {wall}")
    check("WALL_MINUTES read at import", rd.WALL_MINUTES == wall, f"got {rd.WALL_MINUTES}")
    check("finalize slack defined", rd.WALL_FINALIZE_SLACK_MIN >= 10)


def test_launcher_passes_the_deadline():
    src = (REPO / "run_daily.py").read_text()
    check("launcher passes --deadline-epoch", "'--deadline-epoch', '{wall_deadline_epoch}'" in src)
    check("wall epoch computed from session create",
          "wall_deadline_epoch = int(session_created_at + WALL_MINUTES * 60)" in src)
    check("session_created_at set right after colab new",
          re.search(r"failed to create T4 session.*?session_created_at = time\.time\(\)", src, re.S) is not None)


def test_poll_deadline_is_wall_anchored():
    src = (REPO / "run_daily.py").read_text()
    check("poll deadline anchored to wall_deadline_epoch",
          "deadline = wall_deadline_epoch + WALL_FINALIZE_SLACK_MIN * 60" in src)
    check("adaptive extension clamped to the wall",
          "train_budget_s = max(600, deadline - time.time())" in src)


def test_vm_stops_on_the_deadline():
    """The VM callback must stop on the absolute epoch and report why."""
    df = (REPO / "daily_finetune.py").read_text()
    check("--deadline-epoch arg exists", '"--deadline-epoch"' in df)
    check("deadline reached stops training", "time.time() > self.deadline_epoch" in df)
    check("stop reason recorded", 'self.stop_reason = "session-deadline"' in df)
    check("metrics carry stop_reason", '"stop_reason": budget.stop_reason or "epoch-complete"' in df)
    check("train() takes deadline_epoch", "drive, run_folder, deadline_epoch: float = 0.0" in df)
    check("train() called with it", "args.deadline_epoch)" in df)

    # Behavioural: the callback stops on the wall even with an unlimited budget.
    import time as _t

    class _State:
        global_step = 7

    class _Ctl:
        should_training_stop = False

    class _Args:
        pass

    # Disable the expensive/foreign imports by exercising the callback logic only.
    ns = {}
    src = df[df.index("    class TimeBudgetCallback"):df.index("    class LossCurveCallback")]
    body = "\n".join(l[4:] if l.startswith("    ") else l for l in src.splitlines())
    body = body.replace("TrainerCallback", "object")
    exec(compile(body, "timebudget", "exec"), ns)
    ns.setdefault("time", _t)
    cb = ns["TimeBudgetCallback"](0, _t.time() - 1)  # no training budget, past wall
    ctl = _Ctl()
    cb.on_step_end(_Args(), _State(), ctl)
    check("wall reached -> should_training_stop", ctl.should_training_stop is True)
    check("wall reached -> timed_out (partial=True downstream)", cb.timed_out is True)
    check("wall reached -> reason", cb.stop_reason == "session-deadline", cb.stop_reason)

    cb2 = ns["TimeBudgetCallback"](0, _t.time() + 3600)
    ctl2 = _Ctl()
    cb2.on_step_end(_Args(), _State(), ctl2)
    check("future wall -> keeps training", ctl2.should_training_stop is False)
    check("future wall -> not partial", cb2.timed_out is False)

    cb3 = ns["TimeBudgetCallback"](0.001, _t.time() + 3600)
    _t.sleep(0.01)
    ctl3 = _Ctl()
    cb3.on_step_end(_Args(), _State(), ctl3)
    check("training budget still works", ctl3.should_training_stop is True and cb3.stop_reason == "budget")


def test_workflow_exposes_the_knob():
    wf = (REPO / ".github/workflows/daily-egtkg.yml").read_text()
    check("workflow input wall_minutes", "wall_minutes:" in wf)
    check("workflow env RUN_WALL_MINUTES",
          "RUN_WALL_MINUTES: ${{ github.event.inputs.wall_minutes || '120' }}" in wf)
    # a schedule event carries no inputs: the || fallback must exist for BOTH the
    # workflow env and the input default, or cron runs get an empty wall
    check("workflow input default 120", re.search(r"wall_minutes:.*?default: \"120\"", wf, re.S) is not None)
    check("runner logs the wall for diagnosis", "session wall budget" in (REPO / "run_daily.py").read_text())
    check("VM logs remaining wall at training start",
          "min of training left" in (REPO / "daily_finetune.py").read_text())


if __name__ == "__main__":
    test_wall_defaults_under_the_observed_recycle()
    test_launcher_passes_the_deadline()
    test_poll_deadline_is_wall_anchored()
    test_vm_stops_on_the_deadline()
    test_workflow_exposes_the_knob()
    print()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
