#!/usr/bin/env python3
"""Regression tests for run_daily.py stall/liveness + recovery grace logic.

Background (the bug these lock in): `colab logs` returns rc=1 from ~60 min
onward (token lifetime) while the VM keeps training. The old log-only stall rule
declared HEARTBEAT_STALLED at ~75-82 min every night and ran `colab stop` on a
HEALTHY VM (proven from Drive: 2026-09-10 runner stopped 19:08, VM pushed
step-100 at 19:16). Recovery then copied the OLDER step-50 into adapter_in.

Runs offline: no Colab, no Drive, no network.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_daily as rd  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def _noop_colab(*a, **k):
    return (0, "", "")


def test_parse_iso_and_age():
    ts = rd._parse_iso("2026-09-10T19:16:03.032Z")
    check("_parse_iso epoch", abs(ts - 1789067763.032) < 2, f"got {ts}")
    check("_parse_iso junk -> None", rd._parse_iso("nope") is None)
    check("_age(None) -> -1", rd._age(None) == -1)
    check("_age(recent) small", rd._age(time.time() - 5) in (4, 5, 6))


def test_log_dark_but_drive_fresh_is_not_a_stall():
    """The exact nightly failure: log rc=1, Drive checkpoints still landing."""
    rd.HEARTBEAT_STALL_S = 6        # seconds (test-speed stand-in for 45 min)
    rd.RECOVER_POLL_S = 1
    rd.POLL_INTERVAL_S = 1
    rd.POLL_INTERVAL_DARK_S = 1
    rd.colab = _noop_colab
    rd.refresh_colab_token = lambda force=False: True
    calls = {"n": 0}

    def fake_logs(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            return (0, "[alive] step=50 loss=0.4592 ram=9.1/12.7GB\n", "")
        # log channel goes dark, exactly like the ~60-min token expiry
        return (1, '{"ok": false, "err": "auth-expired", "msg": "token expired"}', "")

    rd.colab = lambda *a, **k: fake_logs()

    def fresh_drive(folder, run_id):
        return time.time() - 5   # a checkpoint 5s ago = VM alive

    rd.drive_run_activity = fresh_drive
    out = rd.poll_log(time.time() + 20, results_folder="F", run_id="R")
    check("log dark + Drive fresh -> NO stall", out is None, f"got {out!r}")

    # Drive also goes quiet -> now it IS a stall (both channels dead)
    calls["n"] = 0
    rd._LAST_DRIVE_AT = None
    rd.drive_run_activity = lambda folder, run_id: time.time() - 3600
    out = rd.poll_log(time.time() + 40, results_folder="F", run_id="R")
    check("log dark + Drive stale -> stall", out == "HEARTBEAT_STALLED", f"got {out!r}")


def test_recovery_waits_for_newer_artifact():
    """Recovery must not copy a stale checkpoint while a newer push is landing."""
    rd.RECOVER_POLL_S = 1
    rd.colab = _noop_colab
    rd.gdrive = lambda *a, **k: (0, "{}", "")
    seq = {"i": 0}

    def advancing_drive(folder, run_id):
        seq["i"] += 1
        return (time.time() - 600) if seq["i"] < 3 else (time.time() + 10)

    rd.drive_run_activity = advancing_drive
    uploads = []

    class FakeDrive:
        FOLDER_MIME = "application/vnd.google-apps.folder"

        def __init__(self, *a, **k):
            pass

        def list_files(self, folder):
            return []

        def download(self, fid, path):
            with open(path, "wb") as f:
                f.write(b"x" * 16)
            return {"ok": True}

    rd.Drive = FakeDrive
    rd.find_latest_checkpoint = lambda d, r, run: ("folder", 100, "fid", "step-100-adapter_model.safetensors")
    os.environ["DRIVE_RESULTS"] = "F"

    def fake_gdrive(*a, **k):
        uploads.append(list(a))
        return (0, json.dumps({"ok": True, "items": []}), "")

    orig_gdrive = rd.gdrive
    rd.gdrive = fake_gdrive
    t0 = time.time()
    rd.recover_latest_checkpoint_to_adapter_in("IN", "R", baseline_activity=time.time() - 600, wait_s=20)
    check("recovery waited for the newer artifact", seq["i"] >= 3, f"polls={seq['i']}")
    named = [u for u in uploads if "adapter_model.safetensors" in u]
    check("recovery copied adapter to adapter_in", bool(named), f"{uploads}")
    check("recovery wait bounded", time.time() - t0 < 25)
    rd.gdrive = orig_gdrive


def test_stall_window_exceeds_checkpoint_cadence():
    """The window must be longer than the ~25-35 min checkpoint cadence."""
    os.environ.pop("RUN_STALL_MINUTES", None)
    import importlib
    importlib.reload(rd)
    check("default stall window >= 40 min", rd.HEARTBEAT_STALL_S >= 40 * 60,
          f"{rd.HEARTBEAT_STALL_S}s")
    check("recovery grace default >= 10 min", rd.RECOVER_WAIT_S >= 10 * 60,
          f"{rd.RECOVER_WAIT_S}s")


def test_drive_result_json_is_the_success_signal():
    """Log channel dead + result.json on Drive = SUCCESS (not a timeout)."""
    rd.HEARTBEAT_STALL_S = 999
    rd.POLL_INTERVAL_S = 1
    rd.POLL_INTERVAL_DARK_S = 1
    rd.RECOVER_POLL_S = 1
    rd.refresh_colab_token = lambda force=False: True
    rd.colab = lambda *a, **k: (1, '{"ok": false, "err": "auth-expired"}', "")
    rd.drive_run_activity = lambda folder, run_id: time.time() - 5
    rd.drive_run_result = lambda folder, run_id: (True, {"ok": True, "steps": 150, "loss": 0.44})
    rd._RESULT_VIA_DRIVE = False
    out = rd.poll_log(time.time() + 20, results_folder="F", run_id="R")
    check("Drive result.json -> [RESULT] ok=true", "[RESULT] ok=true" in (out or ""), f"{out!r}")
    check("Drive result path flagged", rd._RESULT_VIA_DRIVE is True)
    check("training_succeeded() accepts it", rd.training_succeeded(out) is True)

    # ok=false on Drive must NOT be reported as success
    rd.drive_run_result = lambda folder, run_id: (False, {"ok": False, "steps": 10})
    rd._RESULT_VIA_DRIVE = False
    out = rd.poll_log(time.time() + 20, results_folder="F", run_id="R")
    check("Drive result ok=false -> not success", rd.training_succeeded(out) is False, f"{out!r}")


def test_adaptive_deadline_covers_slow_setup():
    """Setup (~30 min) must not eat the training window (run 34624713317 timed
    out 4 min before the VM pushed result.json)."""
    rd.POLL_INTERVAL_S = 1
    rd.POLL_INTERVAL_DARK_S = 1
    rd.HEARTBEAT_STALL_S = 999
    rd.refresh_colab_token = lambda force=False: True
    state = {"n": 0}

    def logs(*a, **k):
        state["n"] += 1
        # heartbeat only on the 3rd poll, then the log channel dies
        if state["n"] == 3:
            return (0, "[alive] step=10 loss=0.5 ram=9/12GB", "")
        return (1, '{"ok": false, "err": "not-found", "msg": "Not found."}', "")

    rd.colab = logs
    rd.drive_run_activity = lambda folder, run_id: time.time() - 5
    rd.drive_run_result = lambda folder, run_id: None
    t0 = time.time()
    # caller budget expires in 2s (stand-in for a budget eaten by setup);
    # training budget of 6s must extend it past that
    out = rd.poll_log(t0 + 2, results_folder="F", run_id="R", train_budget_s=6)
    elapsed = time.time() - t0
    check("deadline extended past the caller budget", elapsed > 3, f"elapsed={elapsed:.1f}s")
    check("adaptive poll ended by the extended deadline", out is None, f"{out!r}")


def test_settle_reports_success_when_vm_finished():
    """A timeout/stall must check Drive BEFORE overwriting adapter_in."""
    rd.RECOVER_POLL_S = 1
    calls = {"stop": 0, "recover": 0}
    rd.colab = lambda *a, **k: (calls.__setitem__("stop", calls["stop"] + 1) or (0, "stopped", ""))
    rd._stop_keep_alive = lambda ev: None
    rd.wait_for_drive_result = lambda folder, run_id, wait_s: (True, {"ok": True, "steps": 194, "loss": 0.44})

    def boom(*a, **k):
        calls["recover"] += 1
        raise AssertionError("recovery must NOT run when result.json ok=true")

    rd.recover_latest_checkpoint_to_adapter_in = boom
    ok = rd.finish_incomplete("IN", "OUT", "R", None, "training timed out")
    check("settle → SUCCESS when VM finished", ok is True)
    check("settle → no adapter_in overwrite", calls["recover"] == 0)
    check("settle → session stopped", calls["stop"] >= 1)
    check("settle → out/metrics.json written", os.path.exists("out/metrics.json"))

    # no result.json on Drive → real failure path (recover + raise)
    rd.wait_for_drive_result = lambda folder, run_id, wait_s: None
    rd.recover_latest_checkpoint_to_adapter_in = lambda *a, **k: calls.__setitem__("recover", calls["recover"] + 1)
    raised = False
    try:
        rd.finish_incomplete("IN", "OUT", "R", None, "training timed out")
    except SystemExit:
        raised = True
    check("no result.json → SystemExit failure", raised is True)
    check("no result.json → checkpoint recovered", calls["recover"] == 1)


if __name__ == "__main__":
    test_parse_iso_and_age()
    test_log_dark_but_drive_fresh_is_not_a_stall()
    test_recovery_waits_for_newer_artifact()
    test_stall_window_exceeds_checkpoint_cadence()
    test_drive_result_json_is_the_success_signal()
    test_adaptive_deadline_covers_slow_setup()
    test_settle_reports_success_when_vm_finished()
    print()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
