#!/usr/bin/env python3
"""
run_eval.py — GitHub-Actions runner for the held-out base-vs-adapter A/B gate.

The gate that decides whether the adapter may be frozen/deployed. It drives the
same free-Colab machinery as `run_daily.py` (session create with quota retries,
own keep-alive loop, wall budget) but runs the EVALUATION instead of training:

    Colab VM:  eval_run.py  →  fresh ingest → held-out rows → A/B generate
                             → Drive `results/ab-eval-<date>/`
    Runner:    download ab_results.jsonl → ab_score.py (mechanical scoring)

The blinded Gemini judge (`judge_ab.py`) is intentionally NOT run here: Gemini web
auth needs a browser session cookie and runner IPs get the consent page, so the
judge runs from the operator's machine against the same `ab_results.jsonl`.
See AGENTS.md ("The A/B eval is the gate").

Environment (same secrets as the daily workflow):
  COLAB_CLIENT_ID / COLAB_CLIENT_SECRET / COLAB_REFRESH_TOKEN
  GDRIVE_ADC, DRIVE_ADAPTER_IN, DRIVE_RESULTS
  EVAL_LIMIT          held-out rows (default 80)
  EVAL_DAYS           fresh ingest window (default 60)
  EVAL_MAX_NEW_TOKENS default 192
  EVAL_WALL_MINUTES   absolute session wall (default 100) — the eval stops there
                      and finalizes a partial comparison set
  EVAL_SESSION        Colab session name (default gi-egtkg-eval)
"""
import datetime
import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_daily as rd  # noqa: E402

SESSION = os.environ.get("EVAL_SESSION", "gi-egtkg-eval")
WALL_MINUTES = int(float(os.environ.get("EVAL_WALL_MINUTES", "100") or 0))
FINALIZE_SLACK_MIN = 20
POLL_S = int(float(os.environ.get("EVAL_POLL_S", "60")))


def drive_folder_by_name(parent, name):
    rc, out, err = rd.gdrive("list", "--folder", parent, "--max", "100", timeout=180)
    try:
        items = json.loads(out).get("items", [])
    except Exception:
        return None
    for it in items:
        if it.get("n") == name and it.get("m", "").endswith("folder"):
            return it.get("id")
    return None


def drive_files(folder):
    rc, out, err = rd.gdrive("list", "--folder", folder, "--max", "50", timeout=180)
    try:
        return {it.get("n"): it.get("id") for it in json.loads(out).get("items", [])}
    except Exception:
        return {}


def main():
    cid = os.environ["COLAB_CLIENT_ID"]
    csec = os.environ["COLAB_CLIENT_SECRET"]
    cref = os.environ["COLAB_REFRESH_TOKEN"]
    adc = os.environ["GDRIVE_ADC"]
    folder_in = os.environ["DRIVE_ADAPTER_IN"]
    folder_out = os.environ["DRIVE_RESULTS"]
    limit = os.environ.get("EVAL_LIMIT", "80")
    days = os.environ.get("EVAL_DAYS", "60")
    max_new = os.environ.get("EVAL_MAX_NEW_TOKENS", "192")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    date = run_id[:8]
    folder_name = f"ab-eval-{date[:4]}-{date[4:6]}-{date[6:8]}"

    scopes = ["openid",
              "https://www.googleapis.com/auth/userinfo.profile",
              "https://www.googleapis.com/auth/userinfo.email",
              "https://www.googleapis.com/auth/cloud-platform",
              "https://www.googleapis.com/auth/colaboratory",
              "https://www.googleapis.com/auth/drive.file"]
    tok = rd.mint_access_token(cid, csec, cref)
    rd.write_token_file(cid, csec, cref, tok["access_token"], scopes,
                        int(tok.get("expires_in", 3599)) - 60)
    rd.write_adc_file(adc)
    rd.set_colab_creds(cid, csec, cref, scopes)

    rc, out, err = rd.gdrive("about", timeout=120)
    rd.log(f"gdrive about: rc={rc} {out[:160]}")

    # --- session (same quota-retry ladder as the daily run) ---
    rd.log("pre-session cleanup: recovering any orphaned assignment on this account")
    rc, out, err = rd.colab("recover", timeout=120)
    rd.log(f"recover: rc={rc} {out[-160:]}")
    rc, out, err = rd.colab("stop", "-s", SESSION, timeout=120)
    rd.log(f"pre-clean stop: rc={rc} {out[-160:]}")

    rc, out, err = 1, "", ""
    for attempt in range(4):
        rc, out, err = rd.colab("new", "-s", SESSION, "--gpu", "T4", timeout=300)
        rd.log(f"new session attempt {attempt + 1}: rc={rc} {out[-200:]} {err[-200:]}")
        if rc == 0:
            break
        combined = str(out) + str(err)
        if attempt < 3 and ("gpu-unavailable" in combined or "Precondition Failed" in combined
                            or "too many" in combined.lower()):
            rd.log("GPU quota / assignment contention — cleaning orphans and waiting 300s")
            rd.colab("recover", timeout=120)
            rd.colab("stop", "-s", SESSION, timeout=120)
            time.sleep(300)
        else:
            break
    if rc != 0:
        raise SystemExit(f"failed to create T4 session for the eval: {str(out)[-400:]}")

    session_created_at = time.time()
    wall_epoch = int(session_created_at + WALL_MINUTES * 60) if WALL_MINUTES > 0 else 0
    rd.log(f"eval wall budget {WALL_MINUTES} min — eval_run stops generating at "
           f"{datetime.datetime.fromtimestamp(wall_epoch, datetime.timezone.utc).isoformat()}"
           if wall_epoch else "eval wall budget disabled")
    keep_alive_stop = rd._start_keep_alive(SESSION)

    for local, remote in [
        (str(rd.REPO / "eval_run.py"), "/content/eval_run.py"),
        (str(rd.REPO / "eval_build.py"), "/content/eval_build.py"),
        (str(rd.REPO / "eval_ab.py"), "/content/eval_ab.py"),
        (str(rd.REPO / "pubmed_ingest.py"), "/content/pubmed_ingest.py"),
        (str(rd.REPO / "egtkg_build.py"), "/content/egtkg_build.py"),
        (str(rd.REPO / "daily_finetune.py"), "/content/daily_finetune.py"),
        (str(rd.REPO / "gdrive.py"), "/content/gdrive.py"),
    ]:
        rc, o, e = rd.colab("upload", "-s", SESSION, local, remote, timeout=180)
        if rc != 0:
            rd._stop_keep_alive(keep_alive_stop)
            rd.colab("stop", "-s", SESSION, timeout=120)
            raise SystemExit(f"upload {local} failed: {e[-400:]}")
    adc_local = pathlib.Path("/tmp/gdrive_adc_eval.json")
    adc_local.write_text(adc)
    rc, o, e = rd.colab("upload", "-s", SESSION, str(adc_local), "/content/gdrive_adc.json", timeout=180)
    if rc != 0:
        rd._stop_keep_alive(keep_alive_stop)
        raise SystemExit(f"upload ADC failed: {e[-400:]}")

    launcher = pathlib.Path("/tmp/launch_eval.py")
    launcher.write_text(f"""import subprocess, sys
cmd = [sys.executable, '/content/eval_run.py',
       '--gdrive-py', '/content/gdrive.py',
       '--adc-file', '/content/gdrive_adc.json',
       '--results-folder', '{folder_out}',
       '--adapter-folder', '{folder_in}',
       '--run-id', '{run_id}',
       '--limit', '{limit}',
       '--days', '{days}',
       '--max-new-tokens', '{max_new}',
       '--deadline-epoch', '{wall_epoch}',
       '--out-dir', '/content']
print('LAUNCH ' + ' '.join(cmd), flush=True)
r = subprocess.run(cmd)
print('EXIT ' + str(r.returncode), flush=True)
sys.exit(r.returncode)
""")
    rc, out, err = rd.colab("exec_detach", "-s", SESSION, "-f", str(launcher),
                            "--log", "/content/eval.log", timeout=300)
    rd.log(f"exec_detach: rc={rc} {out[-200:]} {err[-200:]}")
    if rc != 0:
        rd._stop_keep_alive(keep_alive_stop)
        rd.colab("stop", "-s", SESSION, timeout=120)
        raise SystemExit(f"exec_detach failed: {out[-400:] or err[-400:]}")

    # --- poll: VM log + Drive marker (the log channel dies ~60 min in) ---
    deadline = (wall_epoch + FINALIZE_SLACK_MIN * 60) if wall_epoch else time.time() + 3 * 3600
    rd.log(f"polling /content/eval.log + Drive {folder_name}/eval_done.json until "
           f"{datetime.datetime.fromtimestamp(deadline, datetime.timezone.utc).isoformat()}")
    log_txt, done_report, dark = "", None, 0
    while time.time() < deadline:
        rc, out, err = rd.colab("logs", "-s", SESSION, "/content/eval.log", "-n", "12", timeout=120)
        if rc != 0:
            dark += 1
            if dark % 5 == 0:
                rd.refresh_colab_token()
        else:
            log_txt = (out if isinstance(out, str) else (out or b"").decode("utf-8", "replace")) or log_txt
            if "[EVALRESULT]" in log_txt:
                # the VM prints this on success AND on failure — distinguish them,
                # otherwise a failed eval surfaces as "artifact missing" (which is
                # exactly how the first run was misreported).
                m = re.search(r"\[EVALRESULT\]\s*ok=(true|false)(.*)", log_txt)
                done_report = {"from": "log",
                               "ok": bool(m and m.group(1) == "true"),
                               "detail": (m.group(2).strip()[:400] if m else log_txt[-300:])}
                break
        fid = drive_folder_by_name(folder_out, folder_name)
        if fid:
            files = drive_files(fid)
            if "eval_done.json" in files:
                rc, out, err = rd.gdrive("download", files["eval_done.json"],
                                         "--out", "/tmp/eval_done.json", timeout=180)
                if rc == 0:
                    try:
                        done_report = json.loads(pathlib.Path("/tmp/eval_done.json").read_text())
                        done_report["from"] = "drive"
                    except Exception:
                        pass
                if done_report:
                    break
        time.sleep(POLL_S)

    rd._stop_keep_alive(keep_alive_stop)
    rd.colab("stop", "-s", SESSION, timeout=120)

    if not done_report:
        raise SystemExit("eval did not report completion (neither [EVALRESULT] nor eval_done.json)")

    out_dir = rd.REPO / "out"
    out_dir.mkdir(exist_ok=True)

    # Always resolve the Drive marker: it carries the structured reason (err +
    # partial/compared counts) that the log line only summarises.
    fid = drive_folder_by_name(folder_out, folder_name)
    files = drive_files(fid) if fid else {}
    if "eval_done.json" in files:
        rc, out, err = rd.gdrive("download", files["eval_done.json"],
                                 "--out", "/tmp/eval_done.json", timeout=180)
        if rc == 0:
            try:
                dr = json.loads(pathlib.Path("/tmp/eval_done.json").read_text())
                dr["from"] = "drive"
                done_report = dr
            except Exception:
                pass
    if done_report and str(done_report.get("ok")).lower() != "true":
        raise SystemExit(f"eval failed on the VM: "
                         f"{done_report.get('err') or done_report.get('detail')}")
    if not done_report:
        raise SystemExit("eval did not report completion (neither [EVALRESULT] nor eval_done.json)")
    if "ab_results.jsonl" not in files:
        raise SystemExit(f"eval finished but ab_results.jsonl is missing from Drive ({folder_name})")
    rc, out, err = rd.gdrive("download", files["ab_results.jsonl"],
                             "--out", str(out_dir / "ab_results.jsonl"), timeout=300)
    if rc != 0:
        raise SystemExit(f"download ab_results.jsonl failed: {err[-300:]}")
    for name in ("eval_rows.jsonl", "eval_done.json"):
        if name in files:
            rd.gdrive("download", files[name], "--out", str(out_dir / name), timeout=300)

    # --- mechanical scoring on the runner (no Gemini here) ---
    rc, out, err = rd.sh([sys.executable, str(rd.REPO / "ab_score.py"),
                          "--results", str(out_dir / "ab_results.jsonl"),
                          "--out", str(out_dir / "scored.jsonl")], timeout=600)
    if rc != 0:
        raise SystemExit(f"ab_score failed: {err[-400:]}")
    (out_dir / "scored_summary.json").write_text(out)
    print(out)
    rd.log(f"scoring done — reviewer: run judge_ab.py locally on "
           f"out/ab_results.jsonl for the blinded Gemini Pro pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
