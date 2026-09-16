#!/usr/bin/env python3
"""
eval_run.py — VM-side orchestrator for the held-out base-vs-adapter A/B gate.

Runs on the free Colab T4 and performs the whole evaluation in one session:

  1. pull the training `knowledge_store.jsonl` from Drive (what the adapter was
     trained on — the held-out set is defined relative to it);
  2. fresh PubMed ingest into a THROWAWAY store (never touch the training store);
  3. `eval_build.py` keeps only articles whose PMID is absent from the training
     store and builds QA rows from that slice;
  4. `eval_ab.py` generates every answer twice — adapter ON vs
     `model.disable_adapter()` OFF — greedy, same prompt;
  5. pushes `eval_rows.jsonl` + `ab_results.jsonl` + `eval_done.json` into
     `results/ab-eval-<date>/` on Drive.

The runner (`run_eval.py`) scores the downloaded results locally with
`ab_score.py`; the blinded Gemini judge (`judge_ab.py`) is run separately from a
machine with live Gemini web cookies, because Gemini web auth is rejected from
GitHub runner IPs (consent page). See AGENTS.md.

Stopping is wall-bounded like training: `--deadline-epoch` (runner passes
session-create + budget) makes `eval_ab.py` finalize a partial comparison set
rather than losing the session to a silent Colab recycle.
"""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

PY = sys.executable


def run(cmd, timeout=None):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    try:
        r = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[eval_run] TIMEOUT after {timeout}s: {' '.join(map(str, cmd))}", flush=True)
        return 1
    return r.returncode


def run_tee(cmd, log_path, timeout=None):
    """Run a child, echoing its output live AND capturing it to log_path.

    The captured file is pushed to Drive with the result marker: a failure that
    only exists in the VM log is undiagnosable once the session is recycled
    (learned the hard way — the first eval failed with 'rc=1' and no reason).
    """
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    with open(log_path, "w") as lf:
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            lf.write(f"spawn failed: {e}\n")
            return 1
        if p.stdout is None:
            lf.write("no stdout pipe\n")
            return 1
        try:
            for line in p.stdout:
                print(line, end="", flush=True)
                lf.write(line)
                lf.flush()
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            lf.write(f"\nTIMEOUT after {timeout}s\n")
            print(f"[eval_run] TIMEOUT after {timeout}s", flush=True)
            return 1
    return p.returncode


def _count_lines(path):
    try:
        with open(path) as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def build_launch_report(args, rows_built, compared, partial, wall_hit, err):
    return {
        "ok": err is None,
        "err": err,
        "run_id": args.run_id,
        "adapter_folder": args.adapter_folder,
        "rows_built": rows_built,
        "compared": compared,
        "partial": partial,
        "wall_hit": wall_hit,
        "deadline_epoch": args.deadline_epoch,
        "days": args.days,
        "limit": args.limit,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdrive-py", required=True, help="vendored gdrive.py on the VM")
    ap.add_argument("--adc-file", required=True, help="Drive ADC json on the VM")
    ap.add_argument("--results-folder", required=True, help="Drive results root folder id")
    ap.add_argument("--adapter-folder", required=True,
                    help="Drive folder id holding the adapter to evaluate (adapter_in)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=80, help="held-out rows to compare")
    ap.add_argument("--days", type=int, default=60, help="fresh ingest window (held-out pool)")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--deadline-epoch", type=float, default=0.0)
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--out-dir", default="/content")
    args = ap.parse_args()

    sys.path.insert(0, "/content")
    from daily_finetune import Drive, _latest_store_local  # noqa: E402

    drive = Drive(args.gdrive_py, args.adc_file)
    date = args.run_id[:4] + "-" + args.run_id[4:6] + "-" + args.run_id[6:8]
    folder_name = f"ab-eval-{date}"
    store = os.path.join(args.out_dir, "kn_store.jsonl")
    fresh = os.path.join(args.out_dir, "fresh_eval.jsonl")
    rows_path = os.path.join(args.out_dir, "eval_rows.jsonl")
    results_path = os.path.join(args.out_dir, "ab_results.jsonl")

    # 1. training store (defines held-out)
    print(f"[eval_run] pulling training knowledge store from Drive", flush=True)
    _latest_store_local(drive, args.results_folder, store)
    if not os.path.exists(store) or os.path.getsize(store) == 0:
        return _finish(drive, args, folder_name, build_launch_report(
            args, 0, 0, False, False, "no training store on Drive — cannot define held-out set"))

    # 2. fresh ingest into a throwaway store
    rc = run([PY, "/content/pubmed_ingest.py", "--days", str(args.days),
              "--store", os.path.join(args.out_dir, "eval_scratch_store.jsonl"),
              "--out", fresh, "--max-oa", "0"], timeout=3600)
    if rc != 0:
        return _finish(drive, args, folder_name, build_launch_report(
            args, 0, 0, False, False, f"fresh ingest failed (rc={rc})"))

    # 3. held-out rows
    rc = run([PY, "/content/eval_build.py", "--train-store", store, "--fresh", fresh,
              "--out", rows_path, "--limit", str(args.limit),
              "--build-py", "/content/egtkg_build.py"], timeout=1800)
    rows_built = _count_lines(rows_path)
    if rc != 0 or rows_built == 0:
        return _finish(drive, args, folder_name, build_launch_report(
            args, rows_built, 0, False, False,
            f"eval_build produced {rows_built} rows (rc={rc}) — nothing held-out to evaluate"))

    # 4. A/B generation (wall-bounded). Output is tee'd to a file that ships with
    #    the marker, so a failure is diagnosable after the VM is gone.
    if args.deadline_epoch:
        left = (args.deadline_epoch - time.time()) / 60
        print(f"[eval_run] session wall: {left:.1f} min left for the {rows_built}-row comparison",
              flush=True)
    ab_log = os.path.join(args.out_dir, "eval_ab.log")
    rc = run_tee([PY, "/content/eval_ab.py", "--rows", rows_path, "--out", results_path,
                  "--adapter-parent", args.adapter_folder, "--adc", args.adc_file,
                  "--max-new-tokens", str(args.max_new_tokens),
                  "--deadline-epoch", str(args.deadline_epoch),
                  ], ab_log, timeout=10800)
    compared = _count_lines(results_path)
    if compared == 0:
        tail = ""
        try:
            tail = pathlib.Path(ab_log).read_text()[-600:]
        except Exception:
            pass
        return _finish(drive, args, folder_name, build_launch_report(
            args, rows_built, 0, False, False,
            f"eval_ab produced 0 comparisons (rc={rc}); log tail: {tail.strip()[-500:]}"),
            extra_files=[ab_log])
    partial = compared < rows_built

    # 5. publish to Drive (runner watches this folder for eval_done.json)
    folder = drive.ensure_folder(args.results_folder, folder_name)
    if not folder:
        print("[eval_run] could not create the Drive result folder", flush=True)
        return 1
    for local, name in ((results_path, "ab_results.jsonl"), (rows_path, "eval_rows.jsonl"),
                        (os.path.join(args.out_dir, "eval_ab.log"), "eval_ab.log")):
        if not os.path.exists(local):
            continue
        r = drive.upload(local, folder, name)
        if not drive.is_ok(r):
            print(f"[eval_run] upload {name} failed: {str(r)[:200]}", flush=True)
    rep = build_launch_report(args, rows_built, compared, partial, partial, None)
    tmp = os.path.join(args.out_dir, "eval_done.json")
    pathlib.Path(tmp).write_text(json.dumps(rep, indent=2))
    drive.upload_replace(tmp, folder, "eval_done.json")
    print(f"[EVALRESULT] ok=true rows={compared}/{rows_built} partial={partial} folder={folder_name}",
          flush=True)
    return 0


def _finish(drive, args, folder_name, report, extra_files=None):
    """Push a FAILURE marker so the runner stops waiting instead of hitting its
    deadline (the log channel dies ~60 min into a session, so stdout alone is
    not a reliable channel). Anything in extra_files ships alongside it — a
    failure with no captured log is undiagnosable after the VM is recycled."""
    try:
        folder = drive.ensure_folder(args.results_folder, folder_name)
        if folder:
            for p in (extra_files or []):
                try:
                    if os.path.exists(p):
                        drive.upload(p, folder, os.path.basename(p))
                except Exception as e:
                    print(f"[eval_run] could not upload {p}: {str(e)[:160]}", flush=True)
            tmp = os.path.join(args.out_dir, "eval_done.json")
            pathlib.Path(tmp).write_text(json.dumps(report, indent=2))
            drive.upload_replace(tmp, folder, "eval_done.json")
    except Exception as e:
        print(f"[eval_run] could not publish failure marker: {str(e)[:200]}", flush=True)
    print(f"[EVALRESULT] ok=false err={report.get('err')!r}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
