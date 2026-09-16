#!/usr/bin/env python3
"""
run_daily.py — headless daily LoRA fine-tune orchestrator.

Runs inside GitHub Actions (no local machine involved). It:
  1. Mints a Colab access token from a stored refresh token (server-side OAuth,
     no browser).
  2. Writes the colab-cli token.json and the gdrive ADC credentials.
  3. Creates a free Colab T4 session, uploads daily_finetune.py + the vendored
     gdrive.py + the ADC JSON, and launches training detached.
  4. Polls the VM log until the run reports [RESULT] (or times out).
  5. On success: downloads the new adapter + metrics for the GH artifact tab
     (Drive continuity is already handled BY THE VM — see below).
  6. Stops the session.

DRIVE CONTINUITY (why timeouts no longer lose work):
  daily_finetune.py mounts Drive itself (vendored gdrive.py + the ADC JSON we
  upload) and pushes a checkpoint to Drive after every --save-steps training
  steps, plus a final checkpoint + archive + adapter_in update when the run
  ends (naturally or via its --max-minutes budget). So even if THIS runner
  times out or the Colab session is recycled mid-training, the newest
  checkpoint already lives on Drive and the next day's run resumes from it.
  On runner timeout we additionally pull the newest Drive checkpoint into the
  adapter_in folder so continuity is preserved even if the VM died before its
  final push.

Secrets come from environment variables (set by the workflow from GitHub
secrets). None are ever committed to the repo.

Environment:
  COLAB_CLIENT_ID         OAuth client id (gcloud "Desktop" OAuth client)
  COLAB_CLIENT_SECRET     OAuth client secret
  COLAB_REFRESH_TOKEN     long-lived refresh token (colaboratory + drive.file scope)
  GDRIVE_ADC              full gcloud ADC JSON (authorized_user) for Drive
  DRIVE_ADAPTER_IN        Drive folder id for the "latest" adapter (shared, gdown pulls it)
  DRIVE_RESULTS           Drive folder id for dated result archives + checkpoints/
  RUN_ROWS                finance-alpaca subset size per day (default 5000)
  RUN_SAVE_STEPS          checkpoint every N training steps (default 100)
  RUN_MAX_MINUTES         stop training after N minutes, save final checkpoint (default 240 — covers a full 2000-row epoch)
"""
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# run_daily.py lives in the same repo as daily_finetune.py — reuse its Drive
# wrapper + checkpoint discovery instead of duplicating Drive logic.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from daily_finetune import Drive, find_latest_checkpoint, find_latest_archive  # noqa: E402

TOKEN_FILE = pathlib.Path.home() / ".config" / "colab-cli" / "token.json"
ADC_FILE = pathlib.Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
REPO = pathlib.Path(__file__).resolve().parent
COLAB_PY = str(REPO / "colab.py")
GDRIVE_PY = str(REPO / "gdrive.py")
SESSION = "gi-egtkg-session"

# Colab free-tier sessions recycle after ~2-3h (some last up to 12h); the VM
# self-stops after RUN_MAX_MINUTES of training (default 100) so a full run fits
# well inside. The runner's poll budget must OUTLIVE the VM's setup (~12 min) +
# training, else the runner declares timeout while the VM is still training.
# Default: max_minutes + 25 (setup slack). Override via TRAIN_TIMEOUT_MIN.
# In UNLIMITED mode (max_minutes=0) there is no self-stop — the VM trains until
# Colab recycles it — so the runner observes a fixed window (default 240 min)
# to confirm checkpoints are flowing, then leaves the session running.
def train_timeout_s():
    ov = os.environ.get("TRAIN_TIMEOUT_MIN", "")
    if ov.strip():
        return int(float(ov)) * 60
    mm = int(os.environ.get("RUN_MAX_MINUTES", "100"))
    if mm <= 0:
        return 240 * 60
    return (mm + 25) * 60


TRAIN_TIMEOUT_S = train_timeout_s()

# WALL BUDGET — the VM's own clock, measured from SESSION CREATION.
#
# Google recycles free-tier T4 sessions without warning and the VM goes
# unreachable the moment it happens (its keep-alive endpoint starts returning
# HTTP 404). Observed: 2026-09-13 and 2026-09-14 both died at ~2h20m after
# creation (17:38:54→20:00:19 and 19:36:01→21:56:13), while 2026-09-11/12
# survived 3h+; the healthy nights happened to finish their epoch before the
# recycle, the two that did not were killed mid-epoch, and because the VM never
# reached its final save/archive the runner could only salvage the last
# periodic checkpoint and had to report FAILURE.
#
# So the VM is told an ABSOLUTE deadline (session create + RUN_WALL_MINUTES) and
# stops training there — it then still has minutes to verify, archive and push
# `result.json {ok:true, partial:true}`, which the runner accepts as SUCCESS.
# RUN_WALL_MINUTES must stay below the observed ~140 min recycle: 120 leaves
# ~20 min of finalize margin and still fits ~80 min of training after the
# ~35-40 min setup (ingest + install + model load).
WALL_MINUTES = int(float(os.environ.get("RUN_WALL_MINUTES", "120") or 0))

# Extra slack after the wall for the finalize pushes before the runner gives up.
WALL_FINALIZE_SLACK_MIN = 25

# Stall detection: a run is only declared DEAD when BOTH liveness channels go
# quiet for this long — the VM log ([alive] heartbeats) AND Drive (the VM pushes
# heartbeat.json every 10 steps plus a step-N checkpoint every RUN_SAVE_STEPS).
#
# Why both, and why 45 min (was 20 min, log-only):
#   `colab logs`/`exec` starts returning rc=1 at ~60 min after session create
#   (the colab-cli access token lives ~59 min; re-minting token.json does not
#   revive the exec path). The old log-only rule therefore declared
#   HEARTBEAT_STALLED at ~75-82 min every single night and ran `colab stop` on a
#   HEALTHY VM — proven from Drive: on 2026-09-10 the runner stopped the session
#   at 19:08 and the VM still pushed step-100 at 19:16; same on 2026-09-04
#   (step-150 at 19:09 after the stop). Worse, recovery then overwrote
#   adapter_in with the OLDER step-50 (step-100 did not exist yet) and the next
#   run resumed 50 steps behind.
#   Checkpoint cadence is ~25-35 min, so the window must exceed it.
HEARTBEAT_STALL_S = int(float(os.environ.get("RUN_STALL_MINUTES", "45"))) * 60

# After a stall is declared, give the VM this long to land one more Drive
# artifact before we overwrite adapter_in, so the newest checkpoint (not the one
# that happened to exist at stall time) is what the next run resumes from.
RECOVER_WAIT_S = int(float(os.environ.get("RUN_RECOVER_WAIT_MINUTES", "12"))) * 60
RECOVER_POLL_S = 30

# Poll cadence: slower while the log channel is healthy (each poll spawns a
# colab.py subprocess), faster once it goes dark and we are watching Drive.
POLL_INTERVAL_S = int(float(os.environ.get("RUN_POLL_INTERVAL_S", "90")))
POLL_INTERVAL_DARK_S = int(float(os.environ.get("RUN_POLL_INTERVAL_DARK_S", "60")))

# Last Drive artifact mtime (epoch) observed while polling — handed to recovery
# as the baseline so it can wait for a newer checkpoint instead of copying a
# stale one.
_LAST_DRIVE_AT = None

# Set when [RESULT] was read from Drive (log channel dead) instead of the VM
# log — the download step is then pointless (same dead channel) because the VM
# already pushed adapter_in + results-<date>/ itself.
_RESULT_VIA_DRIVE = False

# Colab idle-prunes free VM assignments with NO keep-alive ping. The colab-cli
# spawns its own keep-alive daemon at `colab new`, but that daemon caches the
# access token minted at startup (expires ~59 min) and dies with consecutive
# 4xx — which is EXACTLY the ~60-min VM death observed across runs #12-14 and
# the 2026-08-22 dispatch. We therefore run OUR OWN keep-alive loop inside the
# runner's poll window, re-minting a fresh token every ping. This is the
# deterministic fix for the "VM dies at step ~100 / 60 min" pattern.
KEEP_ALIVE_INTERVAL_S = 60
KEEP_ALIVE_URL = "https://colab.research.google.com/tun/m/{endpoint}/keep-alive/"
KEEP_ALIVE_HEADERS = {
    "X-Colab-Tunnel": "Google",
    "Accept": "application/json",
    "X-Colab-Client-Agent": "colab-cli",
}


def get_session_endpoint(session_name):
    """Read the assignment endpoint for a session from the colab-cli registry."""
    store_path = pathlib.Path.home() / ".config" / "colab-cli" / "sessions.json"
    try:
        store = json.loads(store_path.read_text())
        for name, s in store.items():
            if name == session_name and s.get("endpoint"):
                return s["endpoint"]
        # fall back: any entry whose name/endpoint matches the session
        for name, s in store.items():
            if s.get("endpoint") and (name == session_name or s.get("name") == session_name):
                return s["endpoint"]
    except Exception as e:
        log(f"could not read session registry for endpoint: {e}")
    return None


def keep_alive_loop(endpoint, stop_event, cadence=KEEP_ALIVE_INTERVAL_S):
    """Ping the Colab assignment keep-alive endpoint until stop_event is set.

    Each ping mints a FRESH access token (never caches) so the loop outlives
    the colab-cli daemon's token-expiry death. Read timeouts are expected
    (TFE records activity before forwarding); only HTTP errors are surfaced.
    """
    while not stop_event.is_set():
        try:
            tok = mint_access_token(
                os.environ.get("COLAB_CLIENT_ID", ""),
                os.environ.get("COLAB_CLIENT_SECRET", ""),
                os.environ.get("COLAB_REFRESH_TOKEN", ""),
            )
            token = tok.get("access_token", "")
            url = KEEP_ALIVE_URL.format(endpoint=endpoint)
            req = urllib.request.Request(
                url + ("&" if "?" in url else "?") + "authuser=0",
                headers={**KEEP_ALIVE_HEADERS, "Authorization": f"Bearer {token}"},
            )
            urllib.request.urlopen(req, timeout=10)
            log(f"keep-alive ping OK ({endpoint[:24]}...)")
        except TimeoutError:
            # Expected: TFE records the activity before forwarding to the VM,
            # which often doesn't answer on this path. A read timeout means the
            # keep-alive SUCCEEDED (this matches the official colab-cli client).
            log(f"keep-alive ping recorded (read timeout = success) ({endpoint[:24]}...)")
        except urllib.error.HTTPError as e:
            # Real errors (404 deleted assignment, 4xx/5xx) — surface but keep
            # trying; a few transient failures must not kill the loop.
            log(f"keep-alive ping HTTP {e.code} (continuing): {str(e)[:120]}")
        except Exception as e:
            log(f"keep-alive ping failed (transient, continuing): {str(e)[:150]}")
        stop_event.wait(cadence)


def _start_keep_alive(session_name):
    """Start the keep-alive thread for a session; returns the stop event or None."""
    endpoint = get_session_endpoint(session_name)
    if not endpoint:
        log("keep-alive: no endpoint found in session registry — skipping (VM may be pruned at ~60 min)")
        return None
    ev = threading.Event()
    t = threading.Thread(target=keep_alive_loop, args=(endpoint, ev), daemon=True)
    t.start()
    log(f"keep-alive loop started for endpoint {endpoint[:24]}... (ping every {KEEP_ALIVE_INTERVAL_S}s)")
    return ev


def _stop_keep_alive(stop_event):
    if stop_event is not None:
        stop_event.set()
        log("keep-alive loop stopped")


def log(msg):
    # Redact identifiers before ANY of this reaches a log: this repo is (or is
    # about to be) public, and GitHub only masks SECRET values — an account
    # email printed by `gdrive about` or a Drive id echoed by a helper would be
    # world-readable in the workflow log.
    print(f"[run_daily] {_redact(msg)}", flush=True)


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# gdrive.py `about` reports the Drive account's display name — an identifier we
# do not publish on a public repo's logs.
_ACCT_NAME_RE = re.compile(r'"name"\s*:\s*"[^"]*"')


def _redact(msg):
    try:
        s = str(msg)
        s = _EMAIL_RE.sub("<redacted-email>", s)
        return _ACCT_NAME_RE.sub('"name":"<redacted>"', s)
    except Exception:
        return str(msg)


def sh(args, timeout=180, check=False, ok_codes=(0,)):
    """Run a command, return (returncode, stdout, stderr)."""
    log("$ " + " ".join(str(a) for a in args))
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return (-1, (e.stdout or "") if isinstance(e.stdout, str) else "", "TIMEOUT")
    if check and r.returncode not in ok_codes:
        raise RuntimeError(f"command failed rc={r.returncode}: {args}\n{r.stderr[-2000:]}")
    return r.returncode, r.stdout, r.stderr


def mint_access_token(client_id, client_secret, refresh_token):
    body = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def write_token_file(client_id, client_secret, refresh_token, access_token, scopes, expiry_s):
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "token": access_token,
        "refresh_token": refresh_token,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": client_id,
        "client_secret": client_secret,
        "scopes": scopes,
        "universe_domain": "googleapis.com",
        "account": "",
        "expiry": (datetime.datetime.now(datetime.timezone.utc)
                   + datetime.timedelta(seconds=expiry_s)).isoformat(),
    }
    TOKEN_FILE.write_text(json.dumps(doc, indent=2))
    log(f"wrote {TOKEN_FILE}")


def write_adc_file(adc_json):
    ADC_FILE.parent.mkdir(parents=True, exist_ok=True)
    ADC_FILE.write_text(adc_json)
    ADC_FILE.chmod(0o600)
    log(f"wrote {ADC_FILE}")


# Credentials for in-process token refresh. Set by main() before any colab call.
_CREDS = {"cid": "", "csec": "", "cref": "", "scopes": []}
_last_token_refresh = 0.0


def set_colab_creds(cid, csec, cref, scopes):
    _CREDS.update({"cid": cid, "csec": csec, "cref": cref, "scopes": list(scopes)})


def refresh_colab_token(force=False):
    """Re-mint the colab-cli access token (token.json) so `colab logs`/`exec`
    keep working past the initial token's ~59-min expiry.

    The colab-cli daemon and the vendored colab.py cache the access token
    minted at startup (expires_in-60). When it expires, `colab logs` starts
    returning rc=1 even though the VM is ALIVE and training — the runner then
    falsely declares HEARTBEAT_STALLED and stops a healthy VM (verified live:
    run 32596123261 pushed step-200 AFTER the runner's 'recover step-100'
    because the log polls had died at 60 min). Re-mint the token file in
    process, cheaply, before each poll window.
    """
    global _last_token_refresh
    now = time.time()
    if not force and now - _last_token_refresh < 300:
        return True  # at most once per 5 min
    try:
        tok = mint_access_token(_CREDS["cid"], _CREDS["csec"], _CREDS["cref"])
        write_token_file(_CREDS["cid"], _CREDS["csec"], _CREDS["cref"],
                         tok["access_token"], _CREDS["scopes"],
                         int(tok.get("expires_in", 3599)) - 60)
        _last_token_refresh = now
        return True
    except Exception as e:
        log(f"token refresh failed (continuing with cached token): {str(e)[:150]}")
        return False


def gdrive(*args, timeout=300):
    return sh([sys.executable, GDRIVE_PY, *args], timeout=timeout)


def colab(*args, timeout=300):
    return sh([sys.executable, COLAB_PY, *args], timeout=timeout)


def _age(ts):
    """Seconds since ts, or -1 when unknown (never raises)."""
    return int(time.time() - ts) if ts else -1


def _parse_iso(ts):
    """ISO-8601 (Drive) -> epoch seconds, or None."""
    try:
        return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def drive_run_activity(results_folder, run_id):
    """Newest Drive artifact mtime (epoch) for THIS run's checkpoint folder.

    This is the runner's SECOND, out-of-band liveness channel. The VM pushes
    heartbeat.json every 10 steps + a step-N adapter every RUN_SAVE_STEPS into
    checkpoints/<run_id>/, so a fresh mtime there proves the VM is alive even
    when `colab logs` has gone dark. Returns None when the folder/files are not
    reachable (treated as "no information", never as proof of death).
    """
    try:
        drive = Drive(GDRIVE_PY, str(ADC_FILE))
        parent = drive.ensure_folder(results_folder, "checkpoints")
        if not parent:
            return None
        for folder in drive.list_files(parent):
            if not isinstance(folder, dict):
                continue
            if folder.get("m") != Drive.FOLDER_MIME or folder.get("n") != run_id:
                continue
            newest = None
            for f in drive.list_files(folder.get("id")):
                if not isinstance(f, dict):
                    continue
                ts = _parse_iso(f.get("t"))
                if ts and (newest is None or ts > newest):
                    newest = ts
            return newest
    except Exception as e:
        log(f"drive liveness probe failed: {str(e)[:150]}")
    return None


def drive_run_result(results_folder, run_id):
    """Read the run's Drive result.json — the out-of-band success signal.

    Returns (ok:bool, payload_dict) or None. Necessary because the VM log
    channel dies at ~60 min while training continues; without this the runner
    could never see [RESULT] again and every long run would end as a false
    TIMEOUT even though the adapter was archived successfully.
    """
    try:
        drive = Drive(GDRIVE_PY, str(ADC_FILE))
        parent = drive.ensure_folder(results_folder, "checkpoints")
        if not parent:
            return None
        for folder in drive.list_files(parent):
            if not isinstance(folder, dict) or folder.get("m") != Drive.FOLDER_MIME:
                continue
            if folder.get("n") != run_id:
                continue
            for f in drive.list_files(folder.get("id")):
                if not isinstance(f, dict) or f.get("n") != "result.json":
                    continue
                tmp = pathlib.Path("/tmp/drive_result.json")
                drive.download(f.get("id"), str(tmp))
                if tmp.exists() and tmp.stat().st_size > 0:
                    data = json.loads(tmp.read_text())
                    return bool(data.get("ok")), data
            return None
    except Exception as e:
        log(f"drive result probe failed: {str(e)[:150]}")
    return None


def wait_for_drive_result(results_folder, run_id, wait_s, poll_s=None):
    """Poll Drive for this run's result.json for up to wait_s. Returns
    (ok, payload) or None.

    Used as the SETTLE step before any recovery: a run that pushed
    result.json{ok:true} has fully finished (verify + archive + adapter_in all
    happened BEFORE that file was written), so it must not be reported as a
    failure, and adapter_in must not be rewritten from a mid-run checkpoint.
    Live evidence for why: run 34624713317 — the runner timed out at 18:36:29
    and recovered step-150 at 18:38:58, while the VM was finalizing and pushed
    result.json at 18:40:11.
    """
    if poll_s is None:
        poll_s = RECOVER_POLL_S
    t0 = time.time()
    while True:
        res = drive_run_result(results_folder, run_id)
        if res is not None:
            return res
        if time.time() - t0 >= wait_s:
            return None
        time.sleep(poll_s)


def poll_log(deadline, needle="[RESULT]", results_folder=None, run_id=None,
             train_budget_s=None):
    """Poll the VM log until the run finishes, or decide the VM is dead.

    Death is declared only when BOTH channels are stale for HEARTBEAT_STALL_S:
      * the newest [alive] heartbeat read from the VM log, and
      * the newest artifact mtime in this run's Drive checkpoint folder.
    The log channel alone is NOT sufficient — `colab logs`/`exec` starts
    returning {"err":"not-found"} at ~60 min while the VM keeps training, and a
    log-only rule killed a healthy VM every night.

    The poll deadline is also ADAPTIVE: the budget passed in is measured from
    exec_detach, but the VM spends ~30 min on pip installs, ingest and model load
    before step 1. Once training is first seen alive, the deadline becomes at
    least (now + train_budget_s), so a slow setup can no longer cut the training
    window short (run 34624713317 timed out 4 min before the VM finished).
    """
    last_alive_step = -1
    last_alive_at = None
    progress_at = None          # newest proof of life from either channel
    log_dead_since = None       # when the log channel first went unreadable
    training_started_at = None  # first heartbeat seen (either channel)
    recover_attempts = 0
    while time.time() < deadline:
        fresh = refresh_colab_token()
        rc, out, err = colab("logs", "-s", SESSION, "/content/train.log", "-n", "12", timeout=60)
        if rc != 0 and "auth-expired" in str(out or "") + str(err or ""):
            # Login died mid-poll: force a fresh token and retry once right away
            # instead of waiting a whole poll interval.
            log("log poll auth-expired — forcing token refresh and retrying once")
            refresh_colab_token(force=True)
            rc, out, err = colab("logs", "-s", SESSION, "/content/train.log", "-n", "12", timeout=60)
        if rc != 0 and ("not-found" in str(out or "") or "NOT FOUND" in str(out or "")) \
                and recover_attempts < 2:
            # The exec/log channel loses its session mapping (colab-cli reports
            # "not-found", not an auth error). `colab recover` rebuilds
            # sessions.json from the server-side assignment list — try it before
            # giving up on the primary channel entirely.
            recover_attempts += 1
            log(f"log channel reports not-found — running `colab recover` "
                f"(attempt {recover_attempts}) and retrying")
            colab("recover", timeout=120)
            rc, out, err = colab("logs", "-s", SESSION, "/content/train.log", "-n", "12", timeout=60)
        if needle in out or "EXIT" in out or "NOT FOUND" in out:
            return out
        now = time.time()

        if rc == 0:
            log_dead_since = None
        elif log_dead_since is None:
            log_dead_since = now
        # heartbeat channel
        import re as _re
        m = _re.findall(r"\[alive\] step=(\d+)", out)
        if m:
            cur = max(int(x) for x in m)
            if cur > last_alive_step:
                last_alive_step = cur
                last_alive_at = now
                progress_at = now
        # Drive channel — only probed while the log channel is dark (it costs a
        # couple of gdrive.py subprocesses per call, no point on every poll)
        drive_at = None
        if log_dead_since is not None and results_folder and run_id:
            drive_at = drive_run_activity(results_folder, run_id)
            if drive_at and (progress_at is None or drive_at > progress_at):
                progress_at = drive_at
            if drive_at:
                global _LAST_DRIVE_AT
                _LAST_DRIVE_AT = max(_LAST_DRIVE_AT or 0, drive_at)
            # completion signal on the Drive channel: the VM pushed result.json
            res = drive_run_result(results_folder, run_id)
            if res is not None:
                ok, payload = res
                global _RESULT_VIA_DRIVE
                _RESULT_VIA_DRIVE = True
                log(f"run finished — result.json on Drive: ok={ok} "
                    f"steps={payload.get('steps')} loss={payload.get('loss')} "
                    f"(log channel was dead, so [RESULT] came from Drive)")
                return (f"[RESULT] ok={'true' if ok else 'false'} (via Drive result.json) "
                        f"{json.dumps(payload)}")
        # once training is seen alive, the deadline must cover the TRAINING
        # budget from here, not from exec_detach — setup (pip installs, PubMed
        # ingest, model load) eats ~30 min before step 1 and does not count
        # toward --max-minutes on the VM.
        if progress_at is not None and training_started_at is None:
            training_started_at = progress_at
            if train_budget_s:
                extended = training_started_at + train_budget_s
                if extended > deadline:
                    deadline = extended
                    log(f"training started — poll deadline extended to cover "
                        f"{int(train_budget_s)}s of training "
                        f"({int((deadline - now) / 60)} min left)")
        # auth-expired or any rc!=0: surface stdout too — colab.py `die()` writes
        # {"ok":false,"err":...} to STDOUT, so logging stderr alone hid the real
        # cause for 8 straight nights.
        if rc != 0:
            log(f"log poll rc={rc} (log channel dark {_age(log_dead_since)}s): "
                f"out={out[-300:]!r} err={err[-200:]!r}"
                + ("" if fresh else " (token refresh failed)"))
        # stall verdict
        if progress_at is not None and now - progress_at > HEARTBEAT_STALL_S:
            log(f"HEARTBEAT STALLED: no progress on EITHER channel for "
                f"{_age(progress_at)}s (log dead {_age(log_dead_since)}s, "
                f"last [alive] step={last_alive_step}, drive_at={drive_at}) — VM dead")
            return "HEARTBEAT_STALLED"
        if progress_at is None and log_dead_since is not None \
                and now - log_dead_since > HEARTBEAT_STALL_S:
            # never saw a single heartbeat on any channel AND the log is dark
            log(f"HEARTBEAT STALLED: no heartbeat ever seen on either channel "
                f"and log unreadable for {_age(log_dead_since)}s")
            return "HEARTBEAT_STALLED"
        time.sleep(POLL_INTERVAL_S if log_dead_since is None else POLL_INTERVAL_DARK_S)
    return None


def training_succeeded(final_log):
    """A run is a SUCCESS only if the VM reported [RESULT] ok=true — the marker
    daily_finetune.py prints AFTER the adapter passed verify_adapter() and the
    Drive pushes completed. A bare 'EXIT 1' (e.g. a callback crash) must NOT be
    treated as success; the old code did exactly that and shipped empty runs."""
    if not final_log:
        return False
    return "[RESULT] ok=true" in final_log or "[RESULT] ok=True" in final_log


def recover_latest_checkpoint_to_adapter_in(folder_in, run_id, baseline_activity=None,
                                            wait_s=None):
    """Pull the newest Drive checkpoint into the adapter_in continuity folder.
    Called on runner timeout / VM death so the next run resumes from the last
    saved checkpoint instead of the pre-run adapter.

    GRACE WAIT (why): the VM keeps training for a while after the runner gives
    up on a session — on 2026-09-10 the runner stopped the session at 19:08 and
    the VM pushed step-100 at 19:16, but recovery had already copied the older
    step-50 into adapter_in, so the next run resumed 50 steps BEHIND. When we
    know the last Drive activity seen while polling (baseline_activity), wait up
    to wait_s for a NEWER artifact before copying.
    """
    if wait_s is None:
        wait_s = RECOVER_WAIT_S
    if baseline_activity:
        t0 = time.time()
        while time.time() - t0 < wait_s:
            act = drive_run_activity(os.environ.get("DRIVE_RESULTS", ""), run_id)
            if act and act > baseline_activity + 5:
                log(f"recovery: newer Drive artifact after {int(time.time()-t0)}s "
                    f"(activity advanced) — using the newest checkpoint")
                break
            time.sleep(RECOVER_POLL_S)
        else:
            log(f"recovery: no newer Drive artifact within {wait_s}s — "
                f"using the newest available checkpoint")
    try:
        drive = Drive(GDRIVE_PY, str(ADC_FILE))
        found = find_latest_checkpoint(drive, os.environ.get("DRIVE_RESULTS", ""), run_id)
        source = "checkpoint"
        if not found:
            arch = find_latest_archive(drive, os.environ.get("DRIVE_RESULTS", ""))
            if arch:
                folder_id, file_id, file_name, date = arch
                found = (folder_id, 0, file_id, file_name)
                source = f"archive results-{date}"
        if not found:
            log("no Drive checkpoint or archive found to recover")
            return False
        folder_id, step, file_id, file_name = found
        tmp = pathlib.Path("/tmp/ckpt_recover")
        tmp.mkdir(exist_ok=True)
        adapter = tmp / "adapter_model.safetensors"
        drive.download(file_id, str(adapter))
        cfg = tmp / "adapter_config.json"
        if not cfg.exists():
            for f in drive.list_files(folder_id):
                if f.get("n") == "adapter_config.json":
                    drive.download(f.get("id"), str(cfg))
                    break
        if not (adapter.exists() and adapter.stat().st_size > 0):
            log("recovered adapter file is empty/missing")
            return False
        # replace adapter_in contents — purge by NAME in a loop: Drive listings
        # are eventually consistent, so one delete pass leaves stale adapter
        # copies behind (adapter_in had accumulated 11 pairs by 2026-09-11,
        # which makes gdown --folder consumers pick an arbitrary old adapter).
        for _ in range(3):
            rc, o, e = gdrive("list", "--folder", folder_in, "--max", "200", timeout=120)
            items = []
            try:
                data = json.loads(o or "{}")
                items = data.get("items", []) if isinstance(data, dict) and isinstance(data.get("items"), list) else []
            except Exception:
                items = []
            stale = [it for it in items if isinstance(it, dict) and it.get("id")
                     and it.get("n") in ("adapter_model.safetensors", "adapter_config.json")]
            for it in stale:
                gdrive("rm", it["id"], timeout=120)
            if not stale:
                break
        gdrive("upload", str(adapter), "--parent", folder_in, "--name", "adapter_model.safetensors", timeout=300)
        if cfg.exists():
            gdrive("upload", str(cfg), "--parent", folder_in, "--name", "adapter_config.json", timeout=120)
        log(f"recovered {source} step-{step} -> adapter_in (next run resumes from it)")
        return True
    except Exception as e:
        log(f"checkpoint recovery failed: {e}")
        return False


def finish_incomplete(folder_in, folder_out, run_id, keep_alive_stop, reason, detail=""):
    """Terminal handling for any non-[RESULT] exit: settle, then recover.

    1. SETTLE: give the VM up to RECOVER_WAIT_S to push result.json. If it says
       ok=true the run actually COMPLETED (verify + archive + adapter_in all ran
       before that file was written) — the log channel was just dead, so report
       SUCCESS and DO NOT touch adapter_in.
    2. Otherwise recover the newest checkpoint (with its own grace wait) and fail
       loudly with `reason`.
    """
    log(f"settling on Drive for up to {RECOVER_WAIT_S}s before recovery ({reason})")
    res = wait_for_drive_result(folder_out, run_id, RECOVER_WAIT_S)
    if res and res[0]:
        payload = res[1]
        outdir = pathlib.Path("out")
        outdir.mkdir(exist_ok=True)
        (outdir / "metrics.json").write_text(json.dumps(payload, indent=2))
        _stop_keep_alive(keep_alive_stop)
        colab("stop", "-s", SESSION, timeout=120)
        log(f"run actually COMPLETED (result.json on Drive): steps={payload.get('steps')} "
            f"loss={payload.get('loss')} partial={payload.get('partial')} — "
            f"adapter archived by the VM (adapter_in + results-<date>/), adapter_in NOT overwritten")
        print(f"\n[run_daily] SUCCESS — {reason}, but the VM finished cleanly "
              f"(result.json ok=true on Drive); adapter + archive are on Drive", flush=True)
        return True
    if res is not None:
        log(f"VM reported ok=false on Drive ({res[1]}) — treating as a failed run")
    _stop_keep_alive(keep_alive_stop)
    recover_latest_checkpoint_to_adapter_in(folder_in, run_id, _LAST_DRIVE_AT)
    colab("stop", "-s", SESSION, timeout=120)
    raise SystemExit(f"{reason}{': ' + detail if detail else ''} — latest Drive checkpoint recovered into adapter_in")


def main():
    # --- secrets ---
    cid = os.environ["COLAB_CLIENT_ID"]
    csec = os.environ["COLAB_CLIENT_SECRET"]
    cref = os.environ["COLAB_REFRESH_TOKEN"]
    adc = os.environ["GDRIVE_ADC"]
    folder_in = os.environ["DRIVE_ADAPTER_IN"]
    folder_out = os.environ["DRIVE_RESULTS"]
    rows = os.environ.get("RUN_ROWS", "2000")
    save_steps = os.environ.get("RUN_SAVE_STEPS", "100")
    max_minutes = os.environ.get("RUN_MAX_MINUTES", "100")
    epochs = os.environ.get("RUN_EPOCHS", "1")
    unlimited = max_minutes.strip() in ("", "0")
    seed = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    # UNIQUE run id (date+time): same-day re-runs must write into their OWN
    # checkpoint folder. With a date-only run_id two runs collide on the same
    # step-N filenames and find_latest_checkpoint can restore the OLD adapter
    # on a tie (key > best[0] fails) — silently breaking continuity.
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

    # --- auth ---
    tok = mint_access_token(cid, csec, cref)
    write_token_file(cid, csec, cref, tok["access_token"],
                     ["openid",
                      "https://www.googleapis.com/auth/userinfo.profile",
                      "https://www.googleapis.com/auth/userinfo.email",
                      "https://www.googleapis.com/auth/cloud-platform",
                      "https://www.googleapis.com/auth/colaboratory",
                      "https://www.googleapis.com/auth/drive.file"],
                     int(tok.get("expires_in", 3599)) - 60)
    write_adc_file(adc)
    # Store creds for in-process token refresh (keeps `colab logs` working past
    # the initial ~59-min token expiry — without it the runner falsely declares
    # HEARTBEAT_STALLED and stops a healthy VM).
    set_colab_creds(cid, csec, cref,
                    ["openid",
                     "https://www.googleapis.com/auth/userinfo.profile",
                     "https://www.googleapis.com/auth/userinfo.email",
                     "https://www.googleapis.com/auth/cloud-platform",
                     "https://www.googleapis.com/auth/colaboratory",
                     "https://www.googleapis.com/auth/drive.file"])

    # --- sanity: gdrive works headlessly (ADC can be inline JSON or path) ---
    rc, out, err = gdrive("about", timeout=120)
    log(f"gdrive about: rc={rc} {out[:160]}")
    if rc != 0:
        log(f"gdrive about FAILED (continuing, will retry on upload): {err[-300:]}")

    # --- session (retry on transient quota exhaustion: free T4 quota is
    # ~3-4 sessions/account/day and `gpu-unavailable` is common at peak) ---
    # First clean up any orphaned assignment from a previous run that was
    # cancelled mid-flight (a cancelled GH run never ran `colab stop`, so the
    # VM keeps occupying the account's concurrent-assignment slot and `new`
    # fails with "Precondition Failed"). `recover` rebuilds the local registry
    # from server-side list_assignments(), then `stop` releases the orphan.
    log("pre-session cleanup: recovering any orphaned assignment on this account")
    rc, out, err = colab("recover", timeout=120)
    log(f"recover: rc={rc} {out[-160:]} {err[-160:]}")
    rc, out, err = colab("stop", "-s", SESSION, timeout=120)
    log(f"pre-clean stop: rc={rc} {out[-160:]} {err[-160:]}")

    rc, out, err = 1, "", ""
    for attempt in range(4):
        rc, out, err = colab("new", "-s", SESSION, "--gpu", "T4", timeout=300)
        log(f"new session attempt {attempt+1}: rc={rc} {out[-200:]} {err[-200:]}")
        if rc == 0:
            break
        combined = str(out) + str(err)
        if attempt < 3 and ("gpu-unavailable" in combined or "Precondition Failed" in combined or "too many" in combined.lower()):
            log("GPU quota / assignment contention — cleaning orphans and waiting 300s before retry")
            # Re-run the cleanup between attempts: a concurrent run (or a
            # cancelled one still winding down) can hold the slot.
            colab("recover", timeout=120)
            colab("stop", "-s", SESSION, timeout=120)
            time.sleep(300)
        else:
            break
    if rc != 0:
        raise SystemExit(f"failed to create T4 session: {str(out)[-500:] or str(err)[-500:]}")
    # The session's clock starts here — this is the reference point for the
    # absolute deadline handed to the VM (see WALL_MINUTES).
    session_created_at = time.time()
    wall_deadline_epoch = int(session_created_at + WALL_MINUTES * 60) if WALL_MINUTES > 0 else 0
    if wall_deadline_epoch:
        log(f"session wall budget {WALL_MINUTES} min — VM will stop and finalize at "
            f"{datetime.datetime.fromtimestamp(wall_deadline_epoch, datetime.timezone.utc).isoformat()}")
    # Colab idle-prunes free VMs whose keep-alive dies (~60 min). The colab-cli's
    # own daemon caches the startup token (expires ~59 min) and dies — we run our
    # own ping loop for the whole poll window instead.
    keep_alive_stop = _start_keep_alive(SESSION)
    # --- upload training scripts + Drive tooling to the VM ---
    for local, remote in [
        (str(REPO / "daily_finetune.py"), "/content/daily_finetune.py"),
        (str(REPO / "gdrive.py"), "/content/gdrive.py"),
        (str(REPO / "pubmed_ingest.py"), "/content/pubmed_ingest.py"),
        (str(REPO / "egtkg_build.py"), "/content/egtkg_build.py"),
        (str(REPO / "textbook_ingest.py"), "/content/textbook_ingest.py"),
        (str(REPO / "cases_ingest.py"), "/content/cases_ingest.py"),
    ]:
        rc, o, e = colab("upload", "-s", SESSION, local, remote, timeout=180)
        if rc != 0:
            raise SystemExit(f"upload {local} failed: {e[-500:]}")
    # ADC JSON as a file on the VM (gdrive.py GDRIVE_ADC=path form)
    adc_local = pathlib.Path("/tmp/gdrive_adc.json")
    adc_local.write_text(adc)
    rc, o, e = colab("upload", "-s", SESSION, str(adc_local), "/content/gdrive_adc.json", timeout=180)
    if rc != 0:
        raise SystemExit(f"upload ADC failed: {e[-500:]}")

    # --- write + launch launcher (detached, logs to /content/train.log) ---
    launcher = pathlib.Path("/tmp/launch_daily.py")
    launcher.write_text(f"""import subprocess, sys
cmd = [sys.executable, '/content/daily_finetune.py',
       '--rows', '{rows}', '--seed', '{seed}',
       '--adapter-from-drive', '{folder_in}',
       '--adapter-out', '{folder_in}',
       '--drive-results', '{folder_out}',
       '--gdrive-py', '/content/gdrive.py',
       '--adc-file', '/content/gdrive_adc.json',
       '--run-id', '{run_id}',
       '--save-steps', '{save_steps}',
       '--max-minutes', '{max_minutes}',
       '--deadline-epoch', '{wall_deadline_epoch}',
       '--epochs', '{epochs}',
       '--out', '/content/out']
print('LAUNCH ' + ' '.join(cmd), flush=True)
r = subprocess.run(cmd)
print('EXIT ' + str(r.returncode), flush=True)
sys.exit(r.returncode)
""")
    rc, out, err = colab("exec_detach", "-s", SESSION, "-f", str(launcher),
                         "--log", "/content/train.log", timeout=300)
    log(f"exec_detach: rc={rc} {out[-200:]} {err[-200:]}")
    if rc != 0:
        # Don't poll a log that will never appear — fail fast with the error.
        colab("stop", "-s", SESSION, timeout=120)
        raise SystemExit(f"exec_detach failed (session stopped): {out[-400:] or err[-400:]}")

    # --- poll ---
    # The poll budget is anchored to the VM's ABSOLUTE wall (session create +
    # RUN_WALL_MINUTES + finalize slack), not to max_minutes: the VM now stops at
    # the wall, so waiting max_minutes + 20 min after training starts would keep
    # the runner hanging long after the VM death we are trying to stay ahead of.
    if wall_deadline_epoch:
        deadline = wall_deadline_epoch + WALL_FINALIZE_SLACK_MIN * 60
    else:
        deadline = time.time() + TRAIN_TIMEOUT_S
    max_minutes = int(float(max_minutes)) if str(max_minutes).strip() else 100
    # training budget handed to poll_log for the adaptive deadline: never past
    # the same absolute wall (it only extends the deadline for slow setups).
    if wall_deadline_epoch:
        train_budget_s = max(600, deadline - time.time())
    else:
        train_budget_s = (max_minutes + 20) * 60 if max_minutes > 0 else 240 * 60
    log(f"polling /content/train.log until "
        f"{datetime.datetime.fromtimestamp(deadline, datetime.timezone.utc).isoformat()} "
        f"(stall window {HEARTBEAT_STALL_S}s on BOTH channels; wall {WALL_MINUTES} min "
        f"+ {WALL_FINALIZE_SLACK_MIN} min finalize slack)")
    final = poll_log(deadline, results_folder=folder_out, run_id=run_id,
                     train_budget_s=train_budget_s)
    if final == "HEARTBEAT_STALLED":
        # Both liveness channels (VM log AND Drive checkpoints) went quiet for
        # the stall window — the VM really is gone (session recycled / OOM).
        log("HEARTBEAT STALLED: VM appears dead — settling, then recovering the newest checkpoint")
        if finish_incomplete(folder_in, folder_out, run_id, keep_alive_stop,
                             "VM heartbeat stalled (training died)"):
            return
    if final is None:
        if unlimited:
            # max_minutes=0: the VM self-stops only when the Colab session is
            # recycled (~12h). The runner's poll budget is finite, so when it
            # expires we LEAVE THE SESSION RUNNING — training continues
            # detached on the VM, and the Drive checkpoints keep flowing.
            log("UNLIMITED MODE: runner poll budget reached; training continues detached on the VM")
            log("leaving session RUNNING — checkpoints keep being pushed to Drive until VM recycle")
            recover_latest_checkpoint_to_adapter_in(folder_in, run_id, _LAST_DRIVE_AT)
            log("latest Drive checkpoint recovered into adapter_in (next run resumes from real progress)")
            log("DONE (unlimited mode — session NOT stopped)")
            print("\n[run_daily] SUCCESS — training continues detached; Drive checkpoints accumulating", flush=True)
            return
        log("TIMEOUT: training did not finish in time (Colab may have recycled the session)")
        if finish_incomplete(folder_in, folder_out, run_id, keep_alive_stop,
                             "training timed out"):
            return
    if final is None:
        # unreachable: finish_incomplete() either returns True (VM finished) or
        # raises SystemExit (checkpoint recovered). Guard keeps this explicit.
        return
    log(f"training log tail:\n{final[-2000:]}")
    if not training_succeeded(final):
        # Training ran but did NOT finish cleanly (callback crash, OOM, ...).
        # Settle first (a late result.json ok=true means it DID finish), else
        # recover whatever checkpoint the VM pushed — never report SUCCESS on a
        # run that produced nothing.
        if finish_incomplete(folder_in, folder_out, run_id, keep_alive_stop,
                             "training failed on the VM (no [RESULT] ok=true in log)",
                             detail=final[-800:]):
            return

    # --- download results (best-effort; Drive continuity is already done by VM) ---
    outdir = pathlib.Path("out")
    outdir.mkdir(exist_ok=True)
    results = {}
    if _RESULT_VIA_DRIVE:
        # The log/exec channel was dead for this whole phase — `colab download`
        # rides the same dead channel, so don't burn 3 x 5 min timeouts. The VM
        # already pushed adapter_in + results-<date>/ + checkpoints to Drive.
        log("log channel was dead: skipping colab download — adapter is on Drive "
            "(adapter_in + results-<date>/ + checkpoints)")
        res = drive_run_result(folder_out, run_id)
        payload = res[1] if res else {}
        (outdir / "metrics.json").write_text(json.dumps(payload, indent=2))
        log(f"wrote out/metrics.json from the Drive result.json ({payload})")
    else:
        for remote, local in [
            ("/content/out/adapter_model.safetensors", "out/adapter_model.safetensors"),
            ("/content/out/adapter_config.json", "out/adapter_config.json"),
            ("/content/out/metrics.json", "out/metrics.json"),
        ]:
            rc, o, e = colab("download", "-s", SESSION, remote, local, timeout=300)
            results[local] = rc == 0 and pathlib.Path(local).exists()
            log(f"download {local}: rc={rc} exists={results[local]}")
        if not results["out/adapter_model.safetensors"]:
            log("WARNING: adapter download failed — VM already pushed it to Drive (adapter_in + archive)")

    # --- stop session (free tier: don't leave it idle) ---
    _stop_keep_alive(keep_alive_stop)
    colab("stop", "-s", SESSION, timeout=120)

    log("DONE")
    print("\n[run_daily] SUCCESS — adapter + checkpoints on Drive", flush=True)


if __name__ == "__main__":
    main()
