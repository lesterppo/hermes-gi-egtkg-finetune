#!/usr/bin/env python3
"""
daily_finetune.py — daily financial-domain LoRA fine-tune of Llama-3.1-8B-Instruct
on a free Colab T4. Headless, idempotent, self-contained.

The daily loop: load the CURRENT adapter (from Drive checkpoints, or the
"latest" adapter folder) and CONTINUE fine-tuning it on a fresh finance-alpaca
subset, then save the updated adapter + metrics to /content/out/. The
orchestrator (GitHub Action, or the local colabctl driver) handles session
creation and teardown — this script only trains.

DRIVE CHECKPOINTING (the timeout / session-recycle safety net):
  Free-Colab sessions are recycled after ~2-3 h and the VM's ephemeral disk is
  wiped. To survive that, this script MOUNTS Google Drive via the vendored
  gdrive.py (headless OAuth via a gcloud ADC refresh token — no browser, no
  interactive consent) and pushes a checkpoint to Drive after every
  `--save-steps` training steps, plus a final checkpoint when the run ends —
  whether it ends naturally, or because `--max-minutes` budget was reached.
  On startup it looks for the most recent Drive checkpoint (across all run
  folders) and resumes from it, so a killed run loses at most `--save-steps`
  steps of work.

  Drive layout under --drive-results (the results root folder):
    checkpoints/<run-id>/step-<n>-adapter_model.safetensors   (keep last N)
    checkpoints/<run-id>/adapter_config.json                  (constant)
    results-<date>/adapter_model.safetensors + config + metrics.json  (archive)
    --adapter-out/<adapter files>                             (continuity pointer)

Stack (the exact deps Soup wraps, with Soup's known-good pins):
    transformers<5.0   (Colab ships 5.13.1 — must downgrade)
    trl>=0.14,<0.29
    peft, bitsandbytes, accelerate, datasets, safetensors
    google-api-python-client, google-auth  (for the vendored gdrive.py)
    torch (Colab's 2.11.0+cu128)

Fixes baked in:
    - uninstall torchao (Colab's stale 0.10.0 makes peft raise)
    - bf16 mixed precision, NOT fp16 (fp16 crashes at clip_grad_norm on a T4:
      "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
      BFloat16"; bf16 uses no GradScaler so it never hits that kernel)

Config is the proven "v2" recipe: NF4 QLoRA, r=16/alpha=32, all 7 linear layers,
batch 4, seq 512, lr 2e-4, bf16 (T4 has no bf16 hardware, but bf16 mixed
precision is the canonical QLoRA-on-T4 recipe — see soup-colab-finetune skill).

Usage (on the VM):
    python daily_finetune.py --rows 5000 --seed 42 \
        --adapter /content/adapter_in --out /content/out \
        --drive-results <folder_id> --gdrive-py /content/gdrive.py \
        --adc-file /content/gdrive_adc.json --run-id 20260818 \
        --save-steps 100 --max-minutes 100
"""
import argparse
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=False)


def install_deps():
    """Idempotent install of the known-good training stack."""
    run(f"{sys.executable} -m pip uninstall -q -y torchao")
    run(
        f"{sys.executable} -m pip install -q --no-input "
        f"'transformers>=4.46.0,<5.0.0' 'trl>=0.14.0,<0.29' "
        f"'bitsandbytes>=0.41.0' 'accelerate>=0.25.0' 'peft>=0.7.0' "
        f"'datasets>=2.14.0' 'safetensors' 'gdown' "
        f"'google-api-python-client>=2.0' 'google-auth>=2.0' 'google-auth-httplib2>=0.1'"
    )
    # report versions
    for m in ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate"]:
        try:
            mod = __import__(m)
            print(f"[deps] {m} {getattr(mod, '__version__', '?')}", flush=True)
        except Exception as e:
            print(f"[deps] {m} FAILED: {e}", flush=True)


# ---------------------------------------------------------------------------
# Headless Google Drive "mount" (vendored gdrive.py + ADC refresh token)
# ---------------------------------------------------------------------------

class Drive:
    """Thin wrapper around Google Drive for checkpoint sync.

    Uploads use an IN-PROCESS googleapiclient client (built once, reused) to
    avoid spawning a full `gdrive.py` subprocess (python + googleapiclient
    import ≈ 200MB transient RAM spike) per checkpoint push — that spike right
    after the step-100 save is the suspected OOM trigger in Colab's ~12GB RAM
    cgroup. Falls back to the vendored gdrive.py CLI if the in-process path
    can't initialize.
    """

    FOLDER_MIME = "application/vnd.google-apps.folder"
    SCOPE = "https://www.googleapis.com/auth/drive.file"

    def __init__(self, gdrive_py, adc_file):
        self.gdrive = gdrive_py
        self.adc_file = adc_file
        self.env = {**os.environ, "GDRIVE_ADC": adc_file}
        self._api = None
        self._media = None

    def _get_api(self):
        """Lazily build an in-process Drive service from the ADC (no subprocess)."""
        if self._api is not None:
            return self._api
        try:
            import json as _json
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            data = _json.loads(pathlib.Path(self.adc_file).read_text())
            if data.get("type") != "authorized_user":
                return None
            # Discover the quota project the same way the vendored gdrive.py
            # does (ADC client -> owned project with Drive API enabled).
            quota_project = None
            try:
                tok = _json.loads(_json.dumps(data))
                import urllib.request as _u
                import urllib.parse as _up
                _body = _up.urlencode({
                    "client_id": tok.get("client_id", ""),
                    "client_secret": tok.get("client_secret", ""),
                    "refresh_token": tok.get("refresh_token", ""),
                    "grant_type": "refresh_token",
                }).encode()
                _req = _u.Request("https://oauth2.googleapis.com/token", data=_body,
                                  headers={"Content-Type": "application/x-www-form-urlencoded"})
                _at = _json.loads(_u.urlopen(_req, timeout=30).read()).get("access_token")
                _hr = _u.Request(
                    "https://cloudresourcemanager.googleapis.com/v1/projects"
                    "?filter=labels.goog-owners-responsibility%3A%22%22&pageSize=1",
                    headers={"Authorization": f"Bearer {_at}",
                             "x-goog-user-project": ""})
                # simple non-filtered list (first owned project is the gcloud default)
                _hr = _u.Request("https://cloudresourcemanager.googleapis.com/v1/projects?pageSize=5",
                                 headers={"Authorization": f"Bearer {_at}"})
                _pj = _json.loads(_u.urlopen(_hr, timeout=30).read())
                for _p in _pj.get("projects", []):
                    quota_project = _p.get("projectId")
                    break
            except Exception:
                quota_project = None
            if not quota_project:
                return None  # in-process path unusable without quota project
            creds = Credentials(
                token=None,
                refresh_token=data.get("refresh_token"),
                token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=data.get("client_id"),
                client_secret=data.get("client_secret"),
                scopes=[self.SCOPE],
                # quota project: ADC from the gcloud client REQUIRES a quota
                # project on Drive calls, else 403
                # ("authenticating by using local Application Default
                # Credentials..."). Discover it from the ADC's own client id
                # via cloudresourcemanager (matches the gdrive.py vendored
                # CLI behaviour) and pass it on every request below.
                quota_project_id=quota_project,
            )
            creds.refresh(Request())
            self._api = build("drive", "v3", credentials=creds, cache_discovery=False)
            self._media = MediaFileUpload
        except Exception as e:
            print(f"[drive] in-process API unavailable ({str(e)[:120]}) — falling back to gdrive.py subprocess", flush=True)
            self._api = False  # sentinel: don't retry every call
        return self._api if self._api else None

    def upload(self, local, folder, name, timeout=600):
        """Upload a file. Prefers in-process API; falls back to gdrive.py CLI."""
        api = self._get_api()
        if api is not None:
            try:
                import mimetypes
                mime = mimetypes.guess_type(str(local))[0] or "application/octet-stream"
                media = self._media(
                    str(local), mimetype=mime, resumable=os.path.getsize(local) > 8 * 1024 * 1024,
                    chunksize=8 * 1024 * 1024,
                )
                body = {"name": name, "parents": [folder]}
                req = api.files().create(body=body, media_body=media,
                                         fields="id,name,size", supportsAllDrives=True)
                resp = req.execute()
                print(f"[drive] uploaded {name} via in-process API ({resp.get('size')} bytes)", flush=True)
                return {"ok": True, "id": resp.get("id"), "name": resp.get("name")}
            except Exception as e:
                print(f"[drive] in-process upload failed ({str(e)[:150]}) — falling back to gdrive.py", flush=True)
        return self._call("upload", local, "--parent", folder, "--name", name, timeout=timeout)

    def _call(self, *args, timeout=300):
        try:
            r = subprocess.run(
                [sys.executable, self.gdrive, *args],
                capture_output=True, text=True, timeout=timeout, env=self.env,
            )
            try:
                data = json.loads(r.stdout or "{}")
                return data if isinstance(data, dict) else {"e": (r.stdout or r.stderr or "")[-300:]}
            except Exception:
                return {"e": (r.stdout or r.stderr or "")[-300:]}
        except Exception as e:
            return {"e": str(e)}

    def is_ok(self, resp):
        return isinstance(resp, dict) and resp.get("ok") is True

    def ensure_folder(self, parent, name):
        """Find a child folder by name under parent, else create it. Returns id or None."""
        lst = self._call("list", "--folder", parent, "--max", "100")
        for it in lst.get("items", []):
            if it.get("m") == self.FOLDER_MIME and it.get("n") == name:
                return it["id"]
        mk = self._call("mkdir", name, "--parent", parent)
        return mk.get("id") if self.is_ok(mk) else None

    def list_files(self, folder, max_n=200):
        lst = self._call("list", "--folder", folder, "--max", str(max_n))
        return lst.get("items", []) if self.is_ok(lst) else []

    def rm(self, fid):
        return self._call("rm", fid, timeout=120)

    def download(self, fid, out_path, timeout=600):
        return self._call("download", fid, "--out", out_path, timeout=timeout)


def setup_checkpoints(drive, results_folder, run_id):
    """Create/return (checkpoints_parent_folder_id, this_run_folder_id)."""
    if not drive or not results_folder:
        return None, None
    parent = drive.ensure_folder(results_folder, "checkpoints")
    if not parent:
        print("[drive] WARNING: could not create checkpoints folder under results root", flush=True)
        return None, None
    run_folder = drive.ensure_folder(parent, run_id)
    return parent, run_folder


def find_latest_checkpoint(drive, results_folder, current_run_id):
    """Return (run_folder_id, step, file_id, file_name) of the globally newest
    step checkpoint, or None. Scans all run folders under checkpoints/ and
    picks the max (run_id, step). Skips runs strictly newer than current."""
    if not drive or not results_folder:
        return None
    parent = drive.ensure_folder(results_folder, "checkpoints")
    if not parent:
        return None
    best = None
    for folder in drive.list_files(parent):
        if folder.get("m") != drive.FOLDER_MIME:
            continue
        run_name = folder.get("n", "")
        if run_name > current_run_id:
            continue  # future runs don't exist / can't be newer
        fid = folder.get("id")
        for f in drive.list_files(fid):
            m = re.match(r"step-(\d+)-adapter_model\.safetensors$", f.get("n", ""))
            if not m:
                continue
            key = (run_name, int(m.group(1)))
            if best is None or key > best[0]:
                best = (key, fid, f.get("id"), f.get("n"))
    if best is None:
        return None
    (run_name, step), folder_id, file_id, file_name = best
    print(f"[drive] newest checkpoint: {run_name} step-{step} ({file_name})", flush=True)
    return folder_id, step, file_id, file_name


def find_latest_archive(drive, results_folder):
    """Return (folder_id, file_id, file_name, date) of the newest dated archive
    (results-<date>/adapter_model.safetensors — a completed fine-tuned model),
    or None. Used as a resume fallback when no step checkpoints exist."""
    if not drive or not results_folder:
        return None
    best = None
    for folder in drive.list_files(results_folder):
        if folder.get("m") != drive.FOLDER_MIME:
            continue
        name = folder.get("n", "")
        m = re.match(r"^results-(\d{4}-\d{2}-\d{2})$", name)
        if not m:
            continue
        for f in drive.list_files(folder.get("id")):
            if f.get("n") == "adapter_model.safetensors":
                key = m.group(1)  # ISO date sorts lexicographically
                if best is None or key > best[0]:
                    best = (key, folder.get("id"), f.get("id"), f.get("n"))
                break
    if best is None:
        return None
    date, folder_id, file_id, file_name = best
    print(f"[drive] newest archive: results-{date} ({file_name})", flush=True)
    return folder_id, file_id, file_name, date


def resume_from_drive(drive, results_folder, current_run_id, target_dir):
    """Restore the most-updated adapter on Drive into target_dir (as
    adapter_model.safetensors + adapter_config.json). Priority: newest step
    checkpoint (mid-run progress) > newest dated archive (completed model).
    Returns target_dir on success, else None."""
    found = find_latest_checkpoint(drive, results_folder, current_run_id)
    source = "checkpoint"
    if not found:
        arch = find_latest_archive(drive, results_folder)
        if arch:
            folder_id, file_id, file_name, date = arch
            found = (folder_id, 0, file_id, file_name)
            source = f"archive results-{date}"
    if not found:
        return None
    folder_id, step, file_id, file_name = found
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, "adapter_model.safetensors")
    drive.download(file_id, out_path)
    # config is constant — pull it if present in the same folder
    cfg = os.path.join(target_dir, "adapter_config.json")
    if not os.path.exists(cfg):
        for f in drive.list_files(folder_id):
            if f.get("n") == "adapter_config.json":
                drive.download(f.get("id"), cfg)
                break
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[resume] CONTINUING from {source} (step-{step}, {file_name}) -> {target_dir}", flush=True)
        return target_dir
    return None


def push_checkpoint(drive, run_folder, step, local_dir, keep=2):
    """Upload adapter files from local_dir into run_folder as step-<n>, then
    prune old step files keeping the newest `keep`. Never raises."""
    if not drive or not run_folder:
        return
    adapter = os.path.join(local_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter):
        return
    name = f"step-{step}-adapter_model.safetensors"
    r = drive.upload(adapter, run_folder, name)
    if not drive.is_ok(r):
        print(f"[drive] upload failed step {step}: {r.get('e', '')[:200]}", flush=True)
        return
    cfg = os.path.join(local_dir, "adapter_config.json")
    if os.path.exists(cfg):
        has = any(f.get("n") == "adapter_config.json" for f in drive.list_files(run_folder))
        if not has:
            drive.upload(cfg, run_folder, "adapter_config.json")
    files = [f for f in drive.list_files(run_folder)
             if re.match(r"step-\d+-adapter_model\.safetensors$", f.get("n", ""))]
    def _step_of(f):
        m = re.match(r"step-(\d+)-", f.get("n", ""))
        return int(m.group(1)) if m else -1
    files.sort(key=_step_of)
    for f in files[:-keep]:
        drive.rm(f["id"])
    print(f"[drive] checkpoint step-{step} pushed to Drive ({len(files)} step files, keeping {keep})", flush=True)


def purge_names(drive, folder, names, passes=4):
    """Delete every file in `folder` whose name is in `names`, re-listing until
    the listing stops showing them.

    Drive listings are eventually consistent, so a single delete pass leaves
    stale copies behind: by 2026-09-11 adapter_in had accumulated 11 past
    adapter pairs (every nightly recovery + VM final push), which makes
    `gdown --folder` consumers pick an arbitrary/older adapter.
    """
    want = set(names)
    for _ in range(passes):
        for f in drive.list_files(folder, max_n=200):
            if isinstance(f, dict) and f.get("n") in want:
                drive.rm(f.get("id"))
        remaining = {f.get("n") for f in drive.list_files(folder, max_n=200) if isinstance(f, dict)}
        if not (remaining & want):
            return True
    return False


def push_adapter_in(drive, adapter_out_folder, local_dir):
    """Replace the 'latest adapter' continuity folder contents. Best-effort."""
    if not drive or not adapter_out_folder:
        return
    purge_names(drive, adapter_out_folder, ("adapter_model.safetensors", "adapter_config.json"))
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        p = os.path.join(local_dir, name)
        if os.path.exists(p):
            r = drive.upload(p, adapter_out_folder, name)
            if drive.is_ok(r):
                print(f"[drive] adapter_in updated: {name}", flush=True)


def push_archive(drive, results_folder, date, local_dir, metrics):
    """Archive final adapter + metrics under results-<date>/. Best-effort."""
    if not drive or not results_folder:
        return
    folder = drive.ensure_folder(results_folder, f"results-{date}")
    if not folder:
        return
    for name in ("adapter_model.safetensors", "adapter_config.json", "metrics.json", "loss_curve.json"):
        p = os.path.join(local_dir, name)
        if os.path.exists(p):
            drive.upload(p, folder, name)
    print(f"[drive] archived to results-{date}", flush=True)


def build_data(n_rows: int, seed: int, out_path: str) -> int:
    """Generate today's EGT-KG evidence-grounded medical QA training rows by
    running the two staged sibling scripts on the VM:
        pubmed_ingest.py  ->  egtkg_build.py
    pubmed_ingest pulls fresh GI/hepatology evidence (PubMed abstracts +
    Europe PMC OA full texts) into an accumulating knowledge_store.jsonl
    (persisted to Drive, so the model keeps learning NEW evidence each day).
    egtkg_build constructs the typed, reified KG and emits evidence-grounded
    ChatML QA rows (verbatim evidence + provenance; per the linked EGT-KG
    paper a local SLM must ground answers on the original text).
    Both scripts are staged to /content alongside this file by the runner.
    Returns the count of rows written.
    """
    store_path = os.environ.get("KNOWLEDGE_STORE", "/content/kn_store.jsonl")
    ingest_py = "/content/pubmed_ingest.py"
    build_py = "/content/egtkg_build.py"
    if not os.path.exists(ingest_py) or not os.path.exists(build_py):
        print(f"[data] staging scripts missing ({os.path.exists(ingest_py)},{os.path.exists(build_py)})",
              flush=True)
        pathlib.Path(out_path).write_text("")
        return 0
    if not os.path.exists(store_path):
        pathlib.Path(store_path).write_text("")

    days = os.environ.get("INGEST_DAYS", "20")
    r_ing = run(f"{sys.executable} {ingest_py} --days {days} "
                f"--store {store_path} --out /content/new.jsonl "
                f"--max-oa {os.environ.get('MAX_OA','24')} "
                f"--oa-probe 40")
    print(f"[data] ingest rc={r_ing.returncode}", flush=True)
    time.sleep(1)

    # Reference/textbook corpus (StatPearls 171 chapters + NIDDK 56 + WGO 26).
    # Stable knowledge — refresh only on the 1st of the month (cheap: ~250
    # pages, ~10 min) or when missing locally. Mixed into the same store so
    # egtkg_build sees literature + reference knowledge together.
    ref_path = "/content/ref_store.jsonl"
    need_ref = (not os.path.exists(ref_path)) or time.strftime("%d") == "01"
    if need_ref and os.path.exists("/content/textbook_ingest.py"):
        print("[data] refreshing textbook/reference corpus", flush=True)
        r_ref = run(f"{sys.executable} /content/textbook_ingest.py "
                    f"--out {ref_path} --sources statpearls,niddk,wgo")
        print(f"[data] textbook rc={r_ref.returncode}", flush=True)
    if os.path.exists(ref_path):
        run(f"cat {ref_path} >> {store_path}")

    # Open-access case reports (CC-licensed PMC): clinical-reasoning chains
    # (presentation -> workup -> diagnosis -> management) that abstracts
    # compress away. 180d window, newest 40/run, dedup by CR_ pmid key.
    if os.path.exists("/content/cases_ingest.py"):
        r_cr = run(f"{sys.executable} /content/cases_ingest.py "
                   f"--store /content/cr_store.jsonl --max 40")
        print(f"[data] case reports rc={r_cr.returncode}", flush=True)
        if os.path.exists("/content/cr_store.jsonl"):
            run(f"cat /content/cr_store.jsonl >> {store_path}")
    time.sleep(1)

    cap = max(200, n_rows)
    r_build = run(f"{sys.executable} {build_py} --store {store_path} "
                  f"--out {out_path} --limit {cap} --seed {seed}")
    print(f"[data] egtkg rc={r_build.returncode}", flush=True)

    if not os.path.exists(out_path):
        pathlib.Path(out_path).write_text("")
    n = sum(1 for ln in pathlib.Path(out_path).read_text().splitlines() if ln.strip())
    print(f"[data] EGT-KG QA rows ready: {n}", flush=True)
    return n


def load_model_and_adapter(base: str, adapter_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # standard QLoRA recipe (works on T4)
    )
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb,
        device_map="auto",
    )
    # NOTE: deliberately NO prepare_model_for_kbit_training() here. Soup's SFT
    # path skips it; calling it casts the frozen bf16 base to fp32, which then
    # produces bf16 gradients under the fp16 autocast and crashes the fp16
    # GradScaler on a T4 (see Soup PR #429 / issue #425).

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    if adapter_path and os.path.isdir(adapter_path) and os.path.exists(
        os.path.join(adapter_path, "adapter_model.safetensors")
    ):
        print(f"[adapter] continuing from {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print("[adapter] starting fresh LoRA", flush=True)
        model = get_peft_model(model, lora_config)

    from collections import Counter
    trainable = Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)
    print(f"[dtype] trainable: {dict(trainable)}", flush=True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_trainable}", flush=True)
    return model, tok


def train(model, tok, data_path: str, out_dir: str, save_steps: int,
          max_minutes: int, epochs: int, drive, run_folder):
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset
    from transformers import TrainerCallback

    ds = load_dataset("json", data_files=data_path, split="train")
    print(f"[data] loaded {len(ds)} rows", flush=True)

    def formatting_func(example):
        return tok.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    class DriveCheckpointCallback(TrainerCallback):
        def __init__(self, drive_, run_folder_):
            self.drive = drive_
            self.run_folder = run_folder_

        def on_save(self, args, state, control, model=None, **kwargs):
            step = state.global_step
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            if os.path.isdir(ckpt_dir) and os.path.exists(
                os.path.join(ckpt_dir, "adapter_model.safetensors")
            ):
                push_checkpoint(self.drive, self.run_folder, step, ckpt_dir)
            elif os.path.exists(os.path.join(args.output_dir, "adapter_model.safetensors")):
                push_checkpoint(self.drive, self.run_folder, step, args.output_dir)
            # ship the loss curve with every checkpoint so even a killed run
            # leaves a real training curve on Drive
            curve = os.path.join(args.output_dir, "loss_curve.json")
            if self.drive and self.run_folder and os.path.exists(curve):
                r = self.drive.upload(curve, self.run_folder, "loss_curve.json")
                if self.drive.is_ok(r):
                    print(f"[drive] loss_curve.json pushed at step {step}", flush=True)

    class TimeBudgetCallback(TrainerCallback):
        def __init__(self, budget_s):
            self.budget_s = budget_s
            self.start = time.time()
            self.timed_out = False

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if self.budget_s and time.time() - self.start > self.budget_s:
                control.should_training_stop = True
                if not self.timed_out:
                    self.timed_out = True
                    print(f"[train] time budget ({self.budget_s}s) reached at step {state.global_step} — stopping", flush=True)

    class LossCurveCallback(TrainerCallback):
        """Record every logged (step, loss) and persist loss_curve.json after
        each log so the curve survives VM death / runner timeout."""

        def __init__(self, out_dir):
            self.out_dir = out_dir
            self.curve = []
            self._path = os.path.join(out_dir, "loss_curve.json")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and logs.get("loss") is not None:
                self.curve.append({"step": int(state.global_step), "loss": float(logs["loss"])})
                pathlib.Path(self._path).write_text(json.dumps(self.curve))

    class HeartbeatCallback(TrainerCallback):
        """Emit a `[alive] step=N loss=X ram=Y` line every `every` steps with
        flush=True so the runner can see training is progressing, AND mirror the
        same heartbeat to Drive as heartbeat.json.

        The Drive copy is the runner's OUT-OF-BAND liveness channel: `colab logs`
        goes dark after ~60 min (token lifetime) while the VM keeps training, so
        a log-only stall rule kills healthy VMs (it did, nightly, before this).
        A fresh heartbeat.json mtime proves the VM is alive without the log.

        RAM read from /proc/meminfo (free Colab cgroup is ~12GB).
        """

        def __init__(self, every=10, drive=None, run_folder=None, out_dir=None):
            self.every = every
            self.drive = drive
            self.run_folder = run_folder
            self.out_dir = out_dir

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step % self.every != 0:
                return
            loss = logs.get("loss") if logs else None
            ram = ""
            try:
                total = avail = 0
                for line in open("/proc/meminfo"):
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / 1024 / 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) / 1024 / 1024
                ram = f"{total - avail:.1f}/{total:.1f}GB"
            except Exception:
                pass
            print(f"[alive] step={state.global_step} loss={loss} ram={ram}", flush=True)
            if self.drive and self.run_folder and self.out_dir:
                try:
                    hb = os.path.join(self.out_dir, "heartbeat.json")
                    pathlib.Path(hb).write_text(json.dumps({
                        "step": int(state.global_step),
                        "loss": loss,
                        "ram": ram,
                    }))
                    # replace, don't accumulate: a same-name upload creates a NEW
                    # Drive file, so a 240-min run would otherwise leave ~70
                    # heartbeat.json copies in the run folder
                    for f in self.drive.list_files(self.run_folder):
                        if isinstance(f, dict) and f.get("n") == "heartbeat.json":
                            self.drive.rm(f.get("id"))
                    self.drive.upload(hb, self.run_folder, "heartbeat.json")
                except Exception as e:
                    print(f"[drive] heartbeat push failed: {str(e)[:150]}", flush=True)

    budget = TimeBudgetCallback(max_minutes * 60 if max_minutes and max_minutes > 0 else 0)
    curve_cb = LossCurveCallback(out_dir)
    callbacks = [budget, curve_cb,
                 HeartbeatCallback(every=10, drive=drive, run_folder=run_folder, out_dir=out_dir)]
    if drive and run_folder:
        callbacks.append(DriveCheckpointCallback(drive, run_folder))

    args_cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=save_steps,
        save_strategy="steps",
        # model-only checkpoints: no optimizer.pt/scheduler.pt/rng_state (~500MB
        # of writes per save). The full checkpoint write at step 100 is the
        # suspected OOM trigger that kills the 12GB Colab RAM cgroup right after
        # the first Drive push; save_only_model cuts it to the ~80MB adapter.
        save_only_model=True,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=[],
        seed=1234,
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=args_cfg,
        train_dataset=ds,
        processing_class=tok,
        formatting_func=formatting_func,
        callbacks=callbacks,
    )

    print("[train] starting", flush=True)
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    steps = trainer.state.global_step
    print(f"[train] done in {dt:.1f}s at step {steps}", flush=True)

    # save adapter (also triggers one final checkpoint push via callback when
    # the step boundary aligns; push explicitly below for the final step)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[save] adapter -> {out_dir}", flush=True)

    partial = bool(budget.timed_out)
    last_loss = None
    for h in reversed(trainer.state.log_history):
        if h.get("loss") is not None:
            last_loss = float(h["loss"])
            break
    # full loss curve: every logged (step, loss) — the real training curve
    curve = [{"step": c["step"], "loss": c["loss"]} for c in curve_cb.curve]
    metrics = {
        "train_runtime_s": round(dt, 1),
        "train_samples": len(ds),
        "train_steps": steps,
        "partial": partial,
        "train_loss": round(last_loss, 4) if last_loss is not None else None,
        "loss_curve": curve,
    }
    pathlib.Path(out_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
    pathlib.Path(out_dir, "loss_curve.json").write_text(json.dumps(curve))
    print(f"[metrics] {json.dumps(metrics)}", flush=True)

    # final checkpoint push (step = current global_step)
    if drive and run_folder:
        push_checkpoint(drive, run_folder, steps, out_dir)
    return metrics


def verify_adapter(out_dir: str) -> bool:
    from safetensors.torch import load_file

    ad = pathlib.Path(out_dir) / "adapter_model.safetensors"
    if not ad.exists():
        print(f"[ADAPTER] NOT FOUND at {ad} — training failed", flush=True)
        return False
    tensors = load_file(str(ad))
    live = sum(1 for v in tensors.values() if v.abs().max().item() > 0)
    print(f"[ADAPTER] {len(tensors)} tensors, {live} non-zero -> {ad}", flush=True)
    return live > 0


def _latest_store_local(drive, results_folder, target):
    """Pull the newest persisted knowledge_store.jsonl (newest run folder under
    the results root holding a knowledge_store.jsonl, else the dated archive)
    into `target`. Used so the accumulating medical evidence store survives
    between daily jobs."""
    if not drive or not results_folder:
        return None
    best = None
    for folder in drive.list_files(results_folder):
        if folder.get("m") != drive.FOLDER_MIME:
            continue
        name = folder.get("n", "")
        # scan dated archive folders + checkpoints/* run folders
        cand = None
        for f in drive.list_files(folder.get("id")):
            if f.get("n") == "knowledge_store.jsonl":
                cand = (name, folder.get("id"), f.get("id"))
                break
        if cand and (best is None or cand[0] > best[0]):
            best = cand
    if not best:
        return None
    _, fid, file_id = best
    os.makedirs(os.path.dirname(target), exist_ok=True)
    drive.download(file_id, target)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"[drive] restored knowledge store -> {target}", flush=True)
        return target
    return None


def store_archive_push(drive, results_folder, date, local_store):
    """Push knowledge_store.jsonl to results-<date>/ and to the dedicated
    knowledge continuity pointer folder (--knowledge) if drive available."""
    if not drive or not results_folder or not os.path.exists(local_store):
        return
    # dated archive (single canonical home for the growing store)
    folder = drive.ensure_folder(results_folder, f"results-{date}")
    if folder:
        drive.upload(local_store, folder, "knowledge_store.jsonl")
    # continuity pointer (stable folder the VM pulls from next run)
    kb_folder = os.environ.get("KNOWLEDGE_FOLDER", "")
    if kb_folder:
        try:
            for f in drive.list_files(kb_folder):
                drive.rm(f.get("id"))
        except Exception:
            pass
        drive.upload(local_store, kb_folder, "knowledge_store.jsonl")
    print(f"[drive] knowledge store archived (date {date}, size "
          f"{os.path.getsize(local_store)} bytes)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter", default=None, help="dir with adapter_model.safetensors to continue from")
    ap.add_argument("--adapter-from-drive", default=None,
                    help="Google Drive FOLDER id (shared anyone-with-link) to gdown-download the adapter from (fallback)")
    ap.add_argument("--adapter-out", default=None,
                    help="Google Drive folder id to update as the 'latest adapter' continuity pointer")
    ap.add_argument("--out", default="/content/out")
    ap.add_argument("--base", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--drive-results", default=None,
                    help="Google Drive results root folder id (checkpoints/ lives under it)")
    ap.add_argument("--gdrive-py", default=None, help="path to vendored gdrive.py on the VM")
    ap.add_argument("--adc-file", default=None, help="path to gcloud ADC JSON on the VM (Drive auth)")
    ap.add_argument("--run-id", default=None, help="this run's folder name under checkpoints/ (e.g. YYYYMMDD)")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--max-minutes", type=int, default=100,
                    help="stop training after N minutes and save a final checkpoint (0 = unlimited)")
    ap.add_argument("--epochs", type=int, default=1,
                    help="epochs per run; >1 keeps training until the VM is recycled (12h window)")
    args = ap.parse_args()

    if not args.skip_install:
        install_deps()

    drive = None
    if args.gdrive_py and args.adc_file and os.path.exists(args.gdrive_py) and os.path.exists(args.adc_file):
        drive = Drive(args.gdrive_py, args.adc_file)
        print(f"[drive] mounted via gdrive.py ({args.gdrive_py}) + ADC", flush=True)
    elif args.drive_results:
        print("[drive] WARNING: --drive-results given but no --gdrive-py/--adc-file — Drive sync disabled", flush=True)

    run_id = args.run_id or time.strftime("%Y%m%d")
    # date derived from run_id (runner-side UTC) so the archive folder name
    # always matches the checkpoint run, regardless of VM local timezone
    base = run_id[:8] if run_id and len(run_id) >= 8 else time.strftime("%Y%m%d")
    date = f"{base[:4]}-{base[4:6]}-{base[6:8]}"
    _, run_folder = setup_checkpoints(drive, args.drive_results, run_id)

    # --- resume priority: Drive checkpoint > gdown adapter-in > fresh ---
    restored = resume_from_drive(drive, args.drive_results, run_id, "/content/adapter_in")
    if restored:
        args.adapter = restored
    elif args.adapter_from_drive and (not args.adapter or not os.path.isdir(args.adapter)):
        run(f"{sys.executable} -m pip install -q gdown")
        os.makedirs("/content/adapter_in", exist_ok=True)
        r = run(f"gdown --folder {args.adapter_from_drive} -O /content/adapter_in")
        if r.returncode != 0:
            print("[adapter] gdown --folder failed; check the folder is shared anyone-with-link", flush=True)
        args.adapter = "/content/adapter_in"

    # restore + persist the accumulating medical evidence store
    kstore = "/content/kn_store.jsonl"
    if not os.path.exists(kstore):
        _latest_store_local(drive, args.drive_results, kstore)
        if not os.path.exists(kstore):
            pathlib.Path(kstore).write_text("")

    build_data(args.rows, args.seed, "/content/train_egtkg.jsonl")
    model, tok = load_model_and_adapter(args.base, args.adapter)
    metrics = train(model, tok, "/content/train_egtkg.jsonl", args.out,
                    args.save_steps, args.max_minutes, args.epochs, drive, run_folder)
    ok = verify_adapter(args.out)

    # --- Drive continuity: archive + update the 'latest adapter' pointer ---
    if drive:
        store_archive_push(drive, args.drive_results, date, kstore)
        push_archive(drive, args.drive_results, date, args.out, metrics)
        push_adapter_in(drive, args.adapter_out, args.out)

    # --- completion marker for the runner (out-of-band success signal) ---
    # `colab logs` / `colab exec` starts failing with rc=1 ~60 min into a session
    # (colab-cli token lifetime) while training keeps running normally. [RESULT]
    # on stdout is therefore unreadable for the rest of the run, so the runner
    # needs a success signal that does NOT depend on the log channel: this file,
    # pushed to the same Drive folder it already watches for liveness.
    if drive and run_folder:
        try:
            res = {
                "ok": bool(ok),
                "run_id": run_id,
                "steps": metrics.get("train_steps"),
                "loss": metrics.get("train_loss"),
                "partial": metrics.get("partial"),
            }
            rp = pathlib.Path(args.out) / "result.json"
            rp.write_text(json.dumps(res))
            r = drive.upload(str(rp), run_folder, "result.json")
            if drive.is_ok(r):
                print(f"[drive] result.json pushed (ok={res['ok']} steps={res['steps']})", flush=True)
            else:
                print(f"[drive] result.json push failed: {str(r.get('e', ''))[:150]}", flush=True)
        except Exception as e:
            print(f"[drive] result.json push failed: {str(e)[:150]}", flush=True)

    print(f"\n[RESULT] ok={ok} {json.dumps(metrics)}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
