#!/usr/bin/env python3
"""
eval_ab.py — base-vs-adapter A/B evaluation for the GI/hep EGT-KG LoRA.

Runs on a free Colab T4 (or any CUDA box). For every held-out eval row it
generates the answer TWICE from the same prompt:

    * adapter ON  (the fine-tuned model)
    * adapter OFF (model.disable_adapter() — the raw Llama-3.1-8B-Instruct NF4)

Greedy decoding (do_sample=False) so the comparison is deterministic and the
outputs can be scored mechanically (verbatim-evidence quoting, provenance
format) as well as by a human/LLM judge.

Usage (on the VM):
    python3 eval_ab.py --rows /content/eval_rows.jsonl --out /content/ab_results.jsonl \
        --adapter-parent <your adapter_in folder id> --adc /content/gdrive_adc.json
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

BASE = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def run(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw)


def deps():
    """Install the eval stack, then PROVE the imports work.

    The version floors matter: transformers' 4-bit quantizer rejects
    bitsandbytes below its own BITSANDBYTES_MIN_VERSION (0.46.1 with the
    transformers release Colab currently ships), and that failure surfaces as a
    long ImportError traceback only at model-load time — i.e. after the adapter
    download and minutes into the session. Verify here instead.
    """
    run(f"{sys.executable} -m pip uninstall -q -y torchao")
    run(f"{sys.executable} -m pip install -q "
        f"'transformers>=4.46,<5.0' 'peft>=0.7' 'bitsandbytes>=0.46.1' "
        f"'accelerate>=0.25' 'safetensors' gdown")
    verify_deps()


def verify_deps() -> None:
    """Fail fast, with versions, if the quantized-load stack is incomplete."""
    import importlib
    for mod in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        m = importlib.import_module(mod)
        print(f"[eval] {mod} {getattr(m, '__version__', '?')}", flush=True)
    import torch
    if not torch.cuda.is_available():
        print("[eval] WARNING: no CUDA device visible", flush=True)


def fetch_adapter(folder_id, dest, adc=None, gdrive_py="/content/gdrive.py"):
    """Download the adapter into `dest`.

    Prefers the vendored gdrive.py CLI + ADC (the folder is NOT link-shared
    anymore — privacy), which is exactly how the training runs authenticate.
    gdown is kept as a fallback for link-shared folders.
    """
    dest = pathlib.Path(dest)
    if (dest / "adapter_model.safetensors").exists():
        return str(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # 1. in-process Drive API via daily_finetune.Drive (identical to how the
    #    training runs pull/push the adapter — the reliable path when the folder
    #    is owner-only and gdown therefore cannot read it).
    try:
        sys.path.insert(0, "/content")
        from daily_finetune import Drive  # noqa: E402
        drive = Drive(gdrive_py, adc)
        items = drive.list_files(folder_id)
        got = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = it.get("n")
            if nm in ("adapter_model.safetensors", "adapter_config.json") and it.get("id"):
                r = drive.download(it["id"], str(dest / nm))
                if os.path.exists(dest / nm) and (dest / nm).stat().st_size > 0:
                    got += 1
                else:
                    print(f"[eval] Drive download of {nm} failed: {str(r)[:160]}", flush=True)
        if (dest / "adapter_model.safetensors").exists():
            print(f"[eval] adapter fetched via Drive API ({got} files)", flush=True)
            return str(dest)
        print(f"[eval] Drive API listed {len(items)} item(s) but no adapter files — "
              f"check --adapter-parent is the adapter_in FOLDER id", flush=True)
    except Exception as e:
        print(f"[eval] in-process Drive fetch failed: {str(e)[:200]}", flush=True)

    # 2. vendored gdrive.py CLI with the ADC
    if adc and os.path.exists(adc) and os.path.exists(gdrive_py):
        env = {**os.environ, "GDRIVE_ADC": adc}
        r = subprocess.run([sys.executable, gdrive_py, "list", "--folder", folder_id, "--max", "50"],
                           capture_output=True, text=True, env=env)
        try:
            items = json.loads(r.stdout or "{}").get("items", [])
        except Exception:
            items = []
            print(f"[eval] gdrive.py list unparsable: rc={r.returncode} {str(r.stdout)[:160]} {str(r.stderr)[:160]}",
                  flush=True)
        wanted = {"adapter_model.safetensors": "adapter_model.safetensors",
                  "adapter_config.json": "adapter_config.json"}
        got = 0
        for it in items:
            nm = it.get("n") if isinstance(it, dict) else None
            if nm in wanted:
                out = dest / wanted[nm]
                subprocess.run([sys.executable, gdrive_py, "download", it["id"], "--out", str(out)],
                               env=env)
                got += 1 if out.exists() and out.stat().st_size > 0 else 0
        if (dest / "adapter_model.safetensors").exists():
            print(f"[eval] adapter fetched via gdrive.py ({got} files)", flush=True)
            return str(dest)
        print("[eval] gdrive.py fetch incomplete — falling back to gdown", flush=True)

    # 3. gdown (only works for an anyone-with-link folder)
    r = run(f"{sys.executable} -m gdown --folder {folder_id} -O {dest}")
    if not (dest / "adapter_model.safetensors").exists():
        print(f"[eval] adapter download failed (rc={r.returncode}) — all three fetch paths exhausted",
              flush=True)
        return None
    return str(dest)


def load(base, adapter_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def generate(model, tok, messages, max_new_tokens):
    import torch
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-parent", required=True, help="Drive folder id (shared) with the adapter")
    ap.add_argument("--adc", default=None, help="unused placeholder for symmetry with other tools")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--deadline-epoch", type=float, default=0.0,
                    help="absolute wall-clock epoch (from SESSION creation) after which to stop "
                         "generating and finalize the partial result set (0 = no wall)")
    args = ap.parse_args()

    if args.skip_install:
        verify_deps()
    else:
        deps()
    adapter_path = fetch_adapter(args.adapter_parent, "/content/adapter_eval",
                                 adc=args.adc)
    if not adapter_path:
        print("[eval] no adapter — aborting", flush=True)
        return 1
    print(f"[eval] adapter: {adapter_path}", flush=True)

    rows = []
    with open(args.rows) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[eval] {len(rows)} held-out rows, greedy decoding, "
          f"max_new_tokens={args.max_new_tokens}", flush=True)

    model, tok = load(args.base, adapter_path)
    print("[eval] model + adapter loaded", flush=True)

    done = 0
    wall_hit = False
    with open(args.out, "w") as fh:
        for row in rows:
            # Free-tier VMs are recycled without warning; stopping on the wall
            # keeps the comparisons already generated (scoring only needs the
            # rows written so far) instead of losing the whole session.
            if args.deadline_epoch and time.time() > args.deadline_epoch:
                wall_hit = True
                print(f"[eval] SESSION DEADLINE reached after {done}/{len(rows)} rows — "
                      f"finalizing the partial set", flush=True)
                break
            msgs = row["messages"]
            user_only = [m for m in msgs if m.get("role") != "assistant"] or msgs
            adapter_out, t_a = generate(model, tok, user_only, args.max_new_tokens)
            try:
                with model.disable_adapter():
                    base_out, t_b = generate(model, tok, user_only, args.max_new_tokens)
            except Exception as e:
                base_out, t_b = f"<disable_adapter failed: {e}>", 0.0
            fh.write(json.dumps({**row, "adapter_out": adapter_out, "base_out": base_out,
                                 "t_adapter_s": round(t_a, 1), "t_base_s": round(t_b, 1)}) + "\n")
            fh.flush()
            done += 1
            if done % 5 == 0:
                print(f"[eval] {done}/{len(rows)} rows done", flush=True)
    print(f"[eval] wrote {done} comparisons -> {args.out}"
          f"{' (partial: session wall)' if wall_hit else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
