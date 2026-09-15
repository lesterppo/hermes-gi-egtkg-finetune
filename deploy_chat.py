#!/usr/bin/env python3
"""
deploy_chat.py — run the trained GI/hepatology LoRA as a small local model on a
free Colab T4 and answer evidence-grounded questions.

The daily training pipeline (see daily_finetune.py + run_daily.py) writes the
newest adapter into the Drive 'adapter_in' continuity folder under gi-egtkg/.
This script mounts Drive, pulls that adapter, loads Llama-3.1-8B-Instruct in
NF4 QLoRA and merges the LoRA, then answers medical questions.

Usage (Colab VM, GPU runtime):
    python deploy_chat.py --adc /content/gdrive_adc.json \
        --adapter-parent <your adapter_in folder id>
    # then type questions; answer model uses no retrieval here (parametric).
    # For grounded/retrieval QA, run it through the EGT-KG runner instead.

Run on the same free tier used for training:
    colab run --gpu T4 --session deploy < etc.
Single-file; deps install on first run (transformers/peft/bnb/accelerate).
"""
import argparse
import json
import os
import pathlib
import sys


def run(cmd):
    print(f"$ {cmd}", flush=True)
    return os.system(cmd)


def deps():
    run(f"{sys.executable} -m pip uninstall -q -y torchao")
    run(f"{sys.executable} -m pip install -q "
        f"'transformers>=4.46,<5.0' 'peft>=0.7' "
        f"'bitsandbytes>=0.41' 'accelerate>=0.25' 'safetensors' 'gdown' "
        f"'google-api-python-client' 'google-auth' 'google-auth-httplib2'")


def newest_adapter_local():
    """Return local adapter dir (pulled via gdown from a shared Drive folder)
    or None."""
    base = "/content/adapter_local"
    pathlib.Path(base).mkdir(exist_ok=True)
    ad = pathlib.Path(base) / "adapter_model.safetensors"
    return base if ad.exists() and ad.stat().st_size > 0 else None


def load_model(adapter_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto",
        low_cpu_mem_usage=True)
    if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"[load] applied LoRA adapter from {adapter_dir}", flush=True)
    else:
        print("[load] no adapter — running base model", flush=True)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-parent",
                    default=os.environ.get("DRIVE_ADAPTER_IN", ""),
                    help="Drive folder id containing adapter_model.safetensors")
    ap.add_argument("--gdrive-py", default="/content/gdrive.py")
    args = ap.parse_args()

    if not os.path.exists("/content/adapter_local/adapter_model.safetensors"):
        # pull latest adapter via gdown (folder must be shared anyone-with-link)
        if args.adapter_parent:
            os.makedirs("/content/adapter_local", exist_ok=True)
            r = run(f"gdown --folder {args.adapter_parent} -O /content/adapter_local "
                    f"|| echo GDOWN_FAIL")
            if r != 0:
                print("[pull] gdown fallback failed; continuing without adapter", flush=True)

    try:
        model, tok = load_model(newest_adapter_local())
    except Exception as e:
        print(f"[load] failed: {e}", flush=True)
        sys.exit(1)

    gen = tok.model_max_length if hasattr(tok, "model_max_length") else 2048
    print("\n[ready] GI/hepatology local model. Type a question, Ctrl+D to exit.\n", flush=True)
    for line in sys.stdin:
        q = line.rstrip("\n").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        msgs = [{"role": "user", "content": q}]
        inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=220, do_sample=False,
                             temperature=None, top_p=None)
        ans = tok.decode(out[0][inp.shape[-1]:], skip_special_tokens=True).strip()
        print("\n" + ans + "\n---", flush=True)


if __name__ == "__main__":
    main()
