#!/usr/bin/env python3
"""Run NVIDIA Nemotron-Nano-12B-v2-VL zero-shot on evaluation examples. GPU VM.

The per-token Nemotron models on Token Factory are text-only, so the vision Nemotron runs
on our own Nebius VM, following its model card (transformers + trust_remote_code, with
the mamba-ssm / causal-conv1d kernels it needs). It gets the SAME examples as Cosmos in
generate.py: same system prompt, same checkpoint question, same two images, and it writes
the same JSONL, so finetune/score_generations.py scores it identically.

It lives in its own environment (~/nvl/.venv on the VM): it needs transformers 4.53, while
the Cosmos fine-tune uses a newer version.

Usage (on the VM):
    ~/nvl/.venv/bin/python generate_nemotron_vl.py --data eval.jsonl --images scenes --out gen_nemotron_vl.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, device_map="cuda", torch_dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    data = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    if args.limit:
        data = data[: args.limit]
    print(f"[gen] {len(data)} examples, {MODEL}", flush=True)
    with args.out.open("w") as fh:
        for i, ex in enumerate(data, 1):
            images = [Image.open(args.images / p).convert("RGB") for p in ex["images"]]
            # "/no_think" is the model card's switch for a direct answer (no reasoning trace);
            # our verifier instructions follow it in the same system message.
            msgs = [{"role": "system", "content": "/no_think\n" + ex["system"]},
                    {"role": "user", "content": [{"type": "image", "image": ""} for _ in images]
                     + [{"type": "text", "text": "\n" + ex["user"]}]}]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=images, return_tensors="pt").to("cuda")
            t0 = time.monotonic()
            with torch.no_grad():
                out = model.generate(pixel_values=inputs.pixel_values, input_ids=inputs.input_ids,
                                     attention_mask=inputs.attention_mask, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, eos_token_id=tokenizer.eos_token_id)
            ms = (time.monotonic() - t0) * 1000
            n_in = inputs.input_ids.shape[1]
            # Depending on the remote code, generate() returns the prompt + answer or the answer only.
            new = out[0, n_in:] if out.shape[1] > n_in and torch.equal(out[0, :n_in], inputs.input_ids[0]) else out[0]
            raw = tokenizer.decode(new, skip_special_tokens=True)
            fh.write(json.dumps({"seed": ex["seed"], "raw": raw, "latency_ms": round(ms),
                                 "tokens_in": int(n_in), "tokens_out": int(new.shape[0])}) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(data):
                print(f"[gen] {i}/{len(data)} last {ms:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
