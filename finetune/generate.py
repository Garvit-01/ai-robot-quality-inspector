#!/usr/bin/env python3
"""Run Cosmos-Reason1-7B (base, or base + a LoRA adapter) on evaluation examples. GPU VM.

Writes one line per example: the raw reply plus timing. Parsing and scoring happen back on
the project machine (finetune/score_generations.py) with the SAME parser and scorer as
every other benchmark, so the fine-tuned model is judged exactly like Gemma was.

Usage (on the VM):
    .venv/bin/python generate.py --data eval.jsonl --images . --out gen_base.jsonl
    .venv/bin/python generate.py --data eval.jsonl --images . --adapter adapter --out gen_tuned.jsonl

On a smaller GPU (the 15 GB T4, which has no bf16), load the weights compressed:
    python generate.py ... --adapter adapter --quant 4bit --out gen_tuned_4bit.jsonl
The adapter then stays separate (it can't be merged into compressed weights), and the
held-out score must be re-checked: compression changes the numbers slightly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from train_lora import MAX_PIXELS, MODEL, messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="room for a <think> block: Cosmos-Reason1 is a reasoning model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quant", choices=("none", "8bit", "4bit"), default="none",
                    help="compress the weights to fit a smaller GPU (computes in fp16)")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    if args.quant == "none":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    else:
        from transformers import BitsAndBytesConfig
        bnb = (BitsAndBytesConfig(load_in_8bit=True) if args.quant == "8bit" else
               BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda",
            quantization_config=bnb)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
        if args.quant == "none":
            model = model.merge_and_unload()  # same maths, faster generation
    model.eval()
    print(f"[gen] weights {args.quant}, GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    data = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    if args.limit:
        data = data[: args.limit]
    tag = "tuned" if args.adapter else "base"
    print(f"[gen] {len(data)} examples, {tag} model", flush=True)
    with args.out.open("w") as fh:
        for i, ex in enumerate(data, 1):
            msgs = messages(ex, args.images, with_answer=False)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            images, _ = process_vision_info(msgs)
            inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")
            t0 = time.monotonic()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            ms = (time.monotonic() - t0) * 1000
            new = out[0, inputs["input_ids"].shape[1]:]
            raw = processor.decode(new, skip_special_tokens=True)
            fh.write(json.dumps({"seed": ex["seed"], "raw": raw, "latency_ms": round(ms),
                                 "tokens_in": int(inputs["input_ids"].shape[1]),
                                 "tokens_out": int(new.shape[0])}) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(data):
                print(f"[gen] {i}/{len(data)} last {ms:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
