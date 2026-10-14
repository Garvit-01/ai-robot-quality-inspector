#!/usr/bin/env python3
"""LoRA fine-tune of NVIDIA Cosmos-Reason1-7B as the work-cell verifier. Runs on the GPU VM.

Input: JSONL from finetune/build_dataset.py (exact verifier prompt + images + the correct
Verdict JSON). The loss is computed ONLY on the answer tokens; the prompt and images are
context. LoRA adapters go on both the language model and the vision encoder's blocks: the
zero-shot failures (nut size, small chips, a nut caught on a rim) are perception, so the
vision side has to be able to adapt too.

Usage (on the VM):
    .venv/bin/python train_lora.py --data train.jsonl --images . --out adapter --epochs 2
    .venv/bin/python train_lora.py --data dev.jsonl --images . --out /tmp/smoke --max-steps 3   # smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

MODEL = "nvidia/Cosmos-Reason1-7B"
# Images are 1024x768. Cap the visual token budget at that size so nothing is downscaled
# (a 4 mm chip is ~35 px wide in the close-up).
MAX_PIXELS = 1024 * 768
# LLM projections everywhere except the vision tower, plus the vision blocks' attention and MLP.
LLM_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
VISION_TARGETS = r".*visual\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.gate_proj|mlp\.up_proj|mlp\.down_proj)$"
TARGETS = f"{LLM_TARGETS}|{VISION_TARGETS}"  # the Oct 12 run: language model AND vision encoder


def messages(ex: dict, root: Path, with_answer: bool) -> list[dict]:
    content = [{"type": "image", "image": str(root / p), "max_pixels": MAX_PIXELS} for p in ex["images"]]
    content.append({"type": "text", "text": ex["user"]})
    msgs = [{"role": "system", "content": [{"type": "text", "text": ex["system"]}]},
            {"role": "user", "content": content}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]})
    return msgs


def encode(processor, ex: dict, root: Path) -> dict:
    """Full conversation tokens, with labels = -100 on everything but the answer."""
    full = processor.apply_chat_template(messages(ex, root, True), tokenize=False)
    prompt = processor.apply_chat_template(messages(ex, root, False), tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages(ex, root, False))
    batch = processor(text=[full], images=images, return_tensors="pt")
    n_prompt = processor(text=[prompt], images=images, return_tensors="pt")["input_ids"].shape[1]
    labels = batch["input_ids"].clone()
    labels[:, :n_prompt] = -100
    batch["labels"] = labels
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None, help="optional JSONL for a held-out loss each epoch")
    ap.add_argument("--images", type=Path, required=True, help="directory the examples' image paths are relative to")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many examples (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-vision", action="store_true",
                    help="LoRA on the language model only; the vision encoder stays exactly as NVIDIA trained it. "
                         "Research (Oct 14): updating the vision encoder is the likeliest way to learn one renderer's look")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save the adapter every N examples (0 = only at epoch ends), to pick a checkpoint on dev")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    val = [json.loads(l) for l in args.val.read_text().splitlines() if l.strip()] if args.val else []
    print(f"[train] {len(data)} examples, {len(val)} validation, model {MODEL}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=LLM_TARGETS if args.freeze_vision else TARGETS,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps_total = math.ceil(len(data) * args.epochs)
    if args.max_steps:
        steps_total = min(steps_total, args.max_steps)
    opt_steps = math.ceil(steps_total / args.grad_accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, opt_steps // 20), opt_steps)

    def val_loss() -> float | None:
        if not val:
            return None
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for ex in val:
                b = {k: v.to("cuda") for k, v in encode(processor, ex, args.images).items()}
                tot += model(**b).loss.item()
        model.train()
        return tot / len(val)

    model.train()
    order: list[int] = []
    t0 = time.time()
    running = 0.0
    log_every = 10
    for step in range(steps_total):
        if not order:
            order = list(range(len(data)))
            random.shuffle(order)
        ex = data[order.pop()]
        b = {k: v.to("cuda") for k, v in encode(processor, ex, args.images).items()}
        loss = model(**b).loss / args.grad_accum
        loss.backward()
        running += loss.item() * args.grad_accum
        if (step + 1) % args.grad_accum == 0 or step + 1 == steps_total:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0 or step + 1 == steps_total:
            el = time.time() - t0
            eta = el / (step + 1) * (steps_total - step - 1)
            print(f"[train] {step + 1}/{steps_total} loss {running / min(log_every, step + 1):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {el / (step + 1):.2f}s/ex eta {eta / 60:.1f} min "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            running = 0.0
        if args.save_every and (step + 1) % args.save_every == 0:
            model.save_pretrained(args.out / f"step{step + 1}")
        if (step + 1) % len(data) == 0:
            v = val_loss()
            print(f"[train] epoch {(step + 1) // len(data)} done; val loss {v}", flush=True)
            model.save_pretrained(args.out / f"epoch{(step + 1) // len(data)}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    v = val_loss()
    print(f"[train] DONE in {(time.time() - t0) / 60:.1f} min; final val loss {v}; adapter -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
