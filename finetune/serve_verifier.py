#!/usr/bin/env python3
"""Serve the fine-tuned verifier on this machine, for the sorting cell to call.

Loads Cosmos-Reason1-7B + our LoRA once (4-bit by default, so it fits the 15 GB T4 next to
Isaac Sim) and answers HTTP POSTs on localhost. Isaac Sim runs in its own Python, so the
simulator and the model never share a process; the simulation simply waits for the reply.

Request  (POST /verify, JSON): {"system": str, "user": str, "images": [absolute PNG paths]}
Response (JSON):              {"raw": str, "latency_ms": int, "tokens_in": int, "tokens_out": int}
GET /health returns {"ok": true, "quant": ...} once the model is loaded.

Usage (the T4 inference venv):
    ~/cosmos_t4/.venv/bin/python finetune/serve_verifier.py --adapter finetune/adapter --port 8765
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from train_lora import MAX_PIXELS, MODEL


def load(adapter: Path | None, quant: str):
    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    kw = dict(torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    if quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return processor, model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--quant", choices=("none", "8bit", "4bit"), default="4bit")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    processor, model = load(args.adapter, args.quant)
    print(f"[serve] {MODEL} {'+ ' + str(args.adapter) if args.adapter else '(base)'} {args.quant}, "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB, http://127.0.0.1:{args.port}", flush=True)

    def verify(req: dict) -> dict:
        content = [{"type": "image", "image": p, "max_pixels": MAX_PIXELS} for p in req["images"]]
        content.append({"type": "text", "text": req["user"]})
        msgs = [{"role": "system", "content": [{"type": "text", "text": req["system"]}]},
                {"role": "user", "content": content}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(msgs)
        inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda")
        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:]
        raw = processor.decode(new, skip_special_tokens=True)
        del inputs, out
        torch.cuda.empty_cache()  # hand the activation memory back: Isaac Sim shares this GPU
        return {"raw": raw, "latency_ms": round((time.monotonic() - t0) * 1000),
                "tokens_in": 0, "tokens_out": int(new.shape[0])}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self._send(200 if self.path == "/health" else 404, {"ok": True, "quant": args.quant})

        def do_POST(self):  # noqa: N802
            if self.path != "/verify":
                return self._send(404, {"error": "unknown path"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, verify(req))
            except Exception as exc:  # noqa: BLE001 — report to the caller, keep serving
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"[:300]})

        def log_message(self, *_):  # quiet: one line per request is printed by the caller
            pass

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
