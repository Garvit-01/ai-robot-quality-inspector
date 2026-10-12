#!/usr/bin/env python3
"""Score raw replies from finetune/generate.py with the benchmark's own parser and scorer.

Produces the same results.jsonl / summary.json / summary.md as eval/bench_verifier.py, so
a fine-tuned Cosmos-Reason1 run sits in the same tables, reports and A/B/C/D policy
comparison (eval/bench_decisions.py) as the Gemma runs, measured identically.

Usage:
    .venv/bin/python finetune/score_generations.py finetune/data/eval.jsonl gen_tuned.jsonl \
        --scene-dir runs/scenes/feval_v1 --model "Cosmos-Reason1-7B + LoRA" --out runs/bench/feval_v1__cosmos_tuned
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "scripts"))

from pydantic import ValidationError  # noqa: E402

from _common import Verdict, strip_to_json  # noqa: E402
from bench_verifier import render_md, summarise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("examples", type=Path, help="the eval JSONL given to generate.py")
    ap.add_argument("generations", type=Path, help="generate.py's output")
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--model", required=True, help="label for reports")
    ap.add_argument("--prompt-version", default="fastener-v1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    examples = {json.loads(l)["seed"]: json.loads(l) for l in args.examples.read_text().splitlines() if l.strip()}
    gens = [json.loads(l) for l in args.generations.read_text().splitlines() if l.strip()]
    rows = []
    for g in gens:
        ex = examples[g["seed"]]
        pred, err = None, None
        try:
            pred = Verdict.model_validate_json(strip_to_json(g["raw"])).model_dump()
        except (ValidationError, ValueError) as exc:
            err = f"parse: {str(exc)[:200]}"
        rows.append({"seed": g["seed"], "fault": ex["fault"], "checkpoint": ex["checkpoint"],
                     "exp": ex["expected"], "truth": ex.get("truth", {}), "pred": pred,
                     "attempts": 1, "latency_ms": g["latency_ms"], "tokens_in": g["tokens_in"],
                     "tokens_out": g["tokens_out"], "error": err, "raw": g["raw"][:2000]})
    if len(rows) != len(examples):
        print(f"[score] warning: {len(rows)} replies for {len(examples)} examples", flush=True)
    rows.sort(key=lambda r: r["seed"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {"scene_dir": str(args.scene_dir), "model": args.model, "views": ["overview", "closeup"],
            "prompt_version": args.prompt_version, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    s = summarise(rows)
    s["meta"] = meta
    (args.out / "summary.json").write_text(json.dumps(s, indent=2))
    (args.out / "summary.md").write_text(render_md(meta, s))
    o = s["overall"]
    print(f"[score] {args.model}: failure_type {o['failure_type_acc']}% · step_ok {o['step_ok_acc']}% · "
          f"unparseable {o['unparseable']} -> {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
