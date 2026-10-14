#!/usr/bin/env python3
"""Compare vision models as the verifier, on the same scenes, prompt and scorer.

For every model you list it runs, through Nebius Token Factory:
  sim    the 120 held-out rendered scenes  -> runs/bench/feval_v1__<tag>/   (eval/bench_verifier.py)
  real   the 275 real MVTec AD photos       -> runs/bench/mvtec__<tag>/      (eval/bench_realphoto.py)

    .venv/bin/python eval/compare_models.py                                   # default list, sim + real
    .venv/bin/python eval/compare_models.py --sets sim --models moonshotai/Kimi-K3 MiniMaxAI/MiniMax-M3
    .venv/bin/python eval/compare_models.py --table                           # just print every result so far

Costs Token Factory credit (cents to ~$1 per model per set). Reasoning models think before
answering, so replies get a 1,500-token budget; unreadable replies count as wrong. Model IDs:
`scripts/smoke_nemotron.py --list`; only "Public" vision models work per token.

Long runs: start it in the background so it survives a closed VS Code window, e.g.
    nohup .venv/bin/python eval/compare_models.py > runs/compare_models.log 2>&1 &
notebooks/compare_models.ipynb shows the results as a table and chart.

Our own fine-tuned model isn't on Token Factory; it runs locally on the T4:
    ~/cosmos_t4/.venv/bin/python finetune/generate.py --data finetune/data/eval.jsonl \\
        --images runs/scenes --adapter finetune/adapter --quant 4bit --out <file>
then finetune/score_generations.py (sim) or `bench_realphoto.py score` (real).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / "bin" / "python")
DEFAULT_MODELS = ["google/gemma-3-27b-it", "moonshotai/Kimi-K3", "MiniMaxAI/MiniMax-M3", "Qwen/Qwen3.8-27B",
                  "deepseek-ai/DeepSeek-V4.1-Flash", "zai-org/GLM-5.3-Flash", "moonshotai/Kimi-K2.6"]


def tag(model: str) -> str:
    """Folder-safe name: moonshotai/Kimi-K3 -> kimi_k3."""
    return re.sub(r"[^a-z0-9]", "_", model.split("/", 1)[-1].lower())


def run(cmd: list[str]) -> None:
    print("   $", " ".join(cmd[1:4]), "...", flush=True)
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    lines = [l for l in (out.stdout + out.stderr).splitlines() if l.strip()]
    print("  ", "\n   ".join(lines[-3:]), flush=True)


def sim(model: str) -> None:
    run([PY, "eval/bench_verifier.py", "runs/scenes/feval_v1", "--config", "config/selfheal_fastener.yaml",
         "--model", model, "--max-tokens", "1500", "--workers", "3", "--out", f"runs/bench/feval_v1__{tag(model)}"])


def real(model: str) -> None:
    examples = REPO / "runs" / "mvtec" / "examples.jsonl"
    if not examples.exists():
        run([PY, "eval/bench_realphoto.py", "build", "--mvtec", "runs/mvtec", "--out", str(examples)])
    gen = f"runs/mvtec/gen_{tag(model)}.jsonl"
    run([PY, "eval/bench_realphoto.py", "api", "--examples", str(examples), "--images", "runs/mvtec",
         "--model", model, "--max-tokens", "1500", "--out", gen])
    run([PY, "eval/bench_realphoto.py", "score", "--examples", str(examples), "--generations", gen,
         "--model", f"{model} (zero-shot)", "--out", f"runs/bench/mvtec__{tag(model)}"])


# Run folders whose name isn't the model's tag (older runs, named before this script existed).
SIM_ALIASES = {"fastener-v1": "gemma_3_27b_it"}
REAL_ALIASES = {"gemma": "gemma_3_27b_it", "cosmos_tuned": "cosmos_tuned_t4_4bit"}  # real photos ran on the T4


def table() -> list[dict]:
    """Every scored run so far, sim and real, one row per model."""
    rows: dict[str, dict] = {}
    for f in sorted((REPO / "runs" / "bench").glob("feval_v1__*/summary.json")):
        s = json.loads(f.read_text())
        name = f.parent.name.split("__", 1)[1]
        name = SIM_ALIASES.get(name, name)
        rows.setdefault(name, {"run": name})
        rows[name].update(model=s.get("meta", {}).get("model", name), sim_correct=s["overall"]["failure_type_acc"],
                          sim_false_alarms=s.get("false_alarm_rate_on_clean"),
                          sim_missed=s.get("miss_rate_on_faults"), sim_unreadable=s["overall"]["unparseable"])
    for f in sorted((REPO / "runs" / "bench").glob("mvtec__*/summary.json")):
        s = json.loads(f.read_text())
        name = f.parent.name.split("__", 1)[1]
        name = REAL_ALIASES.get(name, name)
        r = rows.setdefault(name, {"run": name, "model": s.get("model", name)})
        for cat, c in s["categories"].items():
            r[f"real_{cat}_caught"] = c["caught_pct"]
            r[f"real_{cat}_false_alarm"] = c["false_alarm_pct"]
    return list(rows.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--sets", nargs="*", choices=("sim", "real"), default=["sim", "real"])
    ap.add_argument("--table", action="store_true", help="only print the results collected so far")
    args = ap.parse_args()
    if not args.table:
        for m in args.models:
            for which in args.sets:
                print(f"=== {which}: {m} {time.strftime('%H:%M')} UTC", flush=True)
                (sim if which == "sim" else real)(m)
    for r in table():
        print({k: v for k, v in r.items() if v is not None})


if __name__ == "__main__":
    sys.exit(main())
