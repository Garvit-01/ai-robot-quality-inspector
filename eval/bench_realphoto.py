#!/usr/bin/env python3
"""Real-photo check: do our verifiers flag damaged parts in REAL factory photos?

Everything else in this project is rendered. This runs the verifier on MVTec AD, a public
set of real industrial photos with labelled defects, using its two fastener categories:
`metal_nut` (bent, colour, flip, scratch) and `screw` (thread and head damage). Their parts
are not our M16 hex nut, so the only fair question is binary: **is a damaged part flagged
(any failure_type other than "none"), and is a good part left alone?**

The question is NOT the cell's pre-grasp question. That one defines the correct part as an
M16 hex nut half the width of the AprilTag, so every MVTec screw is rightly a "wrong part"
and every MVTec nut fails the hex check: the first run (Oct 14) flagged 100% of good parts
for exactly that reason. Here every model gets the same neutral damage question below, with
the verifier's system prompt and JSON schema unchanged. For our fine-tuned model this is a
prompt it was never trained on, which is part of what is being tested.

MVTec AD is CC BY-NC-SA 4.0: evaluation only, images stay in runs/ (gitignored), never
committed or redistributed. Downloaded from the Voxel51/mvtec-ad mirror on Hugging Face.

Usage:
    # 1. examples in generate.py's format
    .venv/bin/python eval/bench_realphoto.py build --mvtec runs/mvtec --out runs/mvtec/examples.jsonl
    # 2a. a Token Factory model (e.g. Gemma), same call path as the benchmark
    .venv/bin/python eval/bench_realphoto.py api --examples runs/mvtec/examples.jsonl --images runs/mvtec \
        --out runs/mvtec/gen_gemma.jsonl
    # 2b. our fine-tuned model, locally on the T4
    ~/cosmos_t4/.venv/bin/python finetune/generate.py --data runs/mvtec/examples.jsonl --images runs/mvtec \
        --adapter finetune/adapter --quant 4bit --out runs/mvtec/gen_cosmos_tuned.jsonl
    # 3. score either
    .venv/bin/python eval/bench_realphoto.py score --examples runs/mvtec/examples.jsonl \
        --generations runs/mvtec/gen_gemma.jsonl --model "Gemma 3 27B" --out runs/bench/mvtec__gemma
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "eval"))

from _common import Verdict, strip_to_json  # noqa: E402

CONFIG = REPO / "config" / "selfheal_fastener.yaml"
CHECKPOINT = "CP_PRE_GRASP"
PARTS = {"metal_nut": "a stamped steel nut (round body, flat lugs on its rim, a centre hole)",
         "screw": "a steel machine screw (a head and a threaded shaft)"}
QUESTION = """Checkpoint: CP_PRE_GRASP. The image is a close-up photo of one part on a production
line: {part}. Decide only whether this part is physically damaged or defective.
Damage includes: scratches, dents, bent or deformed areas, discolouration or stains, damaged
or missing threads, a damaged head, or the part lying upside down.
If you see any of these: step_ok false, failure_type defective_part.
If the part looks intact and correct: step_ok true, failure_type none.
location_hint is carrier in every case.

You are given 1 image(s): image 1 is a close-up of the part.

"""


def cmd_build(args) -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    subset = json.loads((args.mvtec / "subset.json").read_text())
    with args.out.open("w") as fh:
        for i, s in enumerate(subset):
            fh.write(json.dumps({
                "seed": i, "category": s["category"], "defect": s["defect"], "bad": s["defect"] != "good",
                "checkpoint": CHECKPOINT, "system": cfg["prompts"]["system"],
                "user": QUESTION.format(part=PARTS[s["category"]]) + cfg["prompts"]["schema_block"],
                "images": [s["file"]],
            }) + "\n")
    print(f"{len(subset)} examples -> {args.out}")


def cmd_api(args) -> None:
    from _common import client_for, load_dotenv, require_env
    from bench_verifier import call_verifier
    load_dotenv()
    cfg = yaml.safe_load(CONFIG.read_text())
    client = client_for(cfg["verifier"])
    model = args.model or require_env(cfg["verifier"]["model_id_env"])
    examples = [json.loads(l) for l in args.examples.read_text().splitlines() if l.strip()]

    def job(ex):
        r = call_verifier(client, model, cfg, ex["user"], [args.images / p for p in ex["images"]], args.max_tokens)
        return {"seed": ex["seed"], "raw": r["raw"] or f"<error {r['error']}>", "latency_ms": round(r["latency_ms"]),
                "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"]}

    with ThreadPoolExecutor(args.workers) as pool, args.out.open("w") as fh:
        for i, row in enumerate(pool.map(job, examples), 1):
            fh.write(json.dumps(row) + "\n")
            if i % 25 == 0 or i == len(examples):
                print(f"[api] {i}/{len(examples)} {model}", flush=True)


def flagged(raw: str) -> tuple[bool | None, str | None]:
    """(flagged, failure_type); (None, None) if the reply doesn't parse."""
    try:
        v = Verdict.model_validate_json(strip_to_json(raw))
    except Exception:  # noqa: BLE001 — any unparseable reply is counted, not fatal
        return None, None
    return v.failure_type != "none", v.failure_type


def cmd_score(args) -> None:
    examples = {json.loads(l)["seed"]: json.loads(l) for l in args.examples.read_text().splitlines() if l.strip()}
    gens = [json.loads(l) for l in args.generations.read_text().splitlines() if l.strip()]
    per = defaultdict(lambda: {"good": 0, "good_flagged": 0, "bad": 0, "bad_flagged": 0, "unparseable": 0,
                               "labels": Counter()})
    rows = []
    for g in gens:
        ex = examples[g["seed"]]
        f, ft = flagged(g["raw"])
        c = per[ex["category"]]
        if f is None:
            c["unparseable"] += 1
        c["labels"][ft] += 1
        if ex["bad"]:
            c["bad"] += 1
            c["bad_flagged"] += bool(f)
        else:
            c["good"] += 1
            c["good_flagged"] += bool(f)
        rows.append({"seed": g["seed"], "category": ex["category"], "defect": ex["defect"],
                     "flagged": f, "failure_type": ft, "raw": g["raw"][:600]})

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    summary = {"model": args.model, "categories": {}}
    lines = [f"# Real-photo check (MVTec AD) — `{args.model}`", "",
             "Binary: a damaged part should be flagged (any failure_type except none); a good part should not.",
             "Unparseable replies count as not flagged.", "",
             "| Category | Damaged caught | Good wrongly flagged | Unparseable | Answers given |",
             "|---|---|---|---|---|"]
    for cat, c in sorted(per.items()):
        caught, false = pct(c["bad_flagged"], c["bad"]), pct(c["good_flagged"], c["good"])
        summary["categories"][cat] = {**{k: v for k, v in c.items() if k != "labels"},
                                      "caught_pct": caught, "false_alarm_pct": false,
                                      "labels": dict(c["labels"])}
        top = ", ".join(f"{k}: {v}" for k, v in c["labels"].most_common(4))
        lines.append(f"| {cat} | {c['bad_flagged']}/{c['bad']} ({caught}%) | {c['good_flagged']}/{c['good']} ({false}%) "
                     f"| {c['unparseable']} | {top} |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    with (args.out / "results.jsonl").open("w") as fh:
        for r in sorted(rows, key=lambda r: r["seed"]):
            fh.write(json.dumps(r) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--mvtec", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("api")
    a.add_argument("--examples", type=Path, required=True)
    a.add_argument("--images", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--model", default=None, help="default: VERIFIER_MODEL_ID from .env")
    a.add_argument("--workers", type=int, default=3)
    a.add_argument("--max-tokens", type=int, default=400)
    s = sub.add_parser("score")
    s.add_argument("--examples", type=Path, required=True)
    s.add_argument("--generations", type=Path, required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"build": cmd_build, "api": cmd_api, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
