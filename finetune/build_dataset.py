#!/usr/bin/env python3
"""Turn rendered scenes into fine-tuning / evaluation examples (JSONL).

Every example carries the EXACT prompt the verifier sees in eval/bench_verifier.py — same
system prompt, same checkpoint question, same image order — so a fine-tuned model is
trained and tested on the real task, not a simplified one. The target is the correct
Verdict JSON, which the simulator knows by construction: nothing is hand-labelled.

The output is self-contained (prompt text + relative image paths), so the GPU machine
needs only this file and the images, not the project code.

Usage:
    .venv/bin/python finetune/build_dataset.py runs/scenes/ftrain_v1 --config config/selfheal_fastener.yaml \
        --out finetune/data/train.jsonl
    .venv/bin/python finetune/build_dataset.py runs/scenes/feval_v1 --config config/selfheal_fastener.yaml \
        --out finetune/data/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "scripts"))

from bench_verifier import build_prompt  # noqa: E402  (the same prompt the benchmark sends)

VIEWS = ["overview", "closeup"]


def evidence(scene: dict) -> str:
    """A short, true sentence describing what the image shows — the verdict's `evidence`.

    Written from the ground truth, in the style the prompt asks for ("one short sentence
    describing what you actually see"). The model is trained to say what it sees, then
    decide; teaching it to state the cue that matters is the point.
    """
    f, cp = scene["fault"], scene["checkpoint"]
    t = scene.get("truth") or {}
    if f == "F0":
        return {
            "CP_PRE_GRASP": "An undamaged M16 nut sits squarely inside the printed outline.",
            "CP_POST_LIFT": "The nut is clamped between the two finger pads.",
            "CP_POST_TRANSPORT": "The nut is still clamped between the finger pads above the fixture.",
            "CP_POST_PLACE": "The nut sits flat, down inside the round recess.",
        }[cp]
    if f == "F1":
        if cp == "CP_PRE_GRASP":
            return "The nut is shifted off the printed outline and rotated against it."
        return "The fingers are closed on nothing; the nut is still on the carrier below."
    if f == "F2":
        if scene["part_location"] == "table":
            return "The gripper is empty; the nut is lying on the table."
        return "The gripper is empty and the nut is nowhere in the cell: it fell off the table."
    if f == "F3":
        return "The nut is tilted, resting on the rim of the recess instead of down inside it."
    if f == "F4":
        return {
            "bolt_m16": "The part is a bolt with a threaded shaft, not a nut.",
            "nut_m12": "The nut is too small: it leaves a wide gap to the outline on every side.",
            "nut_m20": "The nut is too big: it covers the printed outline on several sides.",
        }[scene["wrong_shape"]]
    if f == "F5":
        return {
            "chip": "One corner of the nut is cut off, showing a bright slanted facet.",
            "crack": "A thin dark crack runs across one side face of the nut.",
            "rust": "Orange-brown rust patches cover part of the nut's surface.",
        }[t.get("defect") or scene.get("defect_type")]
    raise ValueError(f"unknown fault {f}")


def target(scene: dict) -> str:
    e = scene["expected"]
    return json.dumps({
        "checkpoint": e["checkpoint"], "step_ok": e["step_ok"], "failure_type": e["failure_type"],
        "evidence": evidence(scene), "location_hint": e["location_hint"], "confidence": 0.9,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    labels = sorted((args.scene_dir / "labels").glob("*.json"), key=lambda p: int(p.stem))
    if not labels:
        raise SystemExit(f"no labels in {args.scene_dir}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as fh:
        for lp in labels:
            lab = json.loads(lp.read_text())
            sc = lab["scene"]
            fh.write(json.dumps({
                "seed": lab["seed"],
                "fault": sc["fault"],
                "checkpoint": sc["checkpoint"],
                "system": cfg["prompts"]["system"],
                "user": build_prompt(cfg, sc["checkpoint"], VIEWS),
                # Relative to the scene set's directory name, so the GPU machine can mirror it.
                "images": [f"{args.scene_dir.name}/{lab['images'][v]}" for v in VIEWS],
                "target": target(sc),
                "expected": sc["expected"],
                "truth": sc.get("truth", {}),
            }) + "\n")
            n += 1
    print(f"{n} examples -> {args.out}  (prompt {cfg['prompts'].get('version')})")


if __name__ == "__main__":
    main()
