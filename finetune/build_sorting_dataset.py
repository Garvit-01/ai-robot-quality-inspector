#!/usr/bin/env python3
"""Turn sorting-cell runs into fine-tuning examples (JSONL), two per part: the top view and
the underside view, each with its own correct answer.

Why: the verifier, trained only on the original dataset renders, called every good M16 nut
"too small" when the UR10 sorting cell showed it (Oct 13), and per-part relighting only got
2/9 good nuts through (Oct 14). The cell looks slightly different (the real arm, its shadows,
other parts in view), and the underside view is new altogether. So the cell's own photos go
into training. Rendered with `run_sorting.py --verifier oracle` (the arm and cameras do
everything; only the decision comes from ground truth), on the `train` tray split.

Same format as build_dataset.py (system, user, images, target, ...), so train_lora.py,
generate.py and score_generations.py take it unchanged. Prompts come from
selfheal/sorting.prompts(), the exact strings the sorting line sends.

Usage:
    .venv/bin/python finetune/build_sorting_dataset.py runs/sorting/train_data --out finetune/data/sort_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from selfheal.sorting import prompts  # noqa: E402

TOP = {
    "good": "An undamaged M16 nut sits squarely inside the printed outline.",
    "wrong:bolt_m16": "The part is a bolt with a threaded shaft, not a nut.",
    "wrong:nut_m12": "The nut is too small: it leaves a wide gap to the outline on every side.",
    "wrong:nut_m20": "The nut is too big: it covers the printed outline on several sides.",
    "chip": "One corner of the nut is cut off, showing a bright slanted facet.",
    "crack": "A thin dark crack runs across one side face of the nut.",
    "rust": "Orange-brown rust patches cover part of the nut's surface.",
}
UNDER = {
    "good": "An undamaged M16 nut sits squarely against the blue suction cup.",
    "wrong:bolt_m16": "Seen from below, this is a bolt with a round head and a shaft, not a nut.",
    "wrong:nut_m12": "Seen from below, the nut is clearly smaller than an M16 nut.",
    "wrong:nut_m20": "Seen from below, the nut is clearly larger than an M16 nut.",
    "chip": "Seen from below, one corner of the nut is cut off.",
    "crack": "A thin dark crack runs across the nut's bottom face.",
    "rust": "Orange-brown rust patches cover the nut's bottom face.",
}


def answer(rec: dict, view: str) -> tuple[str, str]:
    """(failure_type, evidence) a perfect inspector gives for this part from this view.
    A wrong part looks wrong from both sides; damage only from its own side."""
    if rec["kind"] == "wrong":
        key = f"wrong:{rec['part']}"
        return "wrong_part", (TOP if view == "top" else UNDER)[key]
    if rec["kind"] == "defective" and rec["side"] == view:
        return "defective_part", (TOP if view == "top" else UNDER)[rec["defect"]]
    return "none", (TOP if view == "top" else UNDER)["good"]


def examples(run: Path, cfg: dict) -> list[dict]:
    system, top_q, under_q = prompts(cfg)
    out = []
    for line in (run / "parts.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not (r.get("held_from_feed") and r.get("held_from_nest")):
            continue  # the arm dropped it: the photos don't show the part as labelled
        stem = f"{r['tray']}_{r['slot']}"
        for view, user, imgs, loc in (
                ("top", top_q, [f"images/{stem}_top_overview.png", f"images/{stem}_top_closeup.png"], "carrier"),
                ("underside", under_q, [f"images/{stem}_under_underside.png"], "gripper")):
            ft, ev = answer(r, view)
            target = json.dumps({"checkpoint": "CP_PRE_GRASP", "step_ok": ft == "none", "failure_type": ft,
                                 "evidence": ev, "location_hint": loc, "confidence": 0.9})
            out.append({"seed": f"{stem}_{view}", "view": view, "kind": r["kind"], "part": r["part"],
                        "defect": r["defect"], "side": r["side"], "checkpoint": "CP_PRE_GRASP",
                        "system": system, "user": user,
                        "images": [f"{run.name}/{p}" for p in imgs],  # relative to the runs' parent folder
                        "target": target,
                        "expected": {"checkpoint": "CP_PRE_GRASP", "step_ok": ft == "none",
                                     "failure_type": ft, "location_hint": loc}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+", help="run_sorting.py output folders (oracle runs)")
    ap.add_argument("--config", type=Path, default=REPO / "config" / "selfheal_fastener.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rows = [e for run in args.runs for e in examples(run, cfg)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(e) + "\n" for e in rows))
    by = {}
    for e in rows:
        k = (e["view"], json.loads(e["target"])["failure_type"])
        by[k] = by.get(k, 0) + 1
    print(f"{len(rows)} examples -> {args.out}  {dict(sorted(by.items()))}")


if __name__ == "__main__":
    main()
