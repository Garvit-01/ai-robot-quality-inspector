#!/usr/bin/env python3
"""M0 smoke test for the vision verifier.

Sends one image plus a checkpoint question and asserts the reply parses into the strict
Verdict schema. This is the same call path the real verifier node will use, so if this
passes, the Tier 1 -> Tier 2 model swap is a config change and nothing more.

Usage:
  .venv/bin/python scripts/smoke_verifier.py --image runs/frames/foo.jpg \
      --checkpoint CP_POST_PLACE
  .venv/bin/python scripts/smoke_verifier.py --selftest      # no Isaac Sim needed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from _common import (
    REPO_ROOT,
    Verdict,
    client_for,
    image_data_uri,
    load_config,
    load_dotenv,
    require_env,
    strip_to_json,
)

CHECKPOINTS = ["CP_PRE_GRASP", "CP_POST_LIFT", "CP_POST_TRANSPORT", "CP_POST_PLACE"]


def build_prompt(cfg: dict, checkpoint: str) -> str:
    prompts = cfg["prompts"]
    question = prompts["checkpoints"][checkpoint].format(**cfg["part_reference"])
    return f"{question}\n\n{prompts['schema_block']}"


def make_selftest_image(path: Path) -> Path:
    """Draw a crude 'cube tilted in a fixture nest' scene so this runs with no sim.

    Deliberately ugly. It only proves the image path, encoding, and JSON contract work —
    it proves nothing about whether the model can see real faults. That is M2's job.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 600), (118, 118, 124))
    d = ImageDraw.Draw(img)
    d.rectangle([80, 330, 720, 560], fill=(92, 92, 98))          # table
    d.rectangle([300, 380, 500, 520], fill=(58, 58, 64))          # fixture pocket
    d.polygon([(318, 470), (470, 436), (486, 520), (334, 520)],   # cube, tilted
              fill=(176, 176, 170), outline=(44, 44, 44))
    d.polygon([(318, 470), (470, 436), (452, 408), (300, 442)],
              fill=(201, 201, 196), outline=(44, 44, 44))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92)
    return path


def query(client, model_id: str, cfg: dict, checkpoint: str, image_path: Path,
          nudge: bool = False) -> tuple[str, float]:
    v = cfg["verifier"]
    user_text = build_prompt(cfg, checkpoint)
    if nudge:
        user_text += "\n\nReturn valid JSON only. No prose, no markdown, no code fences."

    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model_id,
        temperature=v.get("temperature", 0.0),
        messages=[
            {"role": "system", "content": cfg["prompts"]["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri(
                                image_path,
                                v["image_longest_side_px"],
                                v["jpeg_quality"],
                            )
                        },
                    },
                ],
            },
        ],
    )
    return resp.choices[0].message.content or "", (time.monotonic() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--checkpoint", default="CP_POST_PLACE", choices=CHECKPOINTS)
    ap.add_argument("--selftest", action="store_true",
                    help="generate a throwaway synthetic frame instead of using --image")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config()

    if args.selftest:
        image_path = make_selftest_image(REPO_ROOT / "runs" / "_selftest" / "scene.jpg")
        print(f"selftest image: {image_path}")
    elif args.image:
        image_path = args.image
        if not image_path.exists():
            sys.exit(f"Image not found: {image_path}")
    else:
        sys.exit("Pass --image <path> or --selftest")

    verifier = cfg["verifier"]
    client = client_for(verifier)
    model_id = require_env(verifier["model_id_env"])
    print(f"model:      {model_id}")
    print(f"checkpoint: {args.checkpoint}\n")

    for attempt in range(verifier["max_retries"] + 1):
        try:
            raw, latency_ms = query(
                client, model_id, cfg, args.checkpoint, image_path, nudge=attempt > 0
            )
        except Exception as exc:  # timeout, transport, 4xx/5xx
            print(f"attempt {attempt + 1}: call failed: {type(exc).__name__}: {exc}")
            continue

        print(f"attempt {attempt + 1}: {latency_ms:.0f} ms")
        print(f"raw response:\n{raw}\n")

        cleaned = strip_to_json(raw)
        try:
            verdict = Verdict.model_validate_json(cleaned)
        except (ValidationError, ValueError) as exc:
            print(f"parse/validation failed: {exc}")
            if attempt < verifier["max_retries"]:
                print("retrying with a JSON-only nudge...\n")
            continue

        print("PASS: schema-valid verdict")
        print(json.dumps(verdict.model_dump(), indent=2))
        if verdict.failure_type == "unclear":
            print("\nNote: model said 'unclear'. That is a valid, conservative answer — "
                  "in conditions C/D it triggers a re-query on the alternate view.")
        return 0

    print("\nFAIL: no schema-valid verdict after all attempts.")
    print("Real pipeline behaviour here: failure_type='unclear' -> re-query alternate "
          "view -> still unclear -> stop_and_alert. Never guess.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
