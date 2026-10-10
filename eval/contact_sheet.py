#!/usr/bin/env python3
"""Tile rendered scenes into one image, each captioned with its ground-truth label.

Looking at the renders is the cheapest test the stager has: a scene whose picture
doesn't show what its label claims would poison every number computed from it.

Usage:
    .venv/bin/python eval/contact_sheet.py runs/scenes/smoke2 --view closeup
    .venv/bin/python eval/contact_sheet.py runs/scenes/smoke2 --view overview --cols 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--view", choices=("overview", "closeup", "both"), default="closeup")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=400, help="tile width in px")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels = sorted((args.scene_dir / "labels").glob("*.json"), key=lambda p: int(p.stem))
    views = ("overview", "closeup") if args.view == "both" else (args.view,)
    tiles = []
    for lp in labels:
        lab = json.loads(lp.read_text())
        sc = lab["scene"]
        e = sc["expected"]
        caption = f"{lab['seed']} {sc['fault']} {sc['checkpoint'].replace('CP_', '')}\n{e['failure_type']} @ {e['location_hint']}"
        extra = sc.get("truth") or {}
        if extra:
            caption += "\n" + " ".join(f"{k}={v}" for k, v in extra.items())
        for v in views:
            im = Image.open(args.scene_dir / lab["images"][v]).convert("RGB")
            h = round(im.height * args.tile / im.width)
            im = im.resize((args.tile, h))
            canvas = Image.new("RGB", (args.tile, h + 48), (255, 255, 255))
            canvas.paste(im, (0, 48))
            ImageDraw.Draw(canvas).multiline_text((6, 2), caption + ("" if len(views) == 1 else f"  [{v}]"),
                                                  fill=(0, 0, 0), spacing=1)
            tiles.append(canvas)

    if not tiles:
        raise SystemExit(f"no labelled scenes in {args.scene_dir}")
    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (cols * tw, rows * th), (230, 230, 230))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    out = args.out or args.scene_dir / f"contact_{args.view}.png"
    sheet.save(out)
    print(f"{len(tiles)} tiles -> {out}")


if __name__ == "__main__":
    main()
