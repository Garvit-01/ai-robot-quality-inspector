#!/usr/bin/env python3
"""Turn a filmed sorting run into a demo video: the wide view of the cell, the inspection
photos of the part in hand, the inspector's verdicts with its one-line reasons, and the
running tally against today's factory.

Reads what `isaac/run_sorting.py --film DIR` wrote: DIR/000000.jpg... (one frame per 4
physics steps, i.e. real time at 15 fps) and DIR/marks.jsonl (the frame at which each photo
was taken and each decision made). The simulation pauses while the model thinks, so model
time doesn't appear in the video; a caption says so.

    .venv/bin/python eval/film_overlay.py runs/sorting/film_demo/frames --out runs/sorting/film_demo/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK, MUTED, BG, PANEL = (235, 238, 242), (150, 158, 168), (22, 26, 31), (34, 39, 46)
GOOD, BAD, WARN = (88, 196, 120), (232, 92, 80), (242, 183, 5)
LABEL = {"none": "OK", "wrong_part": "WRONG PART", "defective_part": "DAMAGED", "unclear": "UNCLEAR"}


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def evidence(verdict: dict) -> str:
    try:
        return json.loads(verdict.get("raw", "")).get("evidence", "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def wrap(draw, text, f, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = (line + " " + w).strip()
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            lines.append(line)
            line = w
    return lines + ([line] if line else [])


def truth(r: dict) -> str:
    if r["kind"] == "good":
        return "good M16 nut"
    if r["kind"] == "wrong":
        return "wrong part: " + r["part"].replace("_", " ").upper()
    return f"{r['defect']} on the {r['side']}"


def compose(frame: Image.Image, state: dict) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(frame.resize((1280, 720)), (0, 0))
    d = ImageDraw.Draw(canvas)

    # Right column: the two inspection photos of the current part.
    for i, (key, title) in enumerate((("top", "TOP CAMERA"), ("under", "UNDERSIDE CAMERA"))):
        y = i * 360
        d.rectangle((1280, y, W, y + 360), fill=PANEL)
        if state.get(key):
            img = Image.open(state[key]).convert("RGB").resize((640, 480)).crop((0, 60, 640, 420))
            canvas.paste(img, (1280, y))
        d.rectangle((1280, y, 1280 + 20 + d.textlength(title, font=font(20, True)) + 20, y + 36), fill=(0, 0, 0))
        d.text((1300, y + 6), title, font=font(20, True), fill=INK)

    # Bottom band: verdicts and reasons, then the tally.
    d.rectangle((0, 720, W, H), fill=BG)
    x0 = 40
    if state.get("injected"):
        d.text((x0, 735), f"OPERATOR INJECTED A FAULT: {state['injected']}", font=font(24, True), fill=WARN)
    dec = state.get("decision")
    if dec:
        y = 780
        for view, key in (("Top", "verdict_top"), ("Underside", "verdict_underside")):
            v = dec[key]
            col = GOOD if v["failure_type"] == "none" else BAD
            d.text((x0, y), f"{view}:", font=font(26, True), fill=MUTED)
            d.text((x0 + 190, y), LABEL.get(v["failure_type"], v["failure_type"]), font=font(26, True), fill=col)
            for j, line in enumerate(wrap(d, evidence(v), font(22), 1000)[:2]):
                d.text((x0 + 400, y + 2 + j * 28), line, font=font(22), fill=INK)
            y += 70
        act = "ACCEPT → good-parts rack" if dec["action"] == "accept" else "REJECT → scrap bin"
        d.text((x0, y + 4), act, font=font(32, True), fill=GOOD if dec["action"] == "accept" else BAD)
        ok = "correct" if dec.get("decision_correct") else "WRONG"
        d.text((x0 + 560, y + 12), f"truth: {truth(dec)} · {ok}", font=font(22), fill=MUTED)
    elif state.get("top"):
        d.text((x0, 790), "Inspecting…", font=font(28, True), fill=MUTED)

    t = state["tally"]
    tx = 1460
    d.text((tx, 745), "BAD PARTS SHIPPED", font=font(20, True), fill=MUTED)
    d.text((tx, 775), f"ours: {t['ours']} / {t['bad']}", font=font(30, True), fill=GOOD)
    d.text((tx, 815), f"today's factory: {t['bad']} / {t['bad']}", font=font(24), fill=BAD)
    d.text((tx, 870), f"parts inspected: {t['parts']}", font=font(22), fill=INK)
    d.text((tx, 985), "Isaac Sim · NVIDIA Cosmos-Reason1-7B", font=font(16), fill=MUTED)
    d.text((tx, 1007), "fine-tuned, 4-bit on one T4 GPU", font=font(16), fill=MUTED)
    d.text((tx, 1029), "sim paused while the model thinks", font=font(16), fill=MUTED)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path, help="the --film folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--hold", type=float, default=2.5, help="seconds to hold each decision on screen")
    args = ap.parse_args()

    marks = [json.loads(l) for l in (args.frames / "marks.jsonl").read_text().splitlines() if l.strip()]
    by_frame: dict[int, list] = {}
    for m in marks:
        by_frame.setdefault(m["frame"], []).append(m)
    frames = sorted(args.frames.glob("[0-9]*.jpg"))
    state = {"tally": {"ours": 0, "bad": 0, "parts": 0}}
    hold = int(args.hold * args.fps)
    with tempfile.TemporaryDirectory() as tmp:
        n = 0

        def emit(img):
            nonlocal n
            img.save(Path(tmp) / f"{n:06d}.jpg", quality=90)
            n += 1

        for i, f in enumerate(frames):
            repeat = 1
            for m in by_frame.get(i, []):
                if m["event"] == "injected":
                    state["injected"] = m["kind"]
                elif m["event"] == "photo":
                    paths = m["paths"]
                    if m["stem"].endswith("_top"):  # a new part: clear the last decision
                        state.update(top=next(p for p in paths if p.endswith("closeup.png")), under=None,
                                     decision=None)
                    else:
                        state["under"] = paths[0]
                elif m["event"] == "decision":
                    state["decision"] = m
                    t = state["tally"]
                    t["parts"] += 1
                    if m["kind"] != "good":
                        t["bad"] += 1
                        t["ours"] += m["action"] == "accept"
                    repeat = hold  # freeze on the verdict so it can be read
            img = compose(Image.open(f).convert("RGB"), state)
            for _ in range(repeat):
                emit(img)
            if state.get("decision") and state.get("injected") and state["decision"].get("injected"):
                state["injected"] = None  # shown until its part is decided
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps), "-i",
                        str(Path(tmp) / "%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        str(args.out)], check=True)
    print(f"[film] {len(frames)} frames + holds -> {n} frames, {n / args.fps:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
