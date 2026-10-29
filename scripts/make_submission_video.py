#!/usr/bin/env python3
"""Assemble the submission video (< 3 min, captions only) from the filmed runs.

Title and explanation cards are drawn here; the footage comes from
`run_sorting.py --film` + `eval/film_overlay.py` (the sorting line) and
`isaac/factory_eval.py` (NVIDIA's threading policy).

    .venv/bin/python scripts/make_submission_video.py --out runs/video/self_healing_work_cell.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
W, H, FPS = 1920, 1080, 30
BG, INK, MUTED, GREEN, RED, AMBER, BLUE = (22, 26, 31), (235, 238, 242), (150, 158, 168), (88, 196, 120), (232, 92, 80), (242, 183, 5), (111, 168, 232)
FONT, BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def f(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def card(lines: list[tuple[str, int, bool, tuple]], path: Path, top: int = 300) -> None:
    """lines: (text, size, bold, colour); blank text adds a gap."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = top
    for text, size, bold, col in lines:
        if text:  # "left\tright": the right part starts at a fixed column, so numbers line up
            left, _, right = text.partition("\t")
            d.text((160, y), left, font=f(size, bold), fill=col)
            if right:
                d.text((1320, y), right, font=f(size, bold), fill=col)
        y += int(size * 1.45)
    im.save(path)


def run(cmd):
    subprocess.run(cmd, check=True)


def still(png: Path, secs: float, out: Path) -> None:
    run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-t", str(secs), "-i", str(png),
         "-vf", f"fps={FPS},format=yuv420p,fade=t=in:st=0:d=0.4", "-c:v", "libx264", "-crf", "18", str(out)])


def clip(src: Path, out: Path, speed: float = 1.0, start: float = 0, dur: float | None = None, banner: str = "") -> None:
    vf = [f"setpts=PTS/{speed}", f"scale={W}:{H}:force_original_aspect_ratio=decrease",
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x161a1f", f"fps={FPS}"]
    if banner:
        vf.append(f"drawbox=x=0:y=ih-120:w=iw:h=120:color=0x161a1f@0.85:t=fill")
        vf.append(f"drawtext=fontfile={BOLD}:text='{banner}':fontcolor=0xebeef2:fontsize=40:x=60:y=h-85")
    vf.append("format=yuv420p")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start)]
    if dur:
        cmd += ["-t", str(dur)]
    run(cmd + ["-i", str(src), "-an", "-vf", ",".join(vf), "-c:v", "libx264", "-crf", "18", str(out)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs/video/self_healing_work_cell.mp4")
    args = ap.parse_args()
    sort1 = REPO / "runs/sorting/film_demo/demo.mp4"
    sort2 = REPO / "runs/sorting/film_demo2/demo.mp4"
    thread = REPO / "runs/factory/nutthread/rl-video-step-0.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        parts = []

        def add_card(name, lines, secs, top=300):
            png, mp4 = t / f"{name}.png", t / f"{name}.mp4"
            card(lines, png, top)
            still(png, secs, mp4)
            parts.append(mp4)

        add_card("title", [
            ("Self-Healing Work Cell", 96, True, INK),
            ("", 30, False, INK),
            ("A robot that inspects every nut from above and below,", 44, False, INK),
            ("explains what is wrong, and only passes the good ones.", 44, False, INK),
            ("", 30, False, INK),
            ("Nebius x NVIDIA Global AI Hackathon 2026 · Physical AI track", 32, False, MUTED),
        ], 6, top=280)
        add_card("problem", [
            ("The problem", 64, True, AMBER),
            ("", 20, False, INK),
            ("A typical robot cell has one sensor that matters:", 46, False, INK),
            ("\"is a part in my gripper?\"", 46, True, INK),
            ("", 20, False, INK),
            ("A cracked, rusted or wrong-size nut is still \"a part\", so it ships.", 46, False, INK),
            ("Damage underneath is invisible to a camera that only looks down.", 46, False, INK),
        ], 9, top=260)
        add_card("how", [
            ("How it works", 64, True, BLUE),
            ("", 16, False, INK),
            ("1  NVIDIA Isaac Sim renders a cell with NVIDIA's own M16 nuts, faults known.", 38, False, INK),
            ("2  We fine-tuned NVIDIA Cosmos-Reason1-7B (LoRA) on ~6,000 of those photos", 38, False, INK),
            ("    on a Nebius AI Cloud H100 GPU.", 38, False, INK),
            ("3  It runs compressed (4-bit) on one small GPU beside the simulator:", 38, False, INK),
            ("    a UR10 arm photographs each nut from the top and from below,", 38, False, INK),
            ("    the model answers good / wrong part / damaged, and why.", 38, False, INK),
            ("4  NVIDIA Nemotron on Nebius Token Factory plans recoveries;", 38, False, INK),
            ("    NVIDIA's pretrained Isaac Lab policy threads the good nuts.", 38, False, INK),
        ], 14, top=170)
        add_card("live1", [
            ("Live run: the model decides every nut", 60, True, GREEN),
            ("", 20, False, INK),
            ("An operator injects a fault mid-run: a crack on the underside.", 44, False, INK),
            ("Watch the top camera say OK, and the underside camera catch it.", 44, False, INK),
        ], 5, top=380)
        clip_a = t / "sort1.mp4"
        clip(sort1, clip_a, speed=1.35)  # 116 s -> ~86 s of the line operating
        parts.append(clip_a)
        add_card("thread_intro", [
            ("Good nuts go on to threading", 60, True, GREEN),
            ("", 20, False, INK),
            ("NVIDIA's pretrained Isaac Lab policy screws the nut onto an M16 bolt.", 44, False, INK),
            ("It is blind: it threads whatever it's handed.", 44, False, INK),
            ("That's why the inspector decides what reaches it.", 44, False, INK),
        ], 6, top=330)
        clip_b = t / "thread.mp4"
        clip(thread, clip_b, speed=1.0, start=0, dur=16, banner="NVIDIA NutThread policy - 47 of 48 nuts threaded (97.9 percent)")
        parts.append(clip_b)
        add_card("results", [
            ("Results: 40 trays the model never saw (240 nuts, 120 bad)", 52, True, INK),
            ("", 18, False, INK),
            ("Bad nuts shipped", 40, True, MUTED),
            ("Today's cell (gripper sensor only)\t120 / 120", 44, False, RED),
            ("Ours, top camera only\t61 / 120", 44, False, AMBER),
            ("Ours, top + underside camera\t24 / 120", 44, True, GREEN),
            ("", 18, False, INK),
            ("240 / 240 nuts physically placed where the robot meant to.", 40, False, INK),
            ("Our first fine-tune: base Cosmos-Reason1 21.7% → 94.2% on 120 held-out scenes.", 40, False, INK),
        ], 13, top=200)
        add_card("honest", [
            ("What it doesn't do yet", 60, True, AMBER),
            ("", 18, False, INK),
            ("All of this is simulation; no real hardware yet.", 42, False, INK),
            ("Small 3-5 mm chips still slip through (22 of 26).", 42, False, INK),
            ("A plain ResNet-18 trained on the same photos is about as accurate;", 42, False, INK),
            ("the NVIDIA model's edge is that it explains every decision in words.", 42, False, INK),
        ], 10, top=280)
        add_card("end", [
            ("Self-Healing Work Cell", 80, True, INK),
            ("", 20, False, INK),
            ("NVIDIA: Isaac Sim · Isaac Lab · Cosmos-Reason1-7B · Nemotron", 40, False, GREEN),
            ("Nebius: AI Cloud (H100 fine-tuning) · Token Factory (Nemotron)", 40, False, BLUE),
            ("", 20, False, INK),
            ("Code, results and how to run it:", 38, False, MUTED),
            ("github.com/Garvit-01/nvidia_nebius_challenge_ws", 40, True, INK),
        ], 8, top=290)

        listing = t / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)])
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(args.out)],
                         capture_output=True, text=True).stdout.strip()
    print(f"[video] {args.out} · {float(dur):.0f} s")


if __name__ == "__main__":
    main()
