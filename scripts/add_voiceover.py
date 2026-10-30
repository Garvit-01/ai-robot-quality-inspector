#!/usr/bin/env python3
"""Rebuild the submission video around a recorded voice-over.

Card durations follow the narration, and each sentence about the robot footage is placed just
before that part's decision appears. PIECES are (start, end) in the recording, found by
transcribing it with word timestamps, and where each goes: a segment of the video plus an
offset in seconds. The recording itself is not in the repository.

    .venv/bin/python scripts/add_voiceover.py --voice recording.m4a --out final.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_submission_video as M  # noqa: E402

SPEED = 1.45  # the sorting footage, so the whole video stays under 3 minutes
CARD_SECS = {"title": 4.0, "problem": 18.8, "how": 18.6, "live1": 4.6, "thread_intro": 6.0,
             "results": 12.0, "honest": 8.5, "end": 9.2}
PIECES = [
    (0.0, 3.4, "title", 0.3),          # introduction
    (3.55, 21.8, "problem", 0.3),
    (21.9, 40.0, "how", 0.3),
    (40.1, 44.3, "live1", 0.3),
    (44.4, 55.95, "footage", 0.3),     # injected crack; top camera; lifted over the underside camera
    (56.0, 64.8, "footage", 12.0),     # crack decision (shown at ~12.4 s into the footage)
    (64.9, 74.05, "footage", 20.9),    # M12 (decision ~25.4 s)
    (74.1, 79.2, "footage", 33.6),     # M20 (decision ~38.2 s)
    (79.25, 85.25, "footage", 46.0),   # good M16 (decision ~51.1 s)
    (86.8, 90.85, "footage", 59.5),    # bolt (decision ~63.4 s)
    (90.85, 100.9, "footage", 74.4),   # last good nut (decision ~76.2 s); "every decision comes with a reason"
    (100.95, 117.7, "thread_intro", 4.8),  # threading; starts later because the line above runs into the card
    (117.9, 129.3, "results", 0.3),
    (129.3, 137.1, "honest", 0.2),
    (138.2, 146.7, "end", 0.3),
]


def duration(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True).stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    parts: list[tuple[str, float]] = []
    still, clip = M.still, M.clip

    def timed_still(png, secs, out):
        secs = CARD_SECS.get(Path(png).stem, secs)
        still(png, secs, out)
        parts.append((Path(png).stem, secs))

    def timed_clip(src, out, speed=1.0, start=0, dur=None, banner=""):
        name = "footage" if "film_demo" in str(src) else "thread"
        clip(src, out, speed=SPEED if name == "footage" else speed, start=start, dur=dur, banner=banner)
        parts.append((name, duration(out)))

    M.still, M.clip = timed_still, timed_clip
    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "silent.mp4"
        sys.argv = [sys.argv[0], "--out", str(silent)]
        M.main()
        starts, clock = {}, 0.0
        for name, secs in parts:
            starts[name], clock = clock, clock + secs
        filters, labels = [], []
        for i, (a, b, seg, off) in enumerate(PIECES):
            at = int((starts[seg] + off) * 1000)
            filters.append(f"[1:a]atrim={a}:{b},asetpts=PTS-STARTPTS,adelay={at}|{at}[p{i}]")
            labels.append(f"[p{i}]")
        filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0,aformat=channel_layouts=mono,"
                       "highpass=f=70,loudnorm=I=-16:TP=-2:LRA=11,alimiter=limit=0.79:level=false,"
                       f"apad,atrim=0:{clock:.2f}[a]")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(args.voice),
                        "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[a]",
                        "-c:v", "libx264", "-profile:v", "main", "-level", "4.0", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(args.out)],
                       check=True)
    print(f"[voice] {args.out} · {duration(args.out):.1f} s · segments {({k: round(v, 1) for k, v in starts.items()})}")


if __name__ == "__main__":
    main()
