#!/usr/bin/env python3
"""Rebuild the submission video around a recorded voice-over.

Card durations follow the narration, and each sentence about the robot footage is placed just
before that part's decision appears. PIECES are (start, end) in the recording, found by
transcribing it with word timestamps, and where each goes: a segment of the video plus an
offset in seconds. The recording itself is not in the repository.

    .venv/bin/python scripts/add_voiceover.py --voice final_take.m4a --voice-old earlier_take.m4a --out final.mp4
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
CARD_SECS = {"title": 3.0, "problem": 15.6, "how": 19.3, "live1": 6.8, "thread_intro": 5.6,
             "results": 12.0, "honest": 8.0, "end": 9.0}
# Final take ("new"), with two sections from an earlier take ("old"):
# the new take stumbles on "Here it is, running live" and on the results sentence. Pieces are
# (start, end, segment, offset, source). Footage events (s into the footage, SPEED 1.45): crack top 7.1,
# underside 10.7, decision 12.4; M12 decision 25.4; M20 38.2; good M16 51.1; bolt 63.4; last nut 76.2.
PIECES = [
    (1.0, 2.6, "title", 0.3, "new"),             # "Hi, I'm Garvit Katyal."
    (6.15, 20.95, "problem", 0.3, "new"),        # skips the spoken heading before "Most factory robots"
    (23.1, 41.8, "how", 0.3, "new"),             # skips "So how it works?"
    (40.1, 44.3, "live1", 0.3, "old"),           # "Here it is, running live. The AI decides every nut."
    (48.55, 55.6, "live1", 4.85, "new"),         # "An operator has just injected a fault..." (runs into the footage)
    (56.0, 58.36, "footage", 5.45, "new"),       # "...under the top camera" ~ top photo 7.1
    (58.40, 62.10, "footage", 8.65, "new"),       # "...over the second camera" ~ underside photo 10.7
    (63.50, 66.42, "footage", 12.55, "new"),      # "From the top it looks perfect..." (decision 12.4)
    (66.92, 71.50, "footage", 15.67, "new"),     # "...but from below it sees a crack... scrap bin"
    (72.10, 72.65, "footage", 20.6, "new"),      # "Next,"
    (73.22, 76.15, "footage", 21.35, "new"),     # "an M12 nut, too small for this job."
    (76.75, 81.95, "footage", 24.48, "new"),     # "...flags it as the wrong part" (decision 25.4)
    (82.95, 86.50, "footage", 36.24, "new"),     # M20 "wrong part again" (decision 38.2)
    (88.05, 93.20, "footage", 47.58, "new"),     # good M16 "both cameras say OK... green rack" (decision 51.1)
    (94.45, 97.50, "footage", 61.2, "new"),      # bolt "rejected" (decision 63.4)
    (97.80, 99.90, "footage", 75.12, "new"),     # "One more good nut: accepted." (decision 76.2)
    (100.45, 105.70, "footage", 77.45, "new"),   # "Every decision comes with a written reason..." (runs into the card)
    (106.40, 107.40, "thread_intro", 2.85, "new"),   # "Good nuts"
    (108.20, 125.65, "thread_intro", 4.1, "new"),    # "...47 out of 48..." inside the threading clip
    (117.9, 129.3, "results", 0.3, "old"),       # results sentence
    (139.5, 146.55, "honest", 0.3, "new"),
    (150.65, 155.95, "end", 0.3, "new"),         # clean retake of the closing line (hesitant take dropped)
    (157.85, 159.70, "end", 6.05, "new"),        # "The code is on GitHub. Thanks for watching." (a click before it dropped)
]
# Pieces for the earlier take alone, with its card durations, kept for reference:
CARD_SECS_OLD = {"title": 4.0, "problem": 18.8, "how": 18.6, "live1": 4.6, "thread_intro": 6.0,
                 "results": 12.0, "honest": 8.5, "end": 9.2}
PIECES_OLD = [
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
    ap.add_argument("--voice", type=Path, required=True, help="the final take (pieces marked 'new')")
    ap.add_argument("--voice-old", type=Path, help="the earlier take (pieces marked 'old')")
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
        inputs = {"new": (1, args.voice), "old": (2, args.voice_old)}
        used = {src: [i for i, pc in enumerate(PIECES) if pc[4] == src] for src in inputs}
        for src, (idx, path) in inputs.items():
            if not used[src]:
                continue
            assert path, f"pieces use the {src} recording: pass --voice-old"
            # each recording is loudness-normalised on its own first, so the two takes sound alike
            outs = "".join(f"[s{i}]" for i in used[src])
            filters.append(f"[{idx}:a]aformat=channel_layouts=mono,loudnorm=I=-16:TP=-2:LRA=11,aresample=48000,"
                           f"asplit={len(used[src])}{outs}")
        for i, (a, b, seg, off, _src) in enumerate(PIECES):
            at = int((starts[seg] + off) * 1000)
            filters.append(f"[s{i}]atrim={a}:{b},asetpts=PTS-STARTPTS,adelay={at}|{at}[p{i}]")
            labels.append(f"[p{i}]")
        filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0,aformat=channel_layouts=mono,"
                       "highpass=f=70,loudnorm=I=-16:TP=-2:LRA=11,alimiter=limit=0.79:level=false,"
                       f"apad,atrim=0:{clock:.2f}[a]")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(args.voice),
                        *(["-i", str(args.voice_old)] if args.voice_old else []),
                        "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[a]",
                        "-c:v", "libx264", "-profile:v", "main", "-level", "4.0", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(args.out)],
                       check=True)
    print(f"[voice] {args.out} · {duration(args.out):.1f} s · segments {({k: round(v, 1) for k, v in starts.items()})}")


if __name__ == "__main__":
    main()
