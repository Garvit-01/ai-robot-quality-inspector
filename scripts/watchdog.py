#!/usr/bin/env python3
"""Watch a long background job and exit as soon as it needs attention. Uses no AI.

Checks every `--every` seconds and exits (with a one-line reason) when:
  DONE      the log shows the finish line (--done)
  ERROR     the log shows an error (--error, default: Traceback / Error executing / out of memory)
  STALLED   no new progress line (--progress) for --stall-min minutes, although the process is
            alive. This is what happened on Oct 14: Isaac Sim crashed inside main(), stayed
            open and idle, and a check that only waited for the process to end never fired.
  DIED      the process (--process, a pgrep -f pattern) is gone without the finish line
Run it in the background; whoever started it is notified when it exits, and it costs
nothing while the job is healthy.

Example (the sorting-cell training render):
    .venv/bin/python scripts/watchdog.py --log runs/sorting_train_data.log \\
        --process "run_sorting.py --trays train" --progress "tray \\d+ done" \\
        --done "tray 22199 done" --stall-min 10
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _ancestors() -> set[int]:
    """This watchdog and every process above it: their command lines contain the job's
    pattern too (it is an argument here), so they must not count as the job."""
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def alive(pattern: str) -> bool:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    mine = _ancestors()
    return any(int(x) not in mine for x in out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--process", required=True, help="pgrep -f pattern for the job")
    ap.add_argument("--progress", required=True, help="regex of a line that means 'still making progress'")
    ap.add_argument("--done", required=True, help="regex of the finish line")
    ap.add_argument("--error", default=r"Traceback|Error executing|out of memory|OutOfMemoryError")
    ap.add_argument("--stall-min", type=float, default=10.0)
    ap.add_argument("--every", type=float, default=30.0)
    args = ap.parse_args()

    prog, done, err = re.compile(args.progress), re.compile(args.done), re.compile(args.error)
    last_count, last_change = -1, time.time()
    while True:
        text = args.log.read_text(errors="replace") if args.log.exists() else ""
        lines = text.splitlines()
        count = sum(1 for l in lines if prog.search(l))
        last_line = next((l for l in reversed(lines) if prog.search(l)), "")
        if done.search(text):
            print(f"DONE: {count} progress lines; last: {last_line[:160]}")
            return 0
        m = err.search(text)
        if m:
            i = text.find(m.group(0))
            print(f"ERROR after {count} progress lines:\n{text[i:i + 1200]}")
            return 2
        if count != last_count:
            last_count, last_change = count, time.time()
        if not alive(args.process):
            print(f"DIED without finishing after {count} progress lines; last: {last_line[:160]}")
            return 3
        idle = (time.time() - last_change) / 60
        if idle > args.stall_min:
            print(f"STALLED: no progress for {idle:.0f} min (process still alive) after {count} lines; "
                  f"last: {last_line[:160]}")
            return 4
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
