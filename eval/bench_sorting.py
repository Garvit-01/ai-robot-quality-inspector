#!/usr/bin/env python3
"""Score sorting-line runs: how many bad parts ship, how many good parts are scrapped.

Reads the per-part log written by isaac/run_sorting.py (parts.jsonl) and compares, on the
very same parts and the very same verdicts:
  B         today's factory (gripper sensor only: accepts everything it can pick)
  top_only  our inspector with the top camera only
  ours      our inspector with the top and the upward-looking camera (what the robot did)

It also reports what happened physically: did each part end up where the robot meant to put
it (rack or bin), or was it dropped somewhere else in transit?

Usage:
    .venv/bin/python eval/bench_sorting.py runs/sorting/dev --out runs/sorting/dev/summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from selfheal.sorting import outcome, policy_actions  # noqa: E402

POLICIES = ("B", "top_only", "ours")
NAMES = {"B": "B · today's factory (gripper sensor only)", "top_only": "ours, top camera only",
         "ours": "ours, top + underside cameras"}


def load(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "parts.jsonl").read_text().splitlines() if l.strip()]


def summarise(rows: list[dict]) -> dict:
    per = {p: Counter() for p in POLICIES}
    by_kind = {p: defaultdict(Counter) for p in POLICIES}
    for r in rows:
        good = r["kind"] == "good"
        kind = "good" if good else (f"wrong:{r['part']}" if r["kind"] == "wrong" else f"{r['defect']}:{r['side']}")
        acts = policy_actions(r["verdict_top"]["failure_type"], r["verdict_underside"]["failure_type"])
        for p in POLICIES:
            o = outcome(acts[p], good)
            per[p][o] += 1
            by_kind[p][kind][o] += 1
    handled = Counter(r.get("ended_in", "?") for r in rows)
    handling_ok = sum(bool(r.get("handled_ok")) for r in rows)
    return {"parts": len(rows), "trays": len({r["tray"] for r in rows}),
            "bad_parts": sum(r["kind"] != "good" for r in rows),
            "policies": {p: dict(per[p]) for p in POLICIES},
            "by_kind": {p: {k: dict(v) for k, v in by_kind[p].items()} for p in POLICIES},
            "physical": {"handled_ok": handling_ok, "ended_in": dict(handled)},
            "seconds_per_part": round(sum(r.get("seconds", 0) for r in rows) / max(1, len(rows)), 1)}


def render(s: dict, run: Path) -> str:
    out = [f"# Sorting line — `{run}`", "",
           f"{s['trays']} trays, {s['parts']} parts ({s['bad_parts']} bad). Same parts and verdicts for every row.", "",
           "| Policy | Correct | Bad parts shipped | Good parts scrapped |", "|---|---|---|---|"]
    for p in POLICIES:
        c = s["policies"][p]
        out.append(f"| {NAMES[p]} | {c.get('correct', 0)}/{s['parts']} | {c.get('shipped_bad', 0)}/{s['bad_parts']} "
                   f"| {c.get('scrapped_good', 0)}/{s['parts'] - s['bad_parts']} |")
    out += ["", "**Bad parts shipped, by kind** (where the extra camera matters):", "",
            "| Kind | " + " | ".join(NAMES[p] for p in POLICIES) + " |", "|---|" + "---|" * len(POLICIES)]
    kinds = sorted({k for p in POLICIES for k in s["by_kind"][p]})
    for k in kinds:
        cells = []
        for p in POLICIES:
            c = s["by_kind"][p].get(k, {})
            n = sum(c.values())
            cells.append(f"{c.get('shipped_bad', 0)}/{n} shipped" if k != "good" else f"{c.get('scrapped_good', 0)}/{n} scrapped")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    ph = s["physical"]
    out += ["", f"**Physical handling** (the robot, on `ours`): {ph['handled_ok']}/{s['parts']} parts ended where "
                f"it meant to put them; final places: {ph['ended_in']}. {s['seconds_per_part']} s per part (simulated "
                "motion + verifier).", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+", help="run_sorting.py output folders")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [r for run in args.runs for r in load(run)]
    s = summarise(rows)
    md = render(s, args.runs[0] if len(args.runs) == 1 else Path("+".join(str(r) for r in args.runs)))
    print(md)
    if args.out:
        args.out.write_text(md)
        args.out.with_suffix(".json").write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
