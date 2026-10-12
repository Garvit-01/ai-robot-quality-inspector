#!/usr/bin/env python3
"""Project progress at a glance: `make status`.

Reads only what is on disk (git, runs/, the GPU), so it is always current and costs
nothing. Run it any time to see where the project stands.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
B, D, G, Y, R, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def latest(pattern: str) -> Path | None:
    runs = sorted(REPO.glob(pattern), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def section(title: str) -> None:
    print(f"\n{B}{title}{X}")


def main() -> None:
    print(f"{B}Self-Healing Work Cell — status{X}  {D}{sh('date +\"%Y-%m-%d %H:%M\"')}{X}")

    section("Code")
    branch = sh("git branch --show-current")
    print(f"  branch      {branch}   ({sh('git log --oneline | wc -l')} commits)")
    print(f"  last commit {sh('git log -1 --format=\"%h %s (%cr)\"')}")
    for b in ("main", "feature/industrial-parts"):
        up = sh(f"git rev-list --count origin/{b}..{b} 2>/dev/null") or sh(f"git rev-list --count {b} 2>/dev/null")
        tracked = sh(f"git rev-parse --verify -q origin/{b}")
        state = f"{up} commits not on GitHub" if tracked else f"not on GitHub yet ({up} commits)"
        colour = G if up == "0" else Y
        print(f"  {b:26} {colour}{state}{X}")
    dirty = [l for l in sh("git status --short").splitlines() if not l.endswith("test.ipynb")]
    print(f"  working tree {'clean' if not dirty else f'{Y}{len(dirty)} uncommitted change(s){X}'}")

    section("Milestones")
    has = lambda p: any(REPO.glob(p))  # noqa: E731
    rows = [
        ("M0  repo, config, live model calls", True),
        ("M1  fault-scene factory (Isaac Sim)", has("runs/scenes/eval_v2/labels/*.json")),
        ("M2  verifier scored on held-out scenes", has("runs/bench/eval_v2__*/summary.json")),
        ("M3  recovery policies A/B/C/D compared", has("runs/decisions/eval_v2__*/summary.json")),
        ("    real fasteners (NVIDIA nuts/bolts)", has("runs/bench/feval_v1__*/summary.json")),
        ("M4  Tier 2 fine-tune", has("runs/bench/*finetune*/summary.json")),
        ("M5  README with results, demo video", False),
    ]
    for name, done in rows:
        print(f"  {G + '✔' if done else D + '·'}{X} {name}")

    section("Latest results")
    for label, pat in (("cube verifier     ", "runs/bench/eval_v*__*/summary.json"),
                       ("fastener verifier ", "runs/bench/feval_v*__*/summary.json")):
        p = latest(pat)
        if not p:
            print(f"  {label} {D}none yet{X}")
            continue
        s = json.loads(p.read_text())
        o = s["overall"]
        gate = f"{G}GO{X}" if s["go_no_go"]["go"] else f"{R}NO-GO{X}"
        print(f"  {label} {o['failure_type_acc']}% correct · {s['false_alarm_rate_on_clean']}% false alarms · "
              f"{gate}  {D}{p.parent.name} ({o['n']} scenes){X}")
    for label, pat in (("cube policies     ", "runs/decisions/eval_v*/summary.json"),
                       ("fastener policies ", "runs/decisions/feval_v*/summary.json")):
        p = latest(pat)
        if not p:
            continue
        s = json.loads(p.read_text())["conditions"]
        print(f"  {label} bad parts shipped: baseline B {s['B']['outcomes']['quality_escape']} → "
              f"ours C {s['C']['outcomes']['quality_escape']}   (LLM C {s['C']['correct_pct']}% vs table D "
              f"{s['D']['correct_pct']}% correct)")
    for r in sorted(REPO.glob("reports/*.pdf")):
        print(f"  report            {r.relative_to(REPO)}")

    section("Machine")
    running = [l for l in sh("pgrep -fa 'stage_scene|bench_verifier|bench_decisions|make_report'").splitlines()
               if "pgrep" not in l]
    print(f"  running jobs {', '.join(l.split()[-1] for l in running) if running else 'none'}")
    gpu = sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader")
    print(f"  GPU          {gpu or 'not visible'}")
    free = shutil.disk_usage(REPO).free / 1e9
    print(f"  disk free    {(Y if free < 15 else '')}{free:.0f} GB{X}")
    print("  paid Nebius endpoints: none created by this project (per-token calls only)")
    print(f"\n{D}Next: see journal/README.md · run `make help` for what you can run{X}")


if __name__ == "__main__":
    main()
