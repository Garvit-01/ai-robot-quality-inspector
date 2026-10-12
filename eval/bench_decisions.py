#!/usr/bin/env python3
"""Compare the four recovery policies (A/B/C/D) on rendered scenes. M3's experiment.

Takes a scene set and a verifier benchmark run over it (eval/bench_verifier.py) and,
for every scene, decides what each policy does and what that leads to:

  A  no checks                          always continue
  B  industrial baseline                gripper sensor only, retry then stop
  C  ours                               verifier -> sensor fusion -> Nemotron -> guardrail
  D  ablation                           verifier -> sensor fusion -> lookup table

Reusing the verifier run means C and D see exactly the verdicts that were scored, and
only the planner (C) makes new API calls — and only on scenes where the fused verdict
reports a problem, as in the real pipeline.

This is a per-checkpoint decision benchmark: one decision per scene. It does not
simulate whole cycles or recovery success; those need the closed loop (see README
limitations). It answers: given what each policy can sense, does it choose the right
response?

Usage:
    .venv/bin/python eval/bench_decisions.py runs/scenes/eval_v2 runs/bench/eval_v2__prompt_v3__gemma
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from _common import client_for, load_config, load_dotenv, require_env  # noqa: E402
from selfheal.fusion import fuse, gripper_signal  # noqa: E402
from selfheal.policies import (  # noqa: E402
    OUTCOMES, ideal_decision, outcome, policy_a, policy_b, policy_c, policy_d,
)

CONDITIONS = ("A", "B", "C", "D")
NAMES = {"A": "A · no checks", "B": "B · industrial baseline (gripper only)",
         "C": "C · ours (verifier + fusion + Nemotron)", "D": "D · verifier + fusion + lookup table"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("verifier_run", type=Path, help="a bench_verifier.py output dir over scene_dir")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--planner-timeout", type=float, default=60.0,
                    help="benchmark timeout per planner call. Production uses planner.timeout_s "
                         "from config (20 s); calls slower than that are counted separately, so "
                         "a slow API shows up as a latency risk instead of as bad judgement")
    ap.add_argument("--planner-model", default=None, help="override PLANNER_MODEL_ID")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None,
                    help="alternative config, e.g. config/selfheal_fastener.yaml for nut scenes")
    args = ap.parse_args()

    load_dotenv()
    if args.config:
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
    else:
        cfg = load_config()
    client = client_for({**cfg["planner"], "timeout_s": args.planner_timeout}).with_options(max_retries=1)
    model = args.planner_model or require_env(cfg["planner"]["model_id_env"])
    prod_budget_ms = 1000 * cfg["planner"]["timeout_s"]

    verdicts = {}
    for line in (args.verifier_run / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        verdicts[r["seed"]] = r
    labels = {int(p.stem): json.loads(p.read_text()) for p in (args.scene_dir / "labels").glob("*.json")}
    seeds = sorted(set(labels) & set(verdicts))
    if len(seeds) != len(labels):
        print(f"[decide] warning: {len(labels) - len(seeds)} scenes have no verifier result", flush=True)

    out = args.out or REPO / "runs" / "decisions" / f"{args.scene_dir.name}__{args.verifier_run.name}"
    out.mkdir(parents=True, exist_ok=True)

    def prepare(seed: int) -> dict:
        sc = labels[seed]["scene"]
        exp = sc["expected"]
        sig = gripper_signal(sc["gripper_closed"], sc["gripper_holding"])
        raw = verdicts[seed]["pred"]
        if raw is None:
            # The verifier gave no valid answer: the pipeline treats that as unclear.
            raw = {"checkpoint": sc["checkpoint"], "step_ok": False, "failure_type": "unclear",
                   "location_hint": "unknown", "evidence": "no valid verifier answer", "confidence": 0.0}
        fused, rule = fuse(raw, sig, sc["checkpoint"])
        return {"seed": seed, "fault": sc["fault"], "checkpoint": sc["checkpoint"], "signal": sig,
                "exp": exp, "raw": raw, "fused": fused, "fusion_rule": rule}

    rows = [prepare(s) for s in seeds]

    def decide_c(row: dict) -> dict:
        call = policy_c(client, model, row["fused"], row["checkpoint"],
                        temperature=cfg["planner"].get("temperature", 0.0),
                        sensor_confirmed_empty=(row["signal"] == "empty"),
                        expected_part=cfg["part_reference"]["correct"])
        return {"decision": call.decision, "raw_tool": call.raw_tool, "override": call.override,
                "latency_ms": round(call.latency_ms), "tokens_in": call.tokens_in,
                "tokens_out": call.tokens_out, "error": call.error}

    t0 = time.time()
    print(f"[decide] {len(rows)} scenes · planner {model} · verifier run {args.verifier_run.name}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        c_results = list(pool.map(decide_c, rows))

    for row, c in zip(rows, c_results):
        e = row["exp"]
        dec = {
            "A": policy_a(),
            "B": policy_b(checkpoint=row["checkpoint"], signal=row["signal"]),
            "C": tuple(c["decision"]),
            "D": policy_d(fused=row["fused"]),
        }
        row["ideal"] = ideal_decision(e["failure_type"], e["location_hint"])
        row["decisions"] = {k: list(v) for k, v in dec.items()}
        row["outcomes"] = {k: outcome(tuple(v), e["failure_type"], e["location_hint"]) for k, v in dec.items()}
        row["planner"] = c

    with (out / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # --- summarise ---
    summary = {"meta": {"scene_dir": str(args.scene_dir), "verifier_run": str(args.verifier_run),
                        "planner_model": model, "n": len(rows), "elapsed_s": round(time.time() - t0, 1),
                        "date": time.strftime("%Y-%m-%d %H:%M")}}
    per = {}
    for k in CONDITIONS:
        counts = Counter(r["outcomes"][k] for r in rows)
        by_fault = defaultdict(Counter)
        for r in rows:
            by_fault[r["fault"]][r["outcomes"][k]] += 1
        per[k] = {"outcomes": {o: counts.get(o, 0) for o in OUTCOMES},
                  "correct_pct": round(100 * counts["correct"] / len(rows), 1),
                  "by_fault": {f: dict(c) for f, c in sorted(by_fault.items())}}
    summary["conditions"] = per
    calls = [r["planner"] for r in rows if r["planner"]["raw_tool"] is not None or r["planner"]["error"]]
    summary["planner"] = {
        "calls": len(calls),
        "guardrail_overrides": sum(1 for c in calls if c["override"]),
        "override_notes": Counter(c["override"] for c in calls if c["override"]).most_common(),
        "errors": sum(1 for c in calls if c["error"]),
        "latency_ms_p50": round(statistics.median([c["latency_ms"] for c in calls])) if calls else None,
        "production_budget_ms": prod_budget_ms,
        "slower_than_production_budget": sum(1 for c in calls if c["latency_ms"] > prod_budget_ms),
        "tokens": {"in": sum(c["tokens_in"] for c in calls), "out": sum(c["tokens_out"] for c in calls)},
    }
    summary["fusion_rules_fired"] = dict(Counter(r["fusion_rule"] for r in rows if r["fusion_rule"]))
    # C vs D: do they ever disagree, and who is right when they do?
    dis = [r for r in rows if r["decisions"]["C"] != r["decisions"]["D"]]
    summary["c_vs_d"] = {"disagreements": len(dis),
                         "c_right": sum(1 for r in dis if r["outcomes"]["C"] == "correct"),
                         "d_right": sum(1 for r in dis if r["outcomes"]["D"] == "correct")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- markdown ---
    n = len(rows)
    L = [f"# Recovery policies A/B/C/D — {n} scenes", "",
         f"Scenes `{args.scene_dir}` · verifier run `{args.verifier_run.name}` · planner `{model}`", "",
         "One decision per scene, scored against what should have happened. Per-checkpoint "
         "decisions only: no whole-cycle or recovery-success simulation.", "",
         "| Condition | correct | quality escapes | operator calls | silent fails | false alarms | wrong recovery |",
         "|---|---|---|---|---|---|---|"]
    for k in CONDITIONS:
        o = per[k]["outcomes"]
        L.append(f"| {NAMES[k]} | **{per[k]['correct_pct']}%** | {o['quality_escape']} | {o['operator_call']} | "
                 f"{o['silent_fail']} | {o['false_alarm']} | {o['wrong_recovery']} |")
    p = summary["planner"]
    L += ["", f"Planner (C): {p['calls']} calls, {p['guardrail_overrides']} guardrail overrides, "
          f"{p['errors']} errors, p50 {p['latency_ms_p50']} ms, tokens in {p['tokens']['in']:,} / out {p['tokens']['out']:,}.",
          f"Calls slower than the production budget ({p['production_budget_ms'] / 1000:.0f} s), which in the "
          f"real pipeline would time out and become a safe stop: {p['slower_than_production_budget']} of {p['calls']}.",
          f"Override reasons: {p['override_notes'] or 'none'}.",
          f"Fusion rules fired: {summary['fusion_rules_fired'] or 'none'}.",
          f"C vs D: they disagree on {summary['c_vs_d']['disagreements']} scenes; C right on "
          f"{summary['c_vs_d']['c_right']}, D right on {summary['c_vs_d']['d_right']}.", "",
          "## Outcomes by fault", ""]
    for k in CONDITIONS:
        L.append(f"**{NAMES[k]}**  ")
        L.append("; ".join(f"{f}: " + ", ".join(f"{o} {c}" for o, c in sorted(cnt.items()))
                           for f, cnt in per[k]["by_fault"].items()))
        L.append("")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    for k in CONDITIONS:
        o = per[k]["outcomes"]
        print(f"[decide] {k}: correct {per[k]['correct_pct']}% · escapes {o['quality_escape']} · "
              f"operator calls {o['operator_call']} · silent fails {o['silent_fail']} · "
              f"false alarms {o['false_alarm']} · wrong recovery {o['wrong_recovery']}", flush=True)
    print(f"[decide] -> {out / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
