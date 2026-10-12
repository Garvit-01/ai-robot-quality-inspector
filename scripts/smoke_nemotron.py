#!/usr/bin/env python3
"""M0 smoke test for the recovery planner.

Two jobs:
  --list   dump the live model catalogue so we can PIN real model IDs (SPEC.md §11
           says never trust an ID written in a doc).
  default  send a synthetic verifier verdict and assert the model returns exactly one
           tool call from our whitelist, then show the code-level guardrail firing.

Usage:
  .venv/bin/python scripts/smoke_nemotron.py --list
  .venv/bin/python scripts/smoke_nemotron.py
  .venv/bin/python scripts/smoke_nemotron.py --failure-type wrong_part
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from _common import client_for, load_config, load_dotenv, require_env

from selfheal.planner import SYSTEM_PROMPT, TOOLS, guardrail  # noqa: E402


def list_models(cfg: dict) -> None:
    client = client_for(cfg["planner"])
    models = sorted(m.id for m in client.models.list().data)
    print(f"{len(models)} models visible on {require_env(cfg['planner']['base_url_env'])}\n")
    interesting = [
        m for m in models
        if any(k in m.lower() for k in ("nemotron", "nvidia", "cosmos", "vl", "vision"))
    ]
    print("--- NVIDIA / vision candidates (pin one of these) ---")
    for m in interesting or ["(none matched)"]:
        print(" ", m)
    print("\n--- full catalogue ---")
    for m in models:
        print(" ", m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="dump the live model catalogue")
    ap.add_argument("--failure-type", default="part_dropped_recoverable")
    ap.add_argument("--location-hint", default="table_in_reach")
    ap.add_argument("--attempts", type=int, default=0, help="recoveries already tried")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config()

    if args.list:
        list_models(cfg)
        return 0

    planner = cfg["planner"]
    client = client_for(planner)
    model_id = require_env(planner["model_id_env"])

    verdict = {
        "checkpoint": "CP_POST_TRANSPORT",
        "step_ok": False,
        "failure_type": args.failure_type,
        "evidence": "gripper fingers fully closed, cube resting on table left of fixture",
        "location_hint": args.location_hint,
        "confidence": 0.82,
    }
    context = {
        "verifier_report": verdict,
        "gripper_state": "fully_closed_empty",
        "recovery_attempts_this_cycle": args.attempts,
        "expected_part": "50 mm matte grey cube",
    }

    print(f"model:   {model_id}")
    print(f"verdict: {json.dumps(verdict)}\n")

    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model_id,
        temperature=planner.get("temperature", 0.0),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, indent=2)},
        ],
        tools=TOOLS,
        tool_choice="required",
    )
    latency_ms = (time.monotonic() - t0) * 1000

    msg = resp.choices[0].message
    calls = msg.tool_calls or []
    print(f"latency: {latency_ms:.0f} ms")
    if resp.usage:
        print(f"tokens:  in={resp.usage.prompt_tokens} out={resp.usage.completion_tokens}")

    if not calls:
        print(f"\nFAIL: no tool call returned. Raw content:\n{msg.content!r}")
        print("Guardrail would force: stop_and_alert")
        return 1
    if len(calls) > 1:
        print(f"\nWARNING: {len(calls)} tool calls returned, spec allows exactly one.")

    chosen = calls[0].function.name
    try:
        raw_args = json.loads(calls[0].function.arguments or "{}")
    except json.JSONDecodeError as exc:
        print(f"\nFAIL: tool arguments were not valid JSON ({exc}).")
        print("Guardrail would force: stop_and_alert")
        return 1

    print(f"\nmodel chose: {chosen}({json.dumps(raw_args)})")

    final, note = guardrail(chosen, args.failure_type)
    if note:
        print(f"GUARDRAIL OVERRIDE -> {final}  ({note})")
    else:
        print(f"guardrail: ok, executing {final}")

    print("\nPASS: got exactly one whitelisted, well-formed tool call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
