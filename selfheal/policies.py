"""The four recovery policies compared in the experiment (SPEC §8), and how a decision is
scored against what should have happened.

  A  no checks at all: always continue
  B  industrial baseline: gripper sensor only; on an empty hand, retry the pick at the
     ORIGINAL location, then stop and call an operator
  C  ours: verifier + sensor fusion, then Nemotron picks one whitelisted tool, guarded
  D  ablation: verifier + sensor fusion, then a fixed lookup table (no LLM)

D exists to answer the obvious question: if the LLM only picks from five tools, could
a dictionary do it? If D ties C, that is a finding, and it gets reported.

A decision is (tool, region): region is only meaningful for relocalize_and_regrasp.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from selfheal.planner import SYSTEM_PROMPT, TOOLS, guardrail

Decision = tuple[str, str | None]

CONTINUE: Decision = ("continue_cycle", None)
STOP: Decision = ("stop_and_alert", None)

# What SHOULD happen for each true failure type. `region` follows where the part is.
QUALITY_FAULTS = ("misaligned_seat", "wrong_part", "defective_part")


def ideal_decision(failure_type: str, location_hint: str) -> Decision:
    if failure_type == "none":
        return CONTINUE
    if failure_type in ("part_displaced", "empty_grasp"):
        return ("relocalize_and_regrasp", "carrier")
    if failure_type == "part_dropped_recoverable":
        return ("relocalize_and_regrasp", "table" if location_hint != "carrier" else "carrier")
    if failure_type == "misaligned_seat":
        return ("reseat_part", None)
    if failure_type in ("wrong_part", "defective_part"):
        return ("reject_to_bin", None)
    return STOP  # part_dropped_unrecoverable, unclear


# --- A ------------------------------------------------------------------------------
def policy_a(**_) -> Decision:
    return CONTINUE


# --- B ------------------------------------------------------------------------------
def policy_b(checkpoint: str, signal: str, **_) -> Decision:
    """The realistic industrial standard. It only has the finger sensor, and only checks
    where the hand should hold something. On an empty hand it retries the pick at the
    ORIGINAL pick location — where, after a displacement or a drop, the part isn't — so
    both retries close on air and it stops. Its decision is therefore stop_and_alert.
    Everything a finger gap can't measure (identity, damage, seating) it passes."""
    if checkpoint in ("CP_POST_LIFT", "CP_POST_TRANSPORT") and signal == "empty":
        return STOP
    return CONTINUE


# --- D ------------------------------------------------------------------------------
def policy_d(fused: dict, **_) -> Decision:
    """The no-LLM ablation: the same fused verdict, mapped by a fixed table."""
    if fused["step_ok"]:
        return CONTINUE
    return ideal_decision(fused["failure_type"], fused["location_hint"])


# --- C ------------------------------------------------------------------------------
@dataclass
class PlannerCall:
    decision: Decision
    raw_tool: str | None
    override: str | None
    latency_ms: float
    tokens_in: int
    tokens_out: int
    error: str | None


def policy_c(client, model: str, fused: dict, checkpoint: str, attempts: int = 0,
             temperature: float = 0.0, sensor_confirmed_empty: bool = False) -> PlannerCall:
    """Nemotron picks one tool from the whitelist; the guardrail runs after it, in code.
    Called only when the fused verdict reports a problem — a clean step never reaches
    the planner. Any failure (transport, no tool call, bad arguments) is a stop."""
    if fused["step_ok"]:
        return PlannerCall(CONTINUE, None, None, 0.0, 0, 0, None)
    context = {
        "verifier_report": {k: fused[k] for k in ("checkpoint", "step_ok", "failure_type",
                                                   "location_hint", "evidence")},
        "recovery_attempts_this_cycle": attempts,
        "expected_part": "50 mm matte grey cube",
    }
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=temperature, tool_choice="required", tools=TOOLS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": json.dumps(context)}])
    except Exception as exc:  # noqa: BLE001
        return PlannerCall(STOP, None, "transport error -> stop", (time.monotonic() - t0) * 1000,
                           0, 0, f"{type(exc).__name__}: {exc}"[:200])
    ms = (time.monotonic() - t0) * 1000
    tin = resp.usage.prompt_tokens if resp.usage else 0
    tout = resp.usage.completion_tokens if resp.usage else 0
    calls = resp.choices[0].message.tool_calls or []
    if not calls:
        return PlannerCall(STOP, None, "no tool call -> stop", ms, tin, tout, None)
    name = calls[0].function.name
    try:
        args = json.loads(calls[0].function.arguments or "{}")
    except json.JSONDecodeError:
        return PlannerCall(STOP, name, "unparseable arguments -> stop", ms, tin, tout, None)
    final, note = guardrail(name, fused["failure_type"], sensor_confirmed_empty)
    region = args.get("region") if final == "relocalize_and_regrasp" else None
    if final == "relocalize_and_regrasp" and region not in ("carrier", "table"):
        final, region, note = "stop_and_alert", None, f"invalid region {region!r} -> stop"
    return PlannerCall((final, region), name, note, ms, tin, tout, None)


# --- scoring ----------------------------------------------------------------------------
OUTCOMES = ("correct", "quality_escape", "operator_call", "silent_fail",
            "false_alarm", "wrong_recovery")


def outcome(decision: Decision, true_failure: str, true_location: str) -> str:
    """What the decision leads to, judged against the truth.

    correct         the right recovery (or rightly continuing)
    quality_escape  a bad part goes on into the fixture: wrong, defective or misaligned
                    and the cell continues, or "fixes" it in a way that keeps the part
    operator_call   the cell stops when it could have recovered by itself
    silent_fail     a fault goes unhandled and nothing notices (e.g. continuing with an
                    empty gripper); no bad part ships, but the cycle silently produces nothing
    false_alarm     nothing was wrong, and the cell acted anyway (stop or recovery)
    wrong_recovery  it acted, but with the wrong recovery (e.g. regrasp in the wrong
                    place, or scrapping a good part)
    """
    ideal = ideal_decision(true_failure, true_location)
    if decision == ideal:
        return "correct"
    tool = decision[0]
    if true_failure == "none":
        return "false_alarm"
    if true_failure in QUALITY_FAULTS:
        if tool in ("continue_cycle", "relocalize_and_regrasp"):
            return "quality_escape"
        if tool == "reseat_part" and true_failure in ("wrong_part", "defective_part"):
            return "quality_escape"  # re-placed into the fixture: still a bad part
        if tool == "stop_and_alert":
            return "operator_call"
        return "wrong_recovery"  # e.g. a misaligned but good part sent to the bin
    if tool == "continue_cycle":
        return "silent_fail"
    if tool == "stop_and_alert":
        return "operator_call"
    return "wrong_recovery"
