"""Fuse the verifier's verdict with the gripper's finger-width sensor.

The rule from SPEC §10: **the sensor wins on "is something in the hand."** A finger
gap is a direct physical measurement; a vision model looking at an angled photo of a
cube between two dark fingers is not. So where the gripper should be holding the part
(after lift, after transport), the sensor overrules the model in both directions. On
everything a finger gap cannot measure — part identity, damage, position, seating, and
where a dropped part went — the model's verdict stands untouched.

This is a systems decision, not a model one, and the benchmark reports how often each
rule fired so its effect is visible rather than folded into the headline.
"""
from __future__ import annotations

from typing import Literal

GripperSignal = Literal["open", "holding", "empty"]

# Checkpoints at which the gripper is supposed to be holding the part.
HOLDING_CHECKPOINTS = ("CP_POST_LIFT", "CP_POST_TRANSPORT")
DROPPED = ("part_dropped_recoverable", "part_dropped_unrecoverable")


def gripper_signal(closed: bool, holding: bool) -> GripperSignal:
    """What the finger-width sensor reads. In the real cell this is the Robotiq finger
    joint compared against empty_threshold / expected_width in config; here it is derived
    from the scene, because the sensor in a real cell measures it directly."""
    if not closed:
        return "open"
    return "holding" if holding else "empty"


def fuse(verdict: dict, signal: GripperSignal, checkpoint: str) -> tuple[dict, str | None]:
    """Return (fused verdict, name of the rule that changed it, or None)."""
    v = dict(verdict)
    if checkpoint not in HOLDING_CHECKPOINTS or signal == "open":
        return v, None

    if signal == "holding":
        # The fingers stopped at the part's width: something is in the hand. A model
        # that says empty or dropped is overruled. (It can't be overruled on WHAT is in
        # the hand — nothing here checks identity — so other verdicts stand.)
        if v["failure_type"] in ("empty_grasp",) + DROPPED:
            v.update(step_ok=True, failure_type="none", location_hint="gripper",
                     evidence=f"[sensor: fingers at part width] {v.get('evidence', '')}"[:500])
            return v, "sensor_holding_overrules_empty"
        return v, None

    # signal == "empty": the fingers closed fully, nothing is in the hand.
    if checkpoint == "CP_POST_LIFT":
        if v["failure_type"] == "empty_grasp":
            return v, None
        # The grasp missed: the part was never lifted, so it is still at the pick site.
        loc = v["location_hint"] if v["location_hint"] in ("carrier", "table_in_reach") else "carrier"
        v.update(step_ok=False, failure_type="empty_grasp", location_hint=loc,
                 evidence=f"[sensor: fingers fully closed] {v.get('evidence', '')}"[:500])
        return v, "sensor_empty_forces_empty_grasp"

    # CP_POST_TRANSPORT, empty: the part was lifted earlier, so it was dropped on the way.
    if v["failure_type"] in DROPPED:
        return v, None  # the model already says dropped, and where — keep it
    # The model missed the drop. The sensor knows the hand is empty but not where the part
    # went, and a regrasp needs a place to look: report it as unclear, which the planner
    # turns into a stop. Safe, never a guess.
    v.update(step_ok=False, failure_type="unclear", location_hint="unknown",
             evidence=f"[sensor: fingers fully closed, part location unknown] {v.get('evidence', '')}"[:500])
    return v, "sensor_empty_forces_unclear"
