"""Sensor fusion, the four policies, and outcome scoring. These decide every number in
the A/B/C/D comparison, so each rule is pinned here before any API call is made."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac"))

import faults  # noqa: E402
from selfheal.fusion import fuse, gripper_signal  # noqa: E402
from selfheal.policies import (  # noqa: E402
    CONTINUE, OUTCOMES, STOP, ideal_decision, outcome, policy_a, policy_b, policy_d,
)
from selfheal.verdict import FailureType  # noqa: E402


def V(ft, ok=None, loc="carrier", cp="CP_POST_LIFT"):
    return {"checkpoint": cp, "step_ok": (ft == "none") if ok is None else ok,
            "failure_type": ft, "location_hint": loc, "evidence": "e", "confidence": 0.9}


# --- fusion -----------------------------------------------------------------------------
def test_sensor_holding_overrules_a_model_that_says_empty():
    fused, rule = fuse(V("empty_grasp"), "holding", "CP_POST_LIFT")
    assert fused["failure_type"] == "none" and fused["step_ok"] and fused["location_hint"] == "gripper"
    assert rule == "sensor_holding_overrules_empty"


def test_sensor_holding_does_not_overrule_identity():
    # A finger gap can't tell a wrong part from a right one.
    fused, rule = fuse(V("wrong_part"), "holding", "CP_POST_LIFT")
    assert fused["failure_type"] == "wrong_part" and rule is None


def test_sensor_empty_after_lift_forces_empty_grasp():
    fused, rule = fuse(V("none", loc="gripper"), "empty", "CP_POST_LIFT")
    assert fused["failure_type"] == "empty_grasp" and not fused["step_ok"]
    assert fused["location_hint"] == "carrier"  # never lifted, so still at the pick site


def test_sensor_empty_after_transport_keeps_the_models_drop_location():
    fused, rule = fuse(V("part_dropped_recoverable", loc="table_in_reach"), "empty", "CP_POST_TRANSPORT")
    assert fused["failure_type"] == "part_dropped_recoverable" and rule is None


def test_sensor_empty_after_transport_without_a_drop_verdict_is_unclear_not_a_guess():
    fused, rule = fuse(V("none", loc="gripper"), "empty", "CP_POST_TRANSPORT")
    assert fused["failure_type"] == "unclear" and fused["location_hint"] == "unknown"


@pytest.mark.parametrize("cp", ["CP_PRE_GRASP", "CP_POST_PLACE"])
def test_no_fusion_where_the_hand_should_be_open(cp):
    v = V("misaligned_seat", loc="fixture", cp=cp)
    assert fuse(v, "open", cp) == (v, None)


def test_gripper_signal():
    assert gripper_signal(False, False) == "open"
    assert gripper_signal(True, True) == "holding"
    assert gripper_signal(True, False) == "empty"


# --- policies ---------------------------------------------------------------------------
def test_every_failure_type_has_an_ideal_decision():
    for ft in get_args(FailureType):
        tool, region = ideal_decision(ft, "carrier")
        assert tool in {"continue_cycle", "relocalize_and_regrasp", "reseat_part",
                        "reject_to_bin", "stop_and_alert"}


def test_baseline_b_only_reacts_to_an_empty_hand():
    assert policy_b(checkpoint="CP_POST_LIFT", signal="empty") == STOP
    assert policy_b(checkpoint="CP_POST_TRANSPORT", signal="empty") == STOP
    assert policy_b(checkpoint="CP_POST_LIFT", signal="holding") == CONTINUE
    # It has no camera: a wrong part, a crack or a crooked seat all pass.
    assert policy_b(checkpoint="CP_PRE_GRASP", signal="open") == CONTINUE
    assert policy_b(checkpoint="CP_POST_PLACE", signal="open") == CONTINUE


def test_table_policy_d_follows_the_fused_verdict():
    assert policy_d(fused=V("none")) == CONTINUE
    assert policy_d(fused=V("defective_part")) == ("reject_to_bin", None)
    assert policy_d(fused=V("part_dropped_recoverable", loc="table_in_reach")) == ("relocalize_and_regrasp", "table")
    assert policy_d(fused=V("unclear", loc="unknown")) == STOP


def test_policy_a_never_reacts():
    assert policy_a() == CONTINUE


# --- outcomes ---------------------------------------------------------------------------
@pytest.mark.parametrize("decision,true_ft,loc,expected", [
    (CONTINUE, "none", "carrier", "correct"),
    (STOP, "none", "carrier", "false_alarm"),
    (("reject_to_bin", None), "none", "carrier", "false_alarm"),
    (CONTINUE, "defective_part", "carrier", "quality_escape"),
    (("reseat_part", None), "wrong_part", "carrier", "quality_escape"),
    (STOP, "misaligned_seat", "fixture", "operator_call"),
    (("reject_to_bin", None), "misaligned_seat", "fixture", "wrong_recovery"),
    (CONTINUE, "empty_grasp", "carrier", "silent_fail"),
    (STOP, "part_dropped_recoverable", "table_in_reach", "operator_call"),
    (("relocalize_and_regrasp", "carrier"), "part_dropped_recoverable", "table_in_reach", "wrong_recovery"),
    (STOP, "part_dropped_unrecoverable", "off_table", "correct"),
])
def test_outcome(decision, true_ft, loc, expected):
    assert outcome(decision, true_ft, loc) == expected


def test_baseline_b_outcomes_on_the_eval_split_are_what_its_design_implies():
    """B can't see identity, damage or seating, so every F3/F4/F5 scene is an escape
    BY CONSTRUCTION — not by measurement. Pinning it keeps that claim honest."""
    for s in range(5000, 5120):
        sc = faults.make_scene(s)
        sig = gripper_signal(sc.gripper_closed, sc.gripper_holding)
        o = outcome(policy_b(checkpoint=sc.checkpoint, signal=sig),
                    sc.expected.failure_type, sc.expected.location_hint)
        assert o in OUTCOMES
        if sc.fault in ("F3", "F4", "F5"):
            assert o == "quality_escape", s
        if sc.fault == "F0":
            assert o == "correct", s
