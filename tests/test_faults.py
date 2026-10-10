"""Tests for the fault catalogue. Run with: .venv/bin/python -m pytest -q"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "scripts"))

import faults  # noqa: E402
from _common import FailureType, LocationHint  # noqa: E402
from typing import get_args  # noqa: E402

SEEDS = range(1000, 1600)


def test_same_seed_same_scene():
    for seed in (0, 7, 1037, 99999):
        assert faults.make_scene(seed) == faults.make_scene(seed)


def test_fault_classes_are_balanced():
    counts = Counter(faults.make_scene(s).fault for s in SEEDS)
    assert set(counts) == set(faults.FAULT_CLASSES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_clean_scenes_cover_every_checkpoint():
    cps = {faults.make_scene(s).checkpoint for s in SEEDS if faults.make_scene(s).fault == "F0"}
    assert cps == set(faults.CHECKPOINTS)


def test_labels_use_only_schema_enums():
    allowed_failures = set(get_args(FailureType))
    allowed_locations = set(get_args(LocationHint))
    for s in SEEDS:
        e = faults.make_scene(s).expected
        assert e.failure_type in allowed_failures
        assert e.location_hint in allowed_locations
        assert e.checkpoint in faults.CHECKPOINTS


def test_step_ok_iff_no_failure():
    for s in SEEDS:
        e = faults.make_scene(s).expected
        assert e.step_ok == (e.failure_type == "none"), s


def test_clean_scenes_report_nominal_location():
    for s in SEEDS:
        sc = faults.make_scene(s)
        if sc.fault == "F0":
            assert sc.expected.location_hint == faults.NOMINAL_LOCATION[sc.checkpoint]


def test_parameters_stay_inside_spec_ranges():
    for s in SEEDS:
        sc = faults.make_scene(s)
        off_mm = 1000.0 * math.hypot(*sc.part_offset_m)
        if sc.fault == "F0":
            assert off_mm <= faults.NOMINAL_JITTER_MM + 1e-6
        elif sc.fault == "F1":
            assert faults.F1_OFFSET_MM[0] - 0.01 <= off_mm <= faults.F1_OFFSET_MM[1] + 0.01
            assert faults.F1_YAW_DEG[0] <= abs(sc.part_yaw_deg) <= faults.F1_YAW_DEG[1]
        elif sc.fault == "F3":
            assert faults.F3_OFFSET_MM[0] - 0.01 <= off_mm <= faults.F3_OFFSET_MM[1] + 0.01
            assert faults.F3_TILT_DEG[0] <= abs(sc.part_tilt_deg) <= faults.F3_TILT_DEG[1]
        elif sc.fault == "F5":
            assert faults.F5_CHIP_MM[0] <= sc.chip_mm <= faults.F5_CHIP_MM[1]


def test_gripper_state_is_physically_consistent():
    for s in SEEDS:
        sc = faults.make_scene(s)
        # Holding something requires closed fingers, and the part must then be "in" the gripper.
        if sc.gripper_holding:
            assert sc.gripper_closed and sc.part_location == "gripper"
        if sc.part_location == "gripper":
            assert sc.gripper_holding
        # An empty grasp is a closed gripper holding nothing.
        if sc.expected.failure_type == "empty_grasp":
            assert sc.gripper_closed and not sc.gripper_holding


def test_only_f4_and_f5_change_the_part():
    for s in SEEDS:
        sc = faults.make_scene(s)
        assert (sc.part_kind == "wrong") == (sc.fault == "F4")
        assert (sc.part_kind == "defective") == (sc.fault == "F5")


def test_scene_serialises_to_json():
    import json

    for s in (1000, 1001, 1002, 1003, 1004, 1005):
        json.dumps(faults.make_scene(s).to_json())
