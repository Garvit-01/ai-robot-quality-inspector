"""The benchmark's arithmetic. A scoring bug would silently falsify every reported number."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_verifier as bv  # noqa: E402
from _common import FailureType  # noqa: E402


def row(fault, cp, exp_ft, exp_ok, exp_loc, pred_ft=None, pred_ok=None, pred_loc=None, lat=1000):
    pred = None if pred_ft is None else {"failure_type": pred_ft, "step_ok": pred_ok, "location_hint": pred_loc}
    return {"fault": fault, "checkpoint": cp,
            "exp": {"failure_type": exp_ft, "step_ok": exp_ok, "location_hint": exp_loc},
            "pred": pred, "latency_ms": lat, "tokens_in": 100, "tokens_out": 10}


def test_every_failure_type_has_an_action():
    assert set(bv.ACTION) == set(get_args(FailureType))


def test_counts_and_rates():
    rows = [
        row("F0", "CP_POST_PLACE", "none", True, "fixture", "none", True, "fixture"),                # right
        row("F0", "CP_POST_PLACE", "none", True, "fixture", "misaligned_seat", False, "fixture"),    # false alarm
        row("F3", "CP_POST_PLACE", "misaligned_seat", False, "fixture", "none", True, "fixture"),    # miss
        row("F4", "CP_PRE_GRASP", "wrong_part", False, "carrier", "defective_part", False, "carrier"),  # wrong word, right action
        row("F5", "CP_PRE_GRASP", "defective_part", False, "carrier"),                               # unparseable
    ]
    s = bv.summarise(rows)
    o = s["overall"]
    assert o["n"] == 5
    assert o["failure_type_acc"] == 20.0          # only the first row
    assert o["step_ok_acc"] == 40.0               # rows 1 and 4
    assert o["action_acc"] == 40.0                # rows 1 and 4 (wrong vs defective = same action)
    assert o["unparseable"] == 1
    assert s["false_alarm_rate_on_clean"] == 50.0  # 1 of 2 clean scenes flagged
    assert s["miss_rate_on_faults"] == 33.3        # the F3 row: 1 of 3 faulty scenes passed as OK
    conf = {(c["expected"], c["predicted"]): c["count"] for c in s["confusion"]}
    assert conf[("defective_part", "<no answer>")] == 1


def test_unparseable_counts_against_accuracy_not_excluded():
    rows = [row("F0", "CP_PRE_GRASP", "none", True, "carrier"),
            row("F0", "CP_PRE_GRASP", "none", True, "carrier", "none", True, "carrier")]
    assert bv.summarise(rows)["overall"]["failure_type_acc"] == 50.0


def test_gate_requires_both_checkpoints():
    good = [row("F0", "CP_POST_LIFT", "none", True, "gripper", "none", True, "gripper")] * 9
    place_bad = [row("F3", "CP_POST_PLACE", "misaligned_seat", False, "fixture", "none", True, "fixture")] * 9
    s = bv.summarise(good + place_bad)
    assert s["go_no_go"]["go"] is False
    assert "CP_POST_PLACE" in s["go_no_go"]["checkpoints_below_no_go"]
    place_good = [row("F3", "CP_POST_PLACE", "misaligned_seat", False, "fixture", "misaligned_seat", False, "fixture")] * 9
    assert bv.summarise(good + place_good)["go_no_go"]["go"] is True
