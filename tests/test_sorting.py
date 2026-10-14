"""Sorting-line decisions and scoring (selfheal/sorting.py, eval/bench_sorting.py)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

import bench_sorting  # noqa: E402
import trays  # noqa: E402
from selfheal.sorting import decide, outcome, policy_actions  # noqa: E402


def test_decide():
    assert decide("none", "none") == "accept"
    assert decide("wrong_part", "none") == "reject"
    assert decide("none", "defective_part") == "reject"
    assert decide("unclear", "none") == "reject"  # never ship what the inspector can't vouch for
    assert decide("none") == "accept"  # top camera only


def test_outcomes():
    assert outcome("accept", True) == "correct"
    assert outcome("accept", False) == "shipped_bad"
    assert outcome("reject", True) == "scrapped_good"
    assert outcome("reject", False) == "correct"


def test_today_factory_accepts_everything():
    assert policy_actions("wrong_part", "defective_part")["B"] == "accept"


def _oracle_rows(seeds):
    """What run_sorting.py logs when every verdict is the ground truth."""
    rows = []
    for seed in seeds:
        for p in trays.make_tray(seed).parts:
            rows.append({"tray": seed, "slot": p.slot, "part": p.part, "kind": p.kind, "defect": p.defect,
                         "side": p.side, "verdict_top": {"failure_type": p.expected("top")},
                         "verdict_underside": {"failure_type": p.expected("underside")},
                         "ended_in": "rack" if p.good else "bin", "handled_ok": True, "seconds": 10})
    return rows


def test_perfect_verdicts_ship_nothing_and_only_the_underside_camera_catches_underside_damage():
    s = bench_sorting.summarise(_oracle_rows(range(21000, 21009)))  # 9 trays: every bad kind 3 times
    assert s["parts"] == 54 and s["bad_parts"] == 27
    assert s["policies"]["ours"] == {"correct": 54}
    assert s["policies"]["B"]["shipped_bad"] == 27  # the gripper sensor ships every bad part
    top = s["by_kind"]["top_only"]
    for kind in ("chip:underside", "crack:underside", "rust:underside"):
        assert top[kind] == {"shipped_bad": 3}  # invisible from above
    for kind in ("chip:top", "wrong:nut_m12", "wrong:bolt_m16"):
        assert top[kind] == {"correct": 3}


def test_bench_reads_a_run_folder(tmp_path):
    (tmp_path / "parts.jsonl").write_text("\n".join(json.dumps(r) for r in _oracle_rows([21000])) + "\n")
    md = bench_sorting.render(bench_sorting.summarise(bench_sorting.load(tmp_path)), tmp_path)
    assert "Bad parts shipped" in md and "| ours, top + underside cameras | 6/6 |" in md
