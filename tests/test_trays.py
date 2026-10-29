"""The tray catalogue: determinism, balance, ground truth per camera view."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "isaac"))

import partsets  # noqa: E402
import trays  # noqa: E402
from trays import BAD_KINDS, GOOD_PER_TRAY, SLOTS, SPLITS, make_tray

ALL_TRAY_SEEDS = [s for r in SPLITS.values() for s in r]


def test_same_seed_same_tray():
    assert make_tray(21003) == make_tray(21003)
    assert make_tray(21003).to_json() == make_tray(21003).to_json()


def test_different_seeds_differ():
    assert len({make_tray(s) for s in range(21000, 21010)}) == 10


@pytest.mark.parametrize("seed", ALL_TRAY_SEEDS)
def test_tray_shape(seed):
    t = make_tray(seed)
    assert len(t.parts) == SLOTS
    assert [p.slot for p in t.parts] == list(range(SLOTS))
    assert sum(p.good for p in t.parts) == GOOD_PER_TRAY


def test_every_three_trays_cover_all_bad_kinds_once():
    for start in range(21000, 21039, 3):
        kinds = Counter()
        for seed in range(start, start + 3):
            for p in make_tray(seed).parts:
                if p.kind == "wrong":
                    kinds[f"wrong:{p.part}"] += 1
                elif p.kind == "defective":
                    kinds[f"{p.defect}:{p.side}"] += 1
        assert kinds == Counter(BAD_KINDS), start


def test_splits_disjoint_from_single_scene_seeds():
    single = set(range(1000, 1100)) | set(range(2000, 2060)) | set(range(5000, 5120)) | set(range(10000, 11800))
    assert not single & set(ALL_TRAY_SEEDS)
    assert not set(SPLITS["dev"]) & set(SPLITS["eval"])


@pytest.mark.parametrize("seed", ALL_TRAY_SEEDS)
def test_parameters_stay_in_range(seed):
    ps = partsets.FASTENER
    for p in make_tray(seed).parts:
        if p.defect == "chip":
            assert ps.chip_mm[0] <= p.chip_mm <= ps.chip_mm[1]
            allowed = ps.chip_corners if p.side == "top" else trays.UNDERSIDE_CHIP_CORNERS
            assert p.chip_corner in allowed
        if p.defect == "crack":
            assert ps.crack_mm[0] <= p.crack_mm <= ps.crack_mm[1]
        if p.defect == "rust":
            assert trays.RUST_COVERAGE[0] <= p.rust_coverage <= trays.RUST_COVERAGE[1]
        if p.kind != "defective":
            assert p.defect is None and p.side is None


def test_expected_per_view():
    by_kind = {}
    for seed in range(21000, 21003):
        for p in make_tray(seed).parts:
            by_kind[(p.kind, p.side)] = p
    good = by_kind[("good", None)]
    assert (good.expected("top"), good.expected("underside"), good.correct_action) == ("none", "none", "accept")
    wrong = by_kind[("wrong", None)]
    assert (wrong.expected("top"), wrong.expected("underside"), wrong.correct_action) == ("wrong_part", "wrong_part", "reject")
    top = by_kind[("defective", "top")]
    assert (top.expected("top"), top.expected("underside")) == ("defective_part", "none")
    under = by_kind[("defective", "underside")]
    # The case the upward camera exists for: invisible from the top, still a reject.
    assert (under.expected("top"), under.expected("underside"), under.correct_action) == ("none", "defective_part", "reject")


def test_wrong_parts_are_real_part_names():
    names = {d.name for d in partsets.FASTENER.wrong}
    for seed in ALL_TRAY_SEEDS:
        for p in make_tray(seed).parts:
            if p.kind == "wrong":
                assert p.part in names
            else:
                assert p.part == partsets.FASTENER.correct.name


@pytest.mark.parametrize("seed", ALL_TRAY_SEEDS)
def test_feed_yaw_matches_training_jitter(seed):
    for p in make_tray(seed).parts:
        assert trays.FEED_YAW_DEG[0] <= p.feed_yaw_deg <= trays.FEED_YAW_DEG[1]


@pytest.mark.parametrize("kind", trays.BAD_KINDS)
def test_injected_part_is_the_requested_bad_kind(kind):
    p = trays.injected_part(kind, 3, 21000)
    what, detail = kind.split(":")
    assert p.slot == 3 and not p.good and p.correct_action == "reject"
    if what == "wrong":
        assert (p.kind, p.part) == ("wrong", detail)
    else:
        assert (p.kind, p.defect, p.side) == ("defective", what, detail)
    assert p == trays.injected_part(kind, 3, 21000)  # deterministic


def test_injected_part_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        trays.injected_part("good", 0, 21000)


def test_chip_trays_hold_only_chips_on_both_sides():
    sides = Counter()
    for seed in SPLITS["train_chips"]:
        for p in make_tray(seed).parts:
            if not p.good:
                assert (p.kind, p.defect) == ("defective", "chip")
                sides[p.side] += 1
    assert sides["top"] == sides["underside"] == 150
