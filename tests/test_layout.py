"""Placement must be physically possible. A part inside a wall, or floating over the
pocket floor, is a picture of something that cannot happen — and Tier 2 would learn
from it. These tests caught nothing yet only because they were written after the review
that found exactly that bug in the previous version."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "isaac"))
import faults  # noqa: E402
import layout as L  # noqa: E402

SEEDS = range(1000, 2200)
FIXTURE_SCENES = [s for s in SEEDS if faults.make_scene(s).part_location == "fixture"]


def _gaps(spec, n):
    """Height above solid material of every surface sample, on an n x n grid per face."""
    xyz, R = L.part_pose(spec)
    dims = L.PART_DIMS[L.part_key(spec)]
    out = []
    for q in (L.apply(R, p) for p in L.surface_samples(dims, n)):
        x, y, z = (xyz[i] + q[i] for i in range(3))
        out.append(z - L.fixture_support_z(x, y))
    return out


@pytest.mark.parametrize("seed", FIXTURE_SCENES)
def test_fixture_parts_never_penetrate_and_always_touch(seed):
    spec = faults.make_scene(seed)
    # Penetration is checked on grids the solver does NOT use (13 and 41 per face), so a
    # wall edge falling between the solver's own samples is caught. Tolerance 0.5 mm.
    for n in (13, 41):
        worst = min(_gaps(spec, n))
        assert worst > -0.0005, f"part {-worst * 1000:.2f} mm inside the fixture (grid {n})"
    # Contact is checked on the solver's own grid: a coarser grid can simply miss the
    # one point that touches and report a float that isn't there.
    assert min(_gaps(spec, 33)) < 1e-9, "part floats: nothing under it touches"


def test_clean_seated_parts_rest_on_the_pocket_floor():
    for s in FIXTURE_SCENES:
        spec = faults.make_scene(s)
        if spec.fault == "F0":
            z = L.part_pose(spec)[0][2]
            assert z == pytest.approx(L.FIXTURE_BASE_TOP + L.PART_EDGE / 2, abs=1e-9), s


def test_misaligned_parts_are_raised_on_the_side_they_are_offset_toward():
    for s in FIXTURE_SCENES:
        spec = faults.make_scene(s)
        if spec.fault != "F3":
            continue
        xyz, R = L.part_pose(spec)
        seated = L.FIXTURE_BASE_TOP + L.PART_EDGE / 2
        assert xyz[2] > seated + 0.003, f"{s}: only {(xyz[2] - seated) * 1000:.1f} mm proud"
        ox, oy = spec.part_offset_m
        n = math.hypot(ox, oy)
        d = (ox / n * 0.025, oy / n * 0.025, 0.0)
        high, low = L.apply(R, d), L.apply(R, (-d[0], -d[1], 0.0))
        assert high[2] > low[2], f"{s}: raised side faces away from the offset"


def test_carrier_parts_sit_on_the_carrier():
    for s in SEEDS:
        spec = faults.make_scene(s)
        if spec.part_location != "carrier":
            continue
        x, y, _ = L.part_pose(spec)[0]
        assert abs(x - L.CARRIER_CENTER[0]) < L.CARRIER_SIZE[0] / 2
        assert abs(y - L.CARRIER_CENTER[1]) < L.CARRIER_SIZE[1] / 2


def test_floor_drops_land_beyond_the_table_and_on_both_sides():
    sides = set()
    for s in SEEDS:
        spec = faults.make_scene(s)
        if spec.part_location == "floor":
            x, y, z = L.part_pose(spec)[0]
            assert abs(y) > L.TABLE_SIZE[1] / 2 and z < 0
            sides.add(spec.floor_side)
    assert sides == {-1, 1}


def test_usd_matrix_is_rotation_transposed_plus_translation():
    R = L.rot_z(90)
    m = L.usd_matrix_rows(R, (1, 2, 3))
    # Row-vector convention: (1,0,0,1) * M should give R*(1,0,0) + t = (0,1,0) + t.
    p = [sum([1, 0, 0, 1][k] * m[k][j] for k in range(4)) for j in range(4)]
    assert p[:3] == pytest.approx([1, 3, 3])
