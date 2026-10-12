"""Properties of the fastener part set. Each test here would have caught a bug that a
code review found by hand (Oct 12): a background environment tied to the fault class, nuts
cropped out of the close-up, fingers clamping the nut's corners, and a nest outline the
correct part didn't fit."""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "isaac"))

import dr  # noqa: E402
import faults  # noqa: E402
import layout as L  # noqa: E402
import partsets  # noqa: E402
from geometry import hex_corner_xy  # noqa: E402

FS = partsets.FASTENER
EVAL = range(5000, 5120)
WIDE = range(2000, 3200)


def test_environment_is_independent_of_the_fault():
    """Every fault class must see every workshop environment. The old seed % 4 choice,
    with the fault at seed % 6, gave defects two environments and wrong parts the other
    two: a background a model could learn instead of the defect."""
    seen = defaultdict(set)
    for s in WIDE:
        seen[faults.make_scene(s, "fastener").fault].add(dr.make_dr(s, True, "fastener")["hdr"])
    for fault, hdrs in seen.items():
        assert hdrs == set(partsets.WORKSHOP_HDRS), f"{fault} only sees {len(hdrs)} environments"
    # And not just present: no class is dominated by one environment.
    for fault in seen:
        c = Counter(dr.make_dr(s, True, "fastener")["hdr"] for s in WIDE
                    if faults.make_scene(s, "fastener").fault == fault)
        assert max(c.values()) / sum(c.values()) < 0.4, fault


def test_cube_randomisation_has_no_environment():
    assert "hdr" not in dr.make_dr(1000, True, "cube")


def test_fastener_sub_cases_are_balanced():
    def sub(fault, attr):
        return Counter(getattr(faults.make_scene(s, "fastener"), attr) for s in EVAL
                       if faults.make_scene(s, "fastener").fault == fault)
    assert sorted(sub("F4", "wrong_shape").values()) in ([6, 7, 7], [7, 7, 6])
    assert sorted(sub("F5", "defect_type").values()) in ([6, 7, 7], [7, 7, 6])
    assert set(sub("F4", "wrong_shape")) == {p.name for p in FS.wrong}


def test_fastener_ranges_and_serialisation():
    for s in WIDE:
        sc = faults.make_scene(s, "fastener")
        assert sc.to_json()["part_set"] == "fastener"
        if sc.fault == "F1":
            assert FS.f1_offset_mm[0] - 0.01 <= 1000 * math.hypot(*sc.part_offset_m) <= FS.f1_offset_mm[1] + 0.01
        if sc.defect_type == "chip":
            assert sc.chip_corner in FS.chip_corners and FS.chip_mm[0] <= sc.chip_mm <= FS.chip_mm[1]
        if sc.defect_type == "rust":
            assert 0.25 <= sc.truth["rust_coverage"] <= 0.5


def _project(eye, target, p, focal_mm=28.0, aperture_mm=20.955, aspect=4 / 3):
    """Pinhole projection matching cell.py's close-up camera: returns normalised image
    coordinates, |x| <= 1 and |y| <= 1 inside the frame."""
    f = [target[i] - eye[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    r = [f[1], -f[0], 0.0]
    rn = math.hypot(r[0], r[1])
    r = [r[0] / rn, r[1] / rn, 0.0]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    d = [p[i] - eye[i] for i in range(3)]
    z = sum(d[i] * f[i] for i in range(3))
    half_w = aperture_mm / 2 / focal_mm
    return (sum(d[i] * r[i] for i in range(3)) / z / half_w,
            sum(d[i] * u[i] for i in range(3)) / z / (half_w / aspect))


def test_every_pre_grasp_part_stays_in_the_close_up():
    """Every part waiting on the carrier — displaced nuts, the tall bolt, the big M20 — is
    fully in the inspection frame. The first framing cropped 36% of displaced nuts (with
    cube-sized offsets) and later the 41 mm bolt's top in about 1 scene in 10."""
    for s in WIDE:
        sc = faults.make_scene(s, "fastener")
        if sc.checkpoint != "CP_PRE_GRASP":
            continue
        xyz, R = L.part_pose(sc)
        roi = (L.CARRIER_SLOT[0], L.CARRIER_SLOT[1], L.CARRIER_TOP + FS.inspect_aim_dz)
        j = dr.make_dr(s, True, "fastener")["cam_jitter"]
        o = FS.closeup_offset
        eye = (roi[0] + o[0] + j[0] * 0.5, roi[1] + o[1] + j[1] * 0.5, roi[2] + o[2] + j[2] * 0.5)
        dims = L.part_def(sc).dims
        for q in L.surface_samples(dims, 5, L.part_def(sc).shape):
            p = [xyz[i] + L.apply(R, q)[i] for i in range(3)]
            x, y = _project(eye, roi, p)
            assert abs(x) < 1 and abs(y) < 1, f"seed {s}: part leaves the frame ({x:.2f}, {y:.2f})"


def test_held_nut_is_gripped_across_its_flats():
    """The fingers close along y. Turned by held_yaw_deg, the nut's extent along y must
    equal the holding gap — across the flats (24 mm), not the corners (27.7 mm)."""
    R = L.rot_z(FS.held_yaw_deg)
    pts = [L.apply(R, (*hex_corner_xy(i, FS.correct.dims[1]), 0.0)) for i in range(6)]
    extent_y = max(p[1] for p in pts) - min(p[1] for p in pts)
    assert extent_y == pytest.approx(FS.gap_holding, abs=0.0002)


def test_correct_nut_fits_inside_the_nest_outline_but_m20_does_not():
    across_corners = FS.correct.dims[1]
    assert across_corners + 2 * (FS.seated_jitter_mm / 1000) < FS.nest_inner
    m20 = next(p for p in FS.wrong if p.name == "nut_m20")
    assert m20.dims[1] > FS.nest_inner  # oversize: the prompt must call this wrong, not displaced


def test_chip_facets_face_the_close_up_camera():
    """Uses the fastener's real close-up direction and the chip facet's normal, not a 2-D
    stand-in: corner 3 passed a 2-D check while its facet faced the camera at only ~0.5."""
    o = FS.closeup_offset
    view = [c / math.sqrt(sum(v * v for v in o)) for c in o]
    for corner in FS.chip_corners:
        cx, cy = hex_corner_xy(corner, FS.correct.dims[1])
        r = math.hypot(cx, cy)
        normal = (cx / r / math.sqrt(2), cy / r / math.sqrt(2), 1 / math.sqrt(2))
        assert sum(normal[i] * view[i] for i in range(3)) > 0.75, corner
