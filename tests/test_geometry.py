"""The chipped-cube mesh must be a closed, outward-facing solid with the right volume.
A face wound the wrong way can render as a hole, which would silently corrupt F5."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "isaac"))
from geometry import chipped_cube_mesh  # noqa: E402

EDGE = 0.05


def _faces(counts, idx):
    out, k = [], 0
    for c in counts:
        out.append(idx[k:k + c])
        k += c
    return out


def _signed_volume(pts, faces):
    # Divergence theorem over a fan triangulation of each face.
    vol = 0.0
    for f in faces:
        a = pts[f[0]]
        for i in range(1, len(f) - 1):
            b, c = pts[f[i]], pts[f[i + 1]]
            vol += (a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
    return vol


@pytest.mark.parametrize("corner", [0, 1, 2, 3])
@pytest.mark.parametrize("chip", [0.012, 0.016, 0.020])
def test_mesh_is_closed_and_consistently_wound(corner, chip):
    pts, counts, idx = chipped_cube_mesh(EDGE, chip, corner)
    faces = _faces(counts, idx)
    assert len(faces) == 7 and counts[-1] == 3  # 6 cube faces + the chip triangle, last
    directed = Counter()
    for f in faces:
        for i in range(len(f)):
            directed[(f[i], f[(i + 1) % len(f)])] += 1
    for (a, b), n in directed.items():
        assert n == 1, f"edge {a}->{b} used {n} times"
        assert directed[(b, a)] == 1, f"edge {a}->{b} has no opposite twin (hole or flipped face)"


@pytest.mark.parametrize("corner", [0, 1, 2, 3])
def test_volume_is_cube_minus_corner_tetrahedron(corner):
    chip = 0.018
    pts, counts, idx = chipped_cube_mesh(EDGE, chip, corner)
    vol = _signed_volume(pts, _faces(counts, idx))
    expected = EDGE ** 3 - chip ** 3 / 6.0
    # Positive volume also proves the faces point outward.
    assert vol == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("corner", [0, 1, 2, 3])
def test_chip_is_on_a_top_corner_in_the_documented_quadrant(corner):
    pts, counts, idx = chipped_cube_mesh(EDGE, 0.015, corner)
    tri = _faces(counts, idx)[-1]
    cx = sum(pts[i][0] for i in tri) / 3
    cy = sum(pts[i][1] for i in tri) / 3
    cz = sum(pts[i][2] for i in tri) / 3
    sx, sy = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}[corner]
    assert cz > 0 and cx * sx > 0 and cy * sy > 0


def test_rejects_impossible_chip():
    with pytest.raises(ValueError):
        chipped_cube_mesh(EDGE, 0.0, 0)
    with pytest.raises(ValueError):
        chipped_cube_mesh(EDGE, EDGE, 0)


# --- crack ribbon --------------------------------------------------------------------
from geometry import crack_ribbon  # noqa: E402


@pytest.mark.parametrize("seed", range(40))
def test_crack_stays_on_the_top_face(seed):
    import random
    rng = random.Random(seed)
    length, angle = rng.uniform(0.022, 0.040), rng.uniform(0, 180)
    pts, counts, idx = crack_ribbon(length, angle, EDGE / 2, seed)
    assert all(abs(x) < EDGE / 2 - 0.002 and abs(y) < EDGE / 2 - 0.002 and z == 0 for x, y, z in pts)
    assert counts == [4] * len(counts) and len(idx) == 4 * len(counts)


def test_crack_faces_up_and_is_hairline_wide():
    pts, counts, idx = crack_ribbon(0.03, 33.0, EDGE / 2, 7)
    for i in range(len(counts)):
        a, b, c = (pts[j] for j in idx[4 * i:4 * i + 3])
        nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        assert nz > 0, "a crack quad faces down and would be invisible from above"
    widths = [((pts[2 * k][0] - pts[2 * k + 1][0]) ** 2 + (pts[2 * k][1] - pts[2 * k + 1][1]) ** 2) ** 0.5
              for k in range(len(pts) // 2)]
    assert max(widths) <= 0.0009 + 1e-12 and min(widths) > 0


# --- fastener defects ------------------------------------------------------------------
from geometry import hex_corner_cut, hex_corner_xy, hex_flat_frame  # noqa: E402

AC, H = 0.0277, 0.013  # M16 nut: across corners, height


def _hex_cloud(n=40):
    """Points on a hex prism shell, corners on the y axis — like the nut mesh."""
    pts = []
    for i in range(6):
        (x0, y0), (x1, y1) = hex_corner_xy(i, AC), hex_corner_xy((i + 1) % 6, AC)
        for k in range(n):
            t = k / n
            for z in (0.0, H / 2, H):
                pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z))
    return pts


@pytest.mark.parametrize("corner", [3, 4])
def test_hex_chip_removes_only_the_chosen_corner(corner):
    pts = _hex_cloud()
    out, moved = hex_corner_cut(pts, corner, AC, H, 0.004)
    assert moved > 0
    cx, cy = hex_corner_xy(corner, AC)
    for (a, b) in zip(pts, out):
        if a != b:
            # Only points near the chosen corner's top move, and they move inward/down.
            # A 4 mm chip: nothing further than 4 mm (plus sampling slack) from the corner.
            assert ((a[0] - cx) ** 2 + (a[1] - cy) ** 2) ** 0.5 < 0.0045 and a[2] > H - 0.0045
            assert b[2] <= a[2] + 1e-12
    # Nothing is left beyond the cutting plane.
    _, moved_again = hex_corner_cut(out, corner, AC, H, 0.004)
    assert moved_again == 0


def test_chip_corners_face_the_inspection_camera():
    # The close-up camera sits at +x, -y; corners 3 (-y) and 4 (+x,-y) face it.
    for c in (3, 4):
        x, y = hex_corner_xy(c, AC)
        assert 0.34 * x - 0.12 * y > 0


def test_flat_frame_sits_on_the_flat_and_faces_out():
    centre, u, v, n = hex_flat_frame(300.0, 0.024, 0.0065)
    assert abs((centre[0] ** 2 + centre[1] ** 2) ** 0.5 - 0.012) < 0.0005
    assert abs(sum(u[i] * n[i] for i in range(3))) < 1e-12 and v == (0.0, 0.0, 1.0)
