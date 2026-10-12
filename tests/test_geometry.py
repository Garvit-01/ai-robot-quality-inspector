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
