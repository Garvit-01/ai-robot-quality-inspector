"""Cell dimensions and part placement, in pure Python.

Everything that decides WHERE things are lives here, not in cell.py, so it can be tested
without Isaac Sim. That matters because placement is where the scene's worst bugs were:
an earlier version sank tilted parts 11 mm into the fixture wall, floating over the
pocket floor. cell.py only turns the poses computed here into USD.

Frame: metres, Z up, table top at z = 0, robot base at the origin.

                 +y
                  |        [ fixture ]  (0.45, +0.22)
     robot base   |
        (0,0) ----+-----------------------[ reject bin ] (0.74, 0)  ---- +x
                  |
                  |        [ carrier ]  (0.45, -0.22)
"""
from __future__ import annotations

import math

TABLE_SIZE = (1.20, 0.90, 0.04)
TABLE_CENTER = (0.40, 0.0)
FLOOR_Z = -0.75

CARRIER_CENTER = (0.45, -0.22)
CARRIER_SIZE = (0.18, 0.12, 0.012)
CARRIER_TOP = CARRIER_SIZE[2]
CARRIER_SLOT = (CARRIER_CENTER[0] - 0.03, CARRIER_CENTER[1])  # where the part should sit
CARRIER_TAG = (CARRIER_CENTER[0] + 0.055, CARRIER_CENTER[1])
TAG_SIZE = 0.05
NEST_INNER = 0.056  # printed nest outline on the carrier, around the slot
NEST_LINE = 0.003

FIXTURE_CENTER = (0.45, 0.22)
FIXTURE_OUTER = 0.13
FIXTURE_BASE_TOP = 0.012
FIXTURE_POCKET = 0.054  # 50 mm part + 2 x 2 mm clearance
FIXTURE_WALL_TOP = 0.034  # 22 mm walls: a proud seat must look clearly different

BIN_CENTER = (0.74, 0.0)
BIN_SIZE = (0.14, 0.14, 0.07)

PART_EDGE = 0.05

# Regions the F2 "dropped" part can land in. Clear of carrier, fixture, bin and robot
# base by at least 15 mm at any yaw (checked in review).
TABLE_DROP_X = (0.22, 0.62)
TABLE_DROP_Y = (-0.10, 0.10)
FLOOR_DROP_X = (0.30, 0.80)
# Beyond the table's long edges, chosen so the part is NEVER visible from the overview
# camera: hidden behind the table edge, as in a real cell where a part that falls off
# the table disappears from view. Ray-casting 48k part corners across the camera-jitter
# range: 0 visible (an earlier 0.56-0.72 band leaked ~4%).
FLOOR_DROP_Y_ABS = (0.52, 0.66)

GRIPPER_GAP_OPEN = 0.090
GRIPPER_GAP_EMPTY = 0.008
FINGER = (0.020, 0.012, 0.055)  # x, y (thickness), z

# Bounding box (x, y, z) of each part variant, keyed by (part_kind, wrong_shape).
PART_DIMS = {
    ("correct", None): (PART_EDGE, PART_EDGE, PART_EDGE),
    ("defective", None): (PART_EDGE, PART_EDGE, PART_EDGE),
    ("wrong", "cylinder"): (0.05, 0.05, 0.05),
    ("wrong", "small_cube"): (0.035, 0.035, 0.035),
    ("wrong", "tall_block"): (0.05, 0.05, 0.075),
}

Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def part_key(spec) -> tuple[str, str | None]:
    return (spec.part_kind, spec.wrong_shape if spec.part_kind == "wrong" else None)


# --- rotation helpers -------------------------------------------------------------------
def _matmul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def rot_z(deg: float) -> Mat3:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rot_axis(axis: tuple[float, float, float], deg: float) -> Mat3:
    """Rodrigues rotation about a unit axis, right-handed."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    C = 1 - c
    return ((c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C))


def apply(R: Mat3, p) -> tuple[float, float, float]:
    return tuple(R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2] for i in range(3))  # type: ignore[return-value]


def surface_samples(dims, n: int = 33) -> list[tuple[float, float, float]]:
    """Points on all six faces of a centred box, n x n per face.

    Density matters: the pocket wall's inner edge can fall between samples, and the part
    then sinks into the wall by (spacing x slope of the tilted face). At 9 per face
    (6.25 mm spacing) that reached 1 mm, caught by tests/test_layout.py; at 33 (1.6 mm)
    the worst case is about 0.4 mm, for ~6.5k points and a few ms per scene."""
    hx, hy, hz = (d / 2 for d in dims)
    lin = [-1 + 2 * i / (n - 1) for i in range(n)]
    pts = []
    for a in lin:
        for b in lin:
            pts += [(a * hx, b * hy, hz), (a * hx, b * hy, -hz),
                    (a * hx, hy, b * hz), (a * hx, -hy, b * hz),
                    (hx, a * hy, b * hz), (-hx, a * hy, b * hz)]
    return pts


# --- the fixture as a height field ------------------------------------------------------
def fixture_support_z(x: float, y: float) -> float:
    """Height of the top of solid material under (x, y) near the fixture: the pocket
    floor inside the pocket, the wall top over the walls, the table outside."""
    dx, dy = abs(x - FIXTURE_CENTER[0]), abs(y - FIXTURE_CENTER[1])
    if dx <= FIXTURE_POCKET / 2 and dy <= FIXTURE_POCKET / 2:
        return FIXTURE_BASE_TOP
    if dx <= FIXTURE_OUTER / 2 and dy <= FIXTURE_OUTER / 2:
        return FIXTURE_WALL_TOP
    return 0.0


def rest_on_fixture(R: Mat3, xy, dims) -> float:
    """Centre height at which a part with orientation R, centred above `xy`, rests on
    the fixture without any point inside solid material: raise it until the most
    constraining surface point just touches."""
    z = -math.inf
    for p in surface_samples(dims):
        q = apply(R, p)
        z = max(z, fixture_support_z(xy[0] + q[0], xy[1] + q[1]) - q[2])
    return z


# --- the main entry point ---------------------------------------------------------------
def part_pose(spec) -> tuple[tuple[float, float, float], Mat3]:
    """World position and rotation matrix of the part for a SceneSpec.
    The part held by the gripper is handled by the caller (it hangs at the grasp point)."""
    key = part_key(spec)
    dims = PART_DIMS[key]
    half_h = dims[2] / 2
    ox, oy = spec.part_offset_m
    yaw = spec.part_yaw_deg
    loc = spec.part_location

    if loc == "carrier":
        return (CARRIER_SLOT[0] + ox, CARRIER_SLOT[1] + oy, CARRIER_TOP + half_h), rot_z(yaw)

    if loc == "fixture":
        xy = (FIXTURE_CENTER[0] + ox, FIXTURE_CENTER[1] + oy)
        R = rot_z(yaw)
        tilt = abs(spec.part_tilt_deg)
        if tilt > 0:
            # Tilt so the side the part is offset toward (where it caught the rim) is the
            # high side. Rotating about a = (-sin t, cos t, 0) by a NEGATIVE angle raises
            # the points lying in the offset direction (cos t, sin t).
            theta = math.atan2(oy, ox)
            axis = (-math.sin(theta), math.cos(theta), 0.0)
            R = _matmul(rot_axis(axis, -tilt), R)
        return (xy[0], xy[1], rest_on_fixture(R, xy, dims)), R

    if loc == "table":
        fx, fy = spec.landing_frac
        x = TABLE_DROP_X[0] + fx * (TABLE_DROP_X[1] - TABLE_DROP_X[0])
        y = TABLE_DROP_Y[0] + fy * (TABLE_DROP_Y[1] - TABLE_DROP_Y[0])
        return (x, y, half_h), rot_z(yaw)

    if loc == "floor":
        fx, fy = spec.landing_frac
        x = FLOOR_DROP_X[0] + fx * (FLOOR_DROP_X[1] - FLOOR_DROP_X[0])
        y = spec.floor_side * (FLOOR_DROP_Y_ABS[0] + fy * (FLOOR_DROP_Y_ABS[1] - FLOOR_DROP_Y_ABS[0]))
        return (x, y, FLOOR_Z + half_h), rot_z(yaw)

    raise ValueError(f"part_location {loc!r} is posed via the gripper, not here")


def gripper_anchor(spec) -> tuple[float, float, float]:
    if spec.gripper_at == "above_carrier":
        return CARRIER_SLOT[0], CARRIER_SLOT[1], CARRIER_TOP
    return FIXTURE_CENTER[0], FIXTURE_CENTER[1], FIXTURE_BASE_TOP


def usd_matrix_rows(R: Mat3, t) -> list[list[float]]:
    """4x4 rows in USD's row-vector convention (p' = p * M): rotation transposed into the
    upper 3x3, translation in the last row."""
    return [[R[0][0], R[1][0], R[2][0], 0.0],
            [R[0][1], R[1][1], R[2][1], 0.0],
            [R[0][2], R[1][2], R[2][2], 0.0],
            [t[0], t[1], t[2], 1.0]]
