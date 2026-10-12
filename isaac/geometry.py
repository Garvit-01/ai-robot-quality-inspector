"""Pure-Python geometry for the cell's meshes. No Isaac Sim imports, so it can be tested
in the project venv; cell.py converts the plain tuples into USD types."""
from __future__ import annotations

Vec3 = tuple[float, float, float]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def chipped_cube_mesh(edge: float, chip: float, corner: int):
    """A cube centred on the origin with one top corner cut off by a plane.

    The cut passes through the three points `chip` metres from the corner along its
    three edges, leaving a small triangular facet — what a knocked corner on a machined
    block looks like.

    Returns (points, face_vertex_counts, face_vertex_indices). Faces are wound
    counter-clockwise seen from outside, and the chip triangle is always the LAST face
    (index 6) so the scene can bind a separate material to it.

    `corner` picks one of the four top corners, in this order of (x, y) signs:
    0 = (-,-), 1 = (+,-), 2 = (+,+), 3 = (-,+).
    """
    if not 0 < chip < edge:
        raise ValueError(f"chip must be in (0, edge); got chip={chip}, edge={edge}")
    h = edge / 2
    pts: list[Vec3] = [(x, y, z) for z in (-h, h) for (x, y) in ((-h, -h), (h, -h), (h, h), (-h, h))]
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5]]
    v = 4 + (corner % 4)
    V = pts[v]
    neighbours = [i for i, p in enumerate(pts)
                  if i != v and sum(abs(a - b) > 1e-12 for a, b in zip(p, V)) == 1]
    t = chip / edge
    new_idx: dict[int, int] = {}
    for u in neighbours:
        U = pts[u]
        pts.append(tuple(V[k] + (U[k] - V[k]) * t for k in range(3)))  # type: ignore[arg-type]
        new_idx[u] = len(pts) - 1

    out: list[list[int]] = []
    for f in faces:
        if v in f:
            k = f.index(v)
            prev_v, next_v = f[k - 1], f[(k + 1) % len(f)]
            f = f[:k] + [new_idx[prev_v], new_idx[next_v]] + f[k + 1:]
        out.append(f)
    tri = [new_idx[u] for u in neighbours]
    a, b, c = (pts[i] for i in tri)
    if _dot(_cross(_sub(b, a), _sub(c, a)), V) < 0:  # face outward, towards the lost corner
        tri.reverse()
    out.append(tri)

    # The removed corner is no longer referenced; keep indices stable rather than
    # renumbering — an unused point is harmless to the renderer.
    return pts, [len(f) for f in out], [i for f in out for i in f]


def crack_ribbon(length: float, angle_deg: float, half_edge: float, seed: int,
                 width: float = 0.0009, segments: int = 7, margin: float = 0.004):
    """A thin, jagged dark ribbon lying on a cube's top face: a visible crack.

    The crack runs `length` metres at `angle_deg` across the face, wandering up to
    1.5 mm sideways at each kink, and tapers to a point at both ends. It stays at
    least `margin` inside the face. Real hairline cracks are 0.1-0.5 mm wide; 0.9 mm
    is deliberately at the visible end, so the dataset tests whether a model notices a
    crack, not whether our camera can resolve one.

    Points are in the part's frame with z = 0 at the face; the caller lifts the ribbon a
    fraction of a millimetre above the face so it doesn't z-fight with it.
    Returns (points, face_vertex_counts, face_vertex_indices): one quad per segment.
    """
    import math
    import random

    rng = random.Random(seed * 7477 + 11)
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)  # along the crack
    nx, ny = -uy, ux  # across it
    limit = half_edge - margin
    # Shorten if the face can't hold it at this angle, then place its centre so both
    # ends stay inside the face.
    reach = max(abs(ux), abs(uy)) or 1.0
    length = min(length, 2 * limit / reach * 0.98)
    half = length / 2
    cx_max = max(0.0, limit - half * abs(ux) - 0.0015)
    cy_max = max(0.0, limit - half * abs(uy) - 0.0015)
    cx, cy = rng.uniform(-cx_max, cx_max), rng.uniform(-cy_max, cy_max)

    centre = []
    for i in range(segments + 1):
        t = -half + length * i / segments
        wobble = 0.0 if i in (0, segments) else rng.uniform(-0.0015, 0.0015)
        centre.append((cx + ux * t + nx * wobble, cy + uy * t + ny * wobble))

    pts: list[tuple[float, float, float]] = []
    for i, (x, y) in enumerate(centre):
        taper = math.sin(math.pi * i / segments)  # 0 at the tips, 1 in the middle
        w = max(width * taper, width * 0.15) / 2
        pts.append((x + nx * w, y + ny * w, 0.0))
        pts.append((x - nx * w, y - ny * w, 0.0))
    counts, idx = [], []
    for i in range(segments):
        l0, r0, l1, r1 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        counts.append(4)
        idx += [l0, r0, r1, l1]  # counter-clockwise seen from +z, so it faces the camera
    return pts, counts, idx
