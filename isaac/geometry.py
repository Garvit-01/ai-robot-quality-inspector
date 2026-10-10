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
