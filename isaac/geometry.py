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
                 width: float = 0.0009, segments: int = 7, margin: float = 0.004,
                 half_height: float | None = None, wobble: float = 0.0015):
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
    limit_x = half_edge - margin
    limit_y = (half_edge if half_height is None else half_height) - margin
    # Shorten if the face can't hold it at this angle, then place its centre so both
    # ends stay inside the face.
    fit = min(limit_x / abs(ux) if abs(ux) > 1e-9 else float("inf"),
              limit_y / abs(uy) if abs(uy) > 1e-9 else float("inf"))
    length = min(length, 2 * fit * 0.98)
    half = length / 2
    cx_max = max(0.0, limit_x - half * abs(ux) - wobble)
    cy_max = max(0.0, limit_y - half * abs(uy) - wobble)
    cx, cy = rng.uniform(-cx_max, cx_max), rng.uniform(-cy_max, cy_max)

    centre = []
    for i in range(segments + 1):
        t = -half + length * i / segments
        w_off = 0.0 if i in (0, segments) else rng.uniform(-wobble, wobble)
        centre.append((cx + ux * t + nx * w_off, cy + uy * t + ny * w_off))

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



# --- fasteners ---------------------------------------------------------------------------
import math as _math


def hex_corner_xy(corner: int, across_corners: float) -> tuple[float, float]:
    """Top-view position of one of a hex nut's six corners. The nut's corners lie on its
    y axis (the probe measured 27.7 mm across corners along y, 24 mm across flats along
    x), numbered counter-clockwise from +y: 0 = +y, 3 = -y, 4 = (+x, -y)."""
    a = _math.radians(90.0 + 60.0 * corner)
    r = across_corners / 2
    return (r * _math.cos(a), r * _math.sin(a))


def hex_corner_cut(points, corner: int, across_corners: float, top_z: float, chip: float):
    """Chip one top corner of a hex part: every point beyond a cutting plane is projected
    onto it, which is exactly a planar cut for the convex region around a corner.

    The plane faces diagonally out of the corner (half outward, half up). `chip` is what
    you see: how far the cut reaches along each top edge from the corner. (Specifying the
    plane's depth directly was wrong: a hex edge meets the corner at 60 degrees to the
    radial direction, so a "4 mm" depth reached 11 mm along the edge — a test caught it.)
    Works on any mesh, including NVIDIA's real nut with its 13k points; returns new points
    and how many moved."""
    cx, cy = hex_corner_xy(corner, across_corners)
    r = _math.hypot(cx, cy)
    n = (cx / r / _math.sqrt(2), cy / r / _math.sqrt(2), 1 / _math.sqrt(2))
    nx_, ny_ = hex_corner_xy((corner + 1) % 6, across_corners)
    el = _math.hypot(nx_ - cx, ny_ - cy)
    e = ((nx_ - cx) / el, (ny_ - cy) / el)  # unit vector along an adjacent top edge
    depth = chip * abs(n[0] * e[0] + n[1] * e[1])
    corner_pt = (cx, cy, top_z)
    d0 = sum(n[i] * corner_pt[i] for i in range(3)) - depth  # plane: n . p = d0
    out, moved = [], 0
    for p in points:
        excess = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - d0
        if excess > 1e-12:  # points already on the plane stay put (float noise)
            out.append((p[0] - n[0] * excess, p[1] - n[1] * excess, p[2] - n[2] * excess))
            moved += 1
        else:
            out.append(tuple(p))
    return out, moved


def hex_flat_frame(flat_angle_deg: float, across_flats: float, mid_z: float, lift: float = 0.0002):
    """Centre, in-plane axes (u horizontal, v up) and outward normal of one flat side face
    of a hex part. Flats face 0, 60, ... 300 degrees."""
    a = _math.radians(flat_angle_deg)
    nrm = (_math.cos(a), _math.sin(a), 0.0)
    apothem = across_flats / 2 + lift
    centre = (nrm[0] * apothem, nrm[1] * apothem, mid_z)
    u = (-nrm[1], nrm[0], 0.0)
    v = (0.0, 0.0, 1.0)
    return centre, u, v, nrm


def place_on_face(points2d, centre, u, v):
    """Map (x, y, 0) points of a flat 2-D shape onto a face given by its frame."""
    return [(centre[0] + x * u[0] + y * v[0], centre[1] + x * u[1] + y * v[1], centre[2] + x * u[2] + y * v[2])
            for x, y, _ in points2d]


def block_with_round_hole(outer: float, radius: float, z0: float, z1: float, segments: int = 64):
    """A square block of side `outer` from z0 to z1 with a vertical round hole of
    `radius` through it, centred on the origin: the counterbore wall of the fastener
    fixture. Top face, outer walls and the bore wall; no bottom (it sits on a base plate).

    The top face is triangulated by pairing each bore point with the point where the same
    ray from the centre meets the square, so the mesh has no gaps for any segment count.
    Returns (points, face_vertex_counts, face_vertex_indices), wound counter-clockwise
    seen from outside the material.
    """
    import math
    if not 0 < radius < outer / 2:
        raise ValueError("the hole must fit inside the block")
    half = outer / 2
    pts: list[tuple[float, float, float]] = []
    for k in range(segments):
        a = 2 * math.pi * k / segments
        c, s = math.cos(a), math.sin(a)
        t = half / max(abs(c), abs(s))  # ray from the centre to the square
        pts += [(radius * c, radius * s, z1), (t * c, t * s, z1),  # bore top, square top
                (radius * c, radius * s, z0), (t * c, t * s, z0)]  # bore bottom, square bottom
    counts, idx = [], []
    for k in range(segments):
        a, b = 4 * k, 4 * ((k + 1) % segments)
        bore_top, sq_top, bore_bot, sq_bot = a, a + 1, a + 2, a + 3
        nbore_top, nsq_top, nbore_bot, nsq_bot = b, b + 1, b + 2, b + 3
        idx += [bore_top, sq_top, nsq_top, nbore_top]  # top face, facing +z
        idx += [sq_bot, nsq_bot, nsq_top, sq_top]  # outer wall, facing outward
        idx += [bore_bot, nbore_bot, nbore_top, bore_top][::-1]  # bore wall, facing the hole
        counts += [4, 4, 4]
    return pts, counts, idx


def rust_pattern(points, seed: int, coverage: float, cell: float = 0.004):
    """Patchy rust painted per vertex: (colours, metallic, roughness) lists, one per point.

    NVIDIA's RustedMetal material rendered as a flat dark brown on the fastener meshes —
    they have no texture coordinates, so the texture collapsed to one texel and the "rusty"
    nut looked like a normal black-oxide finish (an ambiguous label). This paints rust
    directly instead: smooth 3-D value noise with `cell`-metre features picks the
    `coverage` fraction of the surface that is rust (orange-brown, rough, not metallic);
    the rest stays steel. Deterministic in `seed`.
    """
    import math
    import random

    def h(ix, iy, iz):
        return random.Random((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)).random()

    def smooth(t):
        return t * t * (3 - 2 * t)

    def noise(x, y, z):
        x, y, z = x / cell, y / cell, z / cell
        ix, iy, iz = math.floor(x), math.floor(y), math.floor(z)
        fx, fy, fz = smooth(x - ix), smooth(y - iy), smooth(z - iz)
        v = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                    v += w * h(ix + dx, iy + dy, iz + dz)
        return v

    vals = [noise(*p) + 0.5 * noise(p[0] * 2.3 + 11, p[1] * 2.3 + 7, p[2] * 2.3 + 3) for p in points]
    order = sorted(vals)
    thresh = order[max(0, min(len(order) - 1, int(len(order) * (1 - coverage))))]
    rng = random.Random(seed * 31 + 7)
    rust_a, rust_b = (0.42, 0.17, 0.06), (0.62, 0.30, 0.10)  # dark and light rust
    colours, metal, rough = [], [], []
    for v in vals:
        if v >= thresh:
            t = rng.random()
            colours.append(tuple(rust_a[i] + (rust_b[i] - rust_a[i]) * t for i in range(3)))
            metal.append(0.05)
            rough.append(0.9)
        else:
            colours.append((0.72, 0.72, 0.74))
            metal.append(1.0)
            rough.append(0.25)
    return colours, metal, rough
