"""Tray catalogue for the sorting cell: a seed in, six parts and their correct handling out.

Pure Python, like faults.py, and for the same reasons: the simulator and the evaluation
tools both import it, it is unit-tested without booting Isaac Sim, and everything about a
tray is derived from its seed.

The sorting line (isaac/run_sorting.py) takes each part from the feed tray to the
inspection nest (the carrier the verifier was trained on, so the top photo matches its
training view), lifts it over an upward-looking camera, and then places it on the good-parts
rack or drops it in the reject bin.

What is on a tray:
  3 good parts      a clean M16 nut
  3 bad parts       rotated through BAD_KINDS: a wrong part (M12, M20, M16 bolt) or a damaged
                    M16 nut, with the damage on the TOP (seen by the inspection camera) or the
                    UNDERSIDE (seen only by the upward camera). Every 3 consecutive trays cover
                    all 9 kinds once.

Top damage reuses the single-scene rules and ranges (faults.py, partsets.FASTENER): chips
on the two top corners facing the inspection camera, cracks on the flat facing it, rust
on the surface. Underside damage sits on the bottom face, which the upward camera sees
straight on, so any of the six bottom corners can be chipped.

Seed ranges are disjoint from every single-scene split (1000+, 2000-2059, 5000-5119,
10000-11799), so no tray reuses a training or test scene.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import partsets

SLOTS = 6
GOOD_PER_TRAY = 3
BAD_KINDS = (
    "wrong:nut_m12", "wrong:nut_m20", "wrong:bolt_m16",
    "chip:top", "crack:top", "rust:top",
    "chip:underside", "crack:underside", "rust:underside",
)
SPLITS = {"dev": range(20000, 20030), "eval": range(21000, 21040)}
UNDERSIDE_CHIP_CORNERS = (0, 1, 2, 3, 4, 5)
RUST_COVERAGE = (0.25, 0.5)  # same range as faults.py's F5 rust


@dataclass(frozen=True)
class TrayPart:
    slot: int
    part: str              # partsets name: nut_m16, nut_m12, nut_m20, bolt_m16
    kind: str              # good | wrong | defective
    defect: str | None     # chip | crack | rust
    side: str | None       # top | underside (where the damage is)
    chip_mm: float | None
    chip_corner: int | None
    crack_mm: float | None
    crack_angle_deg: float | None
    rust_coverage: float | None
    feed_yaw_deg: float    # how the part lies on the feed tray

    @property
    def good(self) -> bool:
        return self.kind == "good"

    def expected(self, view: str) -> str:
        """The failure_type a perfect verifier reports from `view` (top | underside).
        A wrong part looks wrong from both sides; damage is only visible from its own side."""
        if self.kind == "wrong":
            return "wrong_part"
        if self.kind == "defective" and self.side == view:
            return "defective_part"
        return "none"

    @property
    def correct_action(self) -> str:
        return "accept" if self.good else "reject"


@dataclass(frozen=True)
class TraySpec:
    seed: int
    parts: tuple[TrayPart, ...]

    def to_json(self) -> dict:
        return {"seed": self.seed,
                "parts": [asdict(p) | {"good": p.good, "correct_action": p.correct_action,
                                       "expected_top": p.expected("top"),
                                       "expected_underside": p.expected("underside")}
                          for p in self.parts]}


def _bad_part(kind: str, slot: int, rng: random.Random, ps: partsets.PartSet) -> TrayPart:
    what, detail = kind.split(":")
    base = dict(slot=slot, chip_mm=None, chip_corner=None, crack_mm=None, crack_angle_deg=None,
                rust_coverage=None, feed_yaw_deg=round(rng.uniform(0.0, 60.0), 3))
    if what == "wrong":
        return TrayPart(part=detail, kind="wrong", defect=None, side=None, **base)
    side = detail
    if what == "chip":
        corners = ps.chip_corners if side == "top" else UNDERSIDE_CHIP_CORNERS
        base.update(chip_mm=round(rng.uniform(*ps.chip_mm), 2), chip_corner=rng.choice(corners))
    elif what == "crack":
        base.update(crack_mm=round(rng.uniform(*ps.crack_mm), 2),
                    crack_angle_deg=round(rng.uniform(0.0, 180.0), 2))
    else:
        base.update(rust_coverage=round(rng.uniform(*RUST_COVERAGE), 3))
    return TrayPart(part=ps.correct.name, kind="defective", defect=what, side=side, **base)


def make_tray(seed: int) -> TraySpec:
    """Derive the full tray for `seed`. Deterministic and side-effect free."""
    ps = partsets.FASTENER
    rng = random.Random(f"tray-{seed}")  # str seeds are stable across runs and machines
    kinds = ["good"] * GOOD_PER_TRAY + [BAD_KINDS[(seed * (SLOTS - GOOD_PER_TRAY) + i) % len(BAD_KINDS)]
                                        for i in range(SLOTS - GOOD_PER_TRAY)]
    rng.shuffle(kinds)
    parts = []
    for slot, kind in enumerate(kinds):
        if kind == "good":
            parts.append(TrayPart(slot=slot, part=ps.correct.name, kind="good", defect=None, side=None,
                                  chip_mm=None, chip_corner=None, crack_mm=None, crack_angle_deg=None,
                                  rust_coverage=None, feed_yaw_deg=round(rng.uniform(0.0, 60.0), 3)))
        else:
            parts.append(_bad_part(kind, slot, rng, ps))
    return TraySpec(seed=seed, parts=tuple(parts))
