"""Fault catalogue: a seed in, a fully specified scene and its correct answer out.

Pure Python on purpose — no Isaac Sim imports. The stager (Isaac's Python 3.11) and the
evaluation tools (the project venv) both import this, and it can be unit-tested in a
second instead of after a two-minute simulator boot.

The one rule that matters: everything about a scene is derived from its seed. Same seed,
same scene, same label, on any machine. That is what makes the dataset reproducible and
lets Tier 1 and Tier 2 be scored on an identical held-out split.

Units: metres and degrees. Positions are relative to the named anchor (carrier slot,
fixture pocket, ...); the scene builder owns absolute world coordinates.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

CHECKPOINTS = ("CP_PRE_GRASP", "CP_POST_LIFT", "CP_POST_TRANSPORT", "CP_POST_PLACE")
FAULT_CLASSES = ("F0", "F1", "F2", "F3", "F4", "F5")

# Where the part is expected to be when nothing went wrong, per checkpoint. This is also
# the location_hint a correct verdict should give for a clean (F0) scene.
NOMINAL_LOCATION = {
    "CP_PRE_GRASP": "carrier",
    "CP_POST_LIFT": "gripper",
    "CP_POST_TRANSPORT": "gripper",
    "CP_POST_PLACE": "fixture",
}

# Parameter ranges, straight from SPEC.md §7. Kept here as named constants so the tests
# can assert the generator never leaves them.
F1_OFFSET_MM = (15.0, 40.0)
F1_YAW_DEG = (15.0, 45.0)
F3_OFFSET_MM = (6.0, 15.0)
F3_TILT_DEG = (5.0, 15.0)
F5_CHIP_MM = (12.0, 20.0)
F2_RECOVERABLE_SHARE = 0.7

# Small, deliberate nuisance variation on clean scenes. A real carrier never presents a
# part at exactly 0.0 mm; without this the model could learn "any offset at all = fault".
NOMINAL_JITTER_MM = 3.0
NOMINAL_YAW_JITTER_DEG = 4.0


@dataclass(frozen=True)
class Expected:
    """The verdict a perfect verifier would return. Mirrors the Verdict schema fields
    that can be scored; `evidence` and `confidence` are free-form and not scored."""

    checkpoint: str
    step_ok: bool
    failure_type: str
    location_hint: str


@dataclass(frozen=True)
class SceneSpec:
    seed: int
    fault: str
    checkpoint: str

    # The part
    part_kind: str  # correct | wrong | defective
    wrong_shape: str | None  # cylinder | small_cube | tall_block (wrong parts only)
    chip_mm: float | None  # chip edge length (defective parts only)
    chip_corner: int | None  # which top corner is chipped, 0-3
    part_location: str  # carrier | gripper | fixture | table | floor
    part_offset_m: tuple[float, float]  # xy offset from the location's nominal slot
    part_yaw_deg: float
    part_tilt_deg: float  # rotation about the part's local x axis (seat misalignment)

    # The gripper (a two-finger proxy, see SPEC.md)
    gripper_at: str  # above_carrier | above_fixture
    gripper_height_m: float  # fingertip-centre height above the anchor surface
    gripper_closed: bool
    gripper_holding: bool

    expected: Expected
    truth: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def fault_for_seed(seed: int) -> str:
    """Balanced by construction: every run of 6 consecutive seeds covers F0..F5 once."""
    return FAULT_CLASSES[seed % len(FAULT_CLASSES)]


def _signed(rng: random.Random, lo: float, hi: float) -> float:
    """A magnitude in [lo, hi] with a random sign."""
    return rng.uniform(lo, hi) * rng.choice((-1.0, 1.0))


def _offset_mm_to_xy_m(rng: random.Random, magnitude_mm: float) -> tuple[float, float]:
    """A planar offset of exactly `magnitude_mm`, in a random direction."""
    import math

    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = magnitude_mm / 1000.0
    return (round(r * math.cos(theta), 5), round(r * math.sin(theta), 5))


def _jitter(rng: random.Random) -> tuple[float, float]:
    return _offset_mm_to_xy_m(rng, rng.uniform(0.0, NOMINAL_JITTER_MM))


def make_scene(seed: int) -> SceneSpec:
    """Derive the complete scene for `seed`. Deterministic and side-effect free."""
    rng = random.Random(seed)
    fault = fault_for_seed(seed)

    # Defaults describe a clean part waiting on the carrier, gripper hovering above it.
    part_kind, wrong_shape, chip_mm, chip_corner = "correct", None, None, None
    part_location = "carrier"
    part_offset = _jitter(rng)
    part_yaw = rng.uniform(-NOMINAL_YAW_JITTER_DEG, NOMINAL_YAW_JITTER_DEG)
    part_tilt = 0.0
    gripper_at, gripper_height, gripper_closed, gripper_holding = (
        "above_carrier", 0.15, False, False,
    )
    truth: dict = {}

    if fault == "F0":
        # Clean run. Cycle through all four checkpoints so "no fault" is never confused
        # with "this checkpoint".
        checkpoint = CHECKPOINTS[(seed // len(FAULT_CLASSES)) % len(CHECKPOINTS)]
        if checkpoint == "CP_POST_LIFT":
            part_location = "gripper"
            gripper_height, gripper_closed, gripper_holding = 0.12, True, True
        elif checkpoint == "CP_POST_TRANSPORT":
            part_location = "gripper"
            gripper_at, gripper_height, gripper_closed, gripper_holding = (
                "above_fixture", 0.12, True, True,
            )
        elif checkpoint == "CP_POST_PLACE":
            part_location = "fixture"
            part_offset = _offset_mm_to_xy_m(rng, rng.uniform(0.0, 1.5))
            gripper_at, gripper_height = "above_fixture", 0.15
        expected = Expected(checkpoint, True, "none", NOMINAL_LOCATION[checkpoint])

    elif fault == "F1":
        # Part shifted on the carrier before the grasp. Half the scenes are caught before
        # grasping (visible displacement); half are the consequence one step later — the
        # gripper closed on air and lifted, part still sitting on the carrier.
        offset_mm = rng.uniform(*F1_OFFSET_MM)
        part_offset = _offset_mm_to_xy_m(rng, offset_mm)
        part_yaw = _signed(rng, *F1_YAW_DEG)
        truth = {"offset_mm": round(offset_mm, 2), "yaw_deg": round(part_yaw, 2)}
        if rng.random() < 0.5:
            checkpoint = "CP_PRE_GRASP"
            expected = Expected(checkpoint, False, "part_displaced", "carrier")
        else:
            checkpoint = "CP_POST_LIFT"
            gripper_height, gripper_closed, gripper_holding = 0.12, True, False
            expected = Expected(checkpoint, False, "empty_grasp", "carrier")

    elif fault == "F2":
        # Slip in transport: gripper arrives over the fixture closed on nothing; the part
        # is lying somewhere it fell.
        checkpoint = "CP_POST_TRANSPORT"
        gripper_at, gripper_height, gripper_closed, gripper_holding = (
            "above_fixture", 0.12, True, False,
        )
        part_yaw = rng.uniform(0.0, 360.0)
        if rng.random() < F2_RECOVERABLE_SHARE:
            part_location = "table"
            expected = Expected(checkpoint, False, "part_dropped_recoverable", "table_in_reach")
        else:
            part_location = "floor"
            expected = Expected(checkpoint, False, "part_dropped_unrecoverable", "off_table")
        # The builder picks a concrete landing spot inside the named region from this
        # pair of unit-interval draws; keeps region geometry out of this module.
        part_offset = (round(rng.random(), 5), round(rng.random(), 5))
        truth = {"landing_region": part_location}

    elif fault == "F3":
        # Misaligned seat: released, but sitting proud on the rim instead of in the pocket.
        checkpoint = "CP_POST_PLACE"
        part_location = "fixture"
        offset_mm = rng.uniform(*F3_OFFSET_MM)
        part_offset = _offset_mm_to_xy_m(rng, offset_mm)
        part_tilt = _signed(rng, *F3_TILT_DEG)
        gripper_at, gripper_height = "above_fixture", 0.15
        truth = {"offset_mm": round(offset_mm, 2), "tilt_deg": round(part_tilt, 2)}
        expected = Expected(checkpoint, False, "misaligned_seat", "fixture")

    elif fault == "F4":
        checkpoint = "CP_PRE_GRASP"
        part_kind = "wrong"
        wrong_shape = rng.choice(("cylinder", "small_cube", "tall_block"))
        truth = {"wrong_shape": wrong_shape}
        expected = Expected(checkpoint, False, "wrong_part", "carrier")

    else:  # F5
        checkpoint = "CP_PRE_GRASP"
        part_kind = "defective"
        chip_mm = round(rng.uniform(*F5_CHIP_MM), 2)
        chip_corner = rng.randrange(4)
        truth = {"chip_mm": chip_mm, "chip_corner": chip_corner}
        expected = Expected(checkpoint, False, "defective_part", "carrier")

    return SceneSpec(
        seed=seed,
        fault=fault,
        checkpoint=checkpoint,
        part_kind=part_kind,
        wrong_shape=wrong_shape,
        chip_mm=chip_mm,
        chip_corner=chip_corner,
        part_location=part_location,
        part_offset_m=part_offset,
        part_yaw_deg=round(part_yaw, 3),
        part_tilt_deg=round(part_tilt, 3),
        gripper_at=gripper_at,
        gripper_height_m=gripper_height,
        gripper_closed=gripper_closed,
        gripper_holding=gripper_holding,
        expected=expected,
        truth=truth,
    )
