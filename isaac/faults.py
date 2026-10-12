"""Fault catalogue: a seed in, a fully specified scene and its correct answer out.

Pure Python on purpose — no Isaac Sim imports. The stager (Isaac's Python 3.11) and the
evaluation tools (the project venv) both import this, and it can be unit-tested in a
second instead of after a two-minute simulator boot.

The one rule that matters: everything about a scene is derived from its seed. Same seed,
same scene, same label, on any machine. That is what makes the dataset reproducible and
lets Tier 1 and Tier 2 be scored on an identical held-out split. tests/test_faults.py
pins a fingerprint of the evaluation split, so an accidental change here can't silently
reshuffle it.

How a seed maps to a scene:
  fault     = FAULT_CLASSES[seed % 6]      every 6 consecutive seeds cover F0..F5 once
  sub-case  = chosen from k = seed // 6    so each fault's variants are balanced too
  numbers   = drawn from random.Random(seed)   offsets, angles, sizes

Units: metres and degrees. Positions are relative to the named anchor (carrier slot,
fixture pocket, ...); isaac/layout.py turns them into world coordinates.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

import partsets

CHECKPOINTS = ("CP_PRE_GRASP", "CP_POST_LIFT", "CP_POST_TRANSPORT", "CP_POST_PLACE")
FAULT_CLASSES = ("F0", "F1", "F2", "F3", "F4", "F5")
WRONG_SHAPES = ("cylinder", "small_cube", "tall_block")
DEFECT_TYPES = ("chip", "crack")

# Per part set: wrong parts and defects come from partsets.py, rotated in its order.
# For cubes these are exactly WRONG_SHAPES and DEFECT_TYPES above.
#
# Chips go only on the two top corners that face the close-up inspection camera
# (it sits at +x, -y of the part). On the other two the chip facet is edge-on or faces
# away, so a single camera physically cannot see it; that would measure camera placement,
# not the model. A real station would add a second camera. Corner numbering is in
# geometry.chipped_cube_mesh: 1 = (+x, -y), 2 = (+x, +y).
CHIP_CORNERS = (1, 2)

# Where the part is expected to be when nothing went wrong, per checkpoint. This is also
# the location_hint a correct verdict should give for a clean (F0) scene.
NOMINAL_LOCATION = {
    "CP_PRE_GRASP": "carrier",
    "CP_POST_LIFT": "gripper",
    "CP_POST_TRANSPORT": "gripper",
    "CP_POST_PLACE": "fixture",
}

# Parameter ranges, from SPEC.md §7. Named so the tests can assert the generator never
# leaves them.
F1_OFFSET_MM = (15.0, 40.0)
F1_YAW_DEG = (15.0, 45.0)
F3_OFFSET_MM = (6.0, 15.0)
F3_TILT_DEG = (5.0, 15.0)
F5_CHIP_MM = (12.0, 20.0)
F5_CRACK_MM = (22.0, 40.0)
# On a fastener the defects are smaller, because the part is: a chip is cut 3-5 mm into a
# hex corner, a crack runs 6-11 mm across a 13 mm flat.
FASTENER_CHIP_MM = (3.0, 5.0)
FASTENER_CRACK_MM = (6.0, 11.0)
# Hex corners of the nut's top face that face the close-up camera (it sits at +x, -y).
# Numbered counter-clockwise from +y in geometry.hex_corner_cut.
FASTENER_CHIP_CORNERS = (3, 4)
F2_TABLE_PER_10 = 7  # 7 of every 10 drops land on the table (recoverable), 3 off it

# Small, deliberate nuisance variation on clean scenes. A real carrier never presents a
# part at exactly 0.0 mm; without this the model could learn "any offset at all = fault".
NOMINAL_JITTER_MM = 3.0
NOMINAL_YAW_JITTER_DEG = 4.0
# A seated part has far less room: 50 mm in a 54 mm pocket. More than about 1 degree of
# yaw plus the offset would push a corner into the wall.
SEATED_JITTER_MM = 1.5
SEATED_YAW_JITTER_DEG = 1.0


@dataclass(frozen=True)
class Expected:
    """The verdict a perfect verifier would return. Mirrors the Verdict schema fields
    that are scored; `evidence` and `confidence` are free-form and not scored."""

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
    wrong_shape: str | None  # cylinder | small_cube | tall_block (F4 only)
    defect_type: str | None  # chip | crack (F5 only)
    chip_mm: float | None  # chip edge length along each of the corner's 3 edges
    chip_corner: int | None  # top corner, see CHIP_CORNERS
    crack_mm: float | None  # crack length across the top face
    crack_angle_deg: float | None  # crack direction on the top face
    part_location: str  # carrier | gripper | fixture | table | floor
    part_offset_m: tuple[float, float]  # xy offset from the anchor's nominal spot
    part_yaw_deg: float
    part_tilt_deg: float  # seat tilt magnitude; layout.py tilts toward the offset wall

    # F2 drops: where in the landing region, as fractions of its width/depth.
    landing_frac: tuple[float, float] | None
    floor_side: int | None  # +1 / -1: which long side of the table it fell off

    # The gripper (a two-finger stand-in)
    gripper_at: str  # above_carrier | above_fixture
    gripper_height_m: float  # height of the grasp point (finger-pad centre) above the anchor
    gripper_closed: bool
    gripper_holding: bool

    expected: Expected
    truth: dict = field(default_factory=dict)
    part_set: str = "cube"

    def to_json(self) -> dict:
        d = asdict(self)
        # Cube scenes serialise exactly as they did before part sets existed, so the
        # pinned eval-split fingerprint (and every cube result) stays valid.
        if d["part_set"] == "cube":
            del d["part_set"]
        return d


def fault_for_seed(seed: int) -> str:
    return FAULT_CLASSES[seed % len(FAULT_CLASSES)]


def _offset_xy(rng: random.Random, magnitude_mm: float) -> tuple[float, float]:
    """A planar offset of exactly `magnitude_mm`, in a random direction, in metres."""
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = magnitude_mm / 1000.0
    return (round(r * math.cos(theta), 6), round(r * math.sin(theta), 6))


def make_scene(seed: int, part_set: str = "cube") -> SceneSpec:
    """Derive the complete scene for `seed`. Deterministic and side-effect free.

    The part set changes WHAT the parts are (and the size of their defects), never which
    fault or sub-case a seed gets: seed 5003 is F5 in both sets."""
    ps = partsets.get(part_set)
    rng = random.Random(seed)
    fault = fault_for_seed(seed)
    k = seed // len(FAULT_CLASSES)  # sub-case selector, balanced within each fault

    f = dict(
        part_kind="correct", wrong_shape=None, defect_type=None, chip_mm=None,
        chip_corner=None, crack_mm=None, crack_angle_deg=None,
        part_location="carrier",
        part_offset_m=_offset_xy(rng, rng.uniform(0.0, NOMINAL_JITTER_MM)),
        part_yaw_deg=rng.uniform(-NOMINAL_YAW_JITTER_DEG, NOMINAL_YAW_JITTER_DEG),
        part_tilt_deg=0.0, landing_frac=None, floor_side=None,
        gripper_at="above_carrier", gripper_height_m=0.15,
        gripper_closed=False, gripper_holding=False,
    )
    truth: dict = {}

    if fault == "F0":
        # Clean run, cycling through all four checkpoints so "no fault" is never
        # confused with "this checkpoint".
        checkpoint = CHECKPOINTS[k % len(CHECKPOINTS)]
        if checkpoint == "CP_POST_LIFT":
            f.update(part_location="gripper", gripper_height_m=0.12,
                     gripper_closed=True, gripper_holding=True)
        elif checkpoint == "CP_POST_TRANSPORT":
            f.update(part_location="gripper", gripper_at="above_fixture",
                     gripper_height_m=0.12, gripper_closed=True, gripper_holding=True)
        elif checkpoint == "CP_POST_PLACE":
            f.update(part_location="fixture", gripper_at="above_fixture",
                     part_offset_m=_offset_xy(rng, rng.uniform(0.0, ps.seated_jitter_mm)),
                     part_yaw_deg=rng.uniform(-SEATED_YAW_JITTER_DEG, SEATED_YAW_JITTER_DEG))
        expected = Expected(checkpoint, True, "none", NOMINAL_LOCATION[checkpoint])

    elif fault == "F1":
        # Part shifted on the carrier before the grasp. Alternates between catching it
        # before the grasp (visible displacement) and its consequence one step later: the
        # gripper closed on air and lifted, part still on the carrier.
        offset_mm = rng.uniform(*F1_OFFSET_MM)
        yaw = rng.uniform(*F1_YAW_DEG) * rng.choice((-1.0, 1.0))
        f.update(part_offset_m=_offset_xy(rng, offset_mm), part_yaw_deg=yaw)
        truth = {"offset_mm": round(offset_mm, 2), "yaw_deg": round(yaw, 2)}
        if k % 2 == 0:
            checkpoint = "CP_PRE_GRASP"
            expected = Expected(checkpoint, False, "part_displaced", "carrier")
        else:
            checkpoint = "CP_POST_LIFT"
            f.update(gripper_height_m=0.12, gripper_closed=True, gripper_holding=False)
            expected = Expected(checkpoint, False, "empty_grasp", "carrier")

    elif fault == "F2":
        # Slip in transport: gripper arrives over the fixture closed on nothing.
        checkpoint = "CP_POST_TRANSPORT"
        on_table = (k % 10) < F2_TABLE_PER_10
        f.update(gripper_at="above_fixture", gripper_height_m=0.12, gripper_closed=True,
                 gripper_holding=False, part_offset_m=(0.0, 0.0),
                 part_yaw_deg=rng.uniform(0.0, 360.0),
                 landing_frac=(round(rng.random(), 6), round(rng.random(), 6)))
        if on_table:
            f["part_location"] = "table"
            expected = Expected(checkpoint, False, "part_dropped_recoverable", "table_in_reach")
        else:
            f.update(part_location="floor", floor_side=1 if (k // 10) % 2 == 0 else -1)
            expected = Expected(checkpoint, False, "part_dropped_unrecoverable", "off_table")
        truth = {"landing_region": f["part_location"]}

    elif fault == "F3":
        # Misaligned seat: released, but caught on the rim instead of dropping into the
        # pocket. layout.py tilts it up on the side it is offset toward and rests it on
        # the rim and floor by contact, so it never intersects the fixture.
        checkpoint = "CP_POST_PLACE"
        offset_mm = rng.uniform(*F3_OFFSET_MM)
        tilt = rng.uniform(*F3_TILT_DEG)
        f.update(part_location="fixture", gripper_at="above_fixture",
                 part_offset_m=_offset_xy(rng, offset_mm), part_tilt_deg=tilt,
                 part_yaw_deg=rng.uniform(-SEATED_YAW_JITTER_DEG, SEATED_YAW_JITTER_DEG))
        truth = {"offset_mm": round(offset_mm, 2), "tilt_deg": round(tilt, 2)}
        expected = Expected(checkpoint, False, "misaligned_seat", "fixture")

    elif fault == "F4":
        checkpoint = "CP_PRE_GRASP"
        shape = ps.wrong[k % len(ps.wrong)].name
        f.update(part_kind="wrong", wrong_shape=shape)
        truth = {"wrong_shape": shape}
        expected = Expected(checkpoint, False, "wrong_part", "carrier")

    else:  # F5
        checkpoint = "CP_PRE_GRASP"
        defect = ps.defects[k % len(ps.defects)]
        f.update(part_kind="defective", defect_type=defect)
        cube = ps.name == "cube"
        if defect == "chip":
            corners = CHIP_CORNERS if cube else FASTENER_CHIP_CORNERS
            corner = corners[(k // len(ps.defects)) % len(corners)]
            chip = round(rng.uniform(*(F5_CHIP_MM if cube else FASTENER_CHIP_MM)), 2)
            f.update(chip_mm=chip, chip_corner=corner)
            truth = {"defect": "chip", "chip_mm": chip, "chip_corner": corner}
        elif defect == "crack":
            length = round(rng.uniform(*(F5_CRACK_MM if cube else FASTENER_CRACK_MM)), 2)
            angle = round(rng.uniform(0.0, 180.0), 2)
            f.update(crack_mm=length, crack_angle_deg=angle)
            truth = {"defect": "crack", "crack_mm": length}
        else:  # rust: the whole part's finish, no geometry change
            truth = {"defect": "rust"}
        expected = Expected(checkpoint, False, "defective_part", "carrier")

    f["part_yaw_deg"] = round(f["part_yaw_deg"], 3)
    f["part_tilt_deg"] = round(f["part_tilt_deg"], 3)
    return SceneSpec(seed=seed, fault=fault, checkpoint=checkpoint, expected=expected,
                     truth=truth, part_set=ps.name, **f)
