"""Part sets: everything that differs between staging cubes and staging real fasteners.

The cell, the fault logic and the evaluation are shared. A part set says which part is
correct, which count as wrong, which defects exist, and how big the fixture pocket, the
printed nest, the gripper and the close-up framing must be for that part.

  cube      the original: a 50 mm cube; wrong = cylinder / 35 mm cube / tall block;
            defects = chip, crack. Unchanged, so every cube result stays reproducible.
  fastener  NVIDIA's own threaded parts (Isaac Sim asset server): an M16 hex nut is
            correct; an M16 bolt, an M12 nut and an M20 nut are wrong; defects are a
            chipped hex corner, a crack across a flat, and rust.

Pure Python: imported by faults.py, layout.py (tests run without Isaac Sim) and cell.py.
Fastener meshes are referenced by URL from NVIDIA's server, never copied into the repo:
the Omniverse licence allows publishing renders of them, not redistributing them.
"""
from __future__ import annotations

from dataclasses import dataclass

ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

# Workshop HDR environments (same asset server). Metal only looks like metal with real
# surroundings to reflect. Kept here, not in cell.py, so the randomisation that picks one
# can be unit-tested without Isaac Sim.
WORKSHOP_HDRS = tuple(f"{ASSET_ROOT}/NVIDIA/Assets/Skies/Indoor/{n}" for n in (
    "autoshop_01_4k.hdr", "ZetoCG_com_WarehouseInterior2b.hdr", "old_bus_depot_4k.hdr",
    "carpentry_shop_01_4k.hdr"))


@dataclass(frozen=True)
class PartDef:
    name: str
    shape: str  # box | hex | cylinder: how placement samples its surface
    dims: tuple[float, float, float]  # bounding box x, y, z in metres, centred
    # Where the geometry comes from: "procedural" (built in cell.py) or a USD reference.
    usd_url: str | None = None
    usd_prim: str | None = None
    # Offset from the referenced mesh's own origin to its bounding-box centre, measured
    # by isaac/probe_parts.py; cell.py subtracts it so every part is centred like a cube.
    usd_center: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _factory(name: str) -> tuple[str, str]:
    """Props/Factory keeps a physics-free visual mesh for each part. The full assets are
    physics bodies, and a nut (a free-floating articulation) isn't drawn until physics
    runs, which staged renders never do."""
    return (f"{ASSET_ROOT}/Isaac/Props/Factory/{name}/instanceable_meshes.usd", f"/{name}/{name}/visuals")


@dataclass(frozen=True)
class PartSet:
    name: str
    correct: PartDef
    wrong: tuple[PartDef, ...]  # rotated through by faults.py, in this order
    defects: tuple[str, ...]  # rotated through by faults.py, in this order
    # Fixture pocket
    pocket_shape: str  # square | round
    pocket_size: float  # square: side; round: diameter
    fixture_outer: float
    fixture_base_top: float
    fixture_wall_top: float
    # Printed nest outline on the carrier
    nest_inner: float
    # Gripper finger gaps
    gap_open: float
    gap_holding: float  # what the fingers read when they hold the correct part
    gap_empty: float
    gripper_scale: float  # stand-in gripper geometry scale (1.0 = sized for the 50 mm cube)
    # Close-up camera offset from the region of interest, for inspecting the part
    # (before grasp, after place)...
    closeup_offset: tuple[float, float, float]
    material: str = "paint"  # paint (procedural colour) | steel (MDL metal)
    # Fault parameter ranges that must scale with the part. Defaults are the cube's.
    f1_offset_mm: tuple[float, float] = (15.0, 40.0)
    chip_mm: tuple[float, float] = (12.0, 20.0)
    crack_mm: tuple[float, float] = (22.0, 40.0)
    # Top corners whose chip faces the inspection camera (numbering per part shape).
    chip_corners: tuple[int, ...] = (1, 2)
    # How a held part is turned in the gripper (degrees about z). A hex nut is gripped
    # across its FLATS: with its corners on the y axis, that needs a 30 degree turn, or the
    # 24 mm finger gap lands on the 27.7 mm corners and cuts 1.85 mm into the nut.
    held_yaw_deg: float = 0.0
    # How far a correctly seated part may sit off centre: less than its clearance in the
    # pocket, or a "clean" seat would rest on the rim.
    seated_jitter_mm: float = 1.5
    # ...and for watching the gripper (after lift, after transport), which needs room for
    # the fingers AND whatever is, or isn't, below them. None = same as closeup_offset.
    closeup_offset_handling: tuple[float, float, float] | None = None
    # Height above the carrier/fixture surface the inspection camera aims at.
    inspect_aim_dz: float = 0.02


CUBE = PartSet(
    name="cube",
    correct=PartDef("cube_50", "box", (0.05, 0.05, 0.05)),
    wrong=(PartDef("cylinder", "cylinder", (0.05, 0.05, 0.05)),
           PartDef("small_cube", "box", (0.035, 0.035, 0.035)),
           PartDef("tall_block", "box", (0.05, 0.05, 0.075))),
    defects=("chip", "crack"),
    pocket_shape="square", pocket_size=0.054, fixture_outer=0.13,
    fixture_base_top=0.012, fixture_wall_top=0.034,
    nest_inner=0.056,
    gap_open=0.090, gap_holding=0.05, gap_empty=0.008, gripper_scale=1.0,
    closeup_offset=(0.34, -0.12, 0.28),
)

_nut16_url, _nut16_prim = _factory("factory_nut_m16_loose")
_nut12_url, _nut12_prim = _factory("factory_nut_m12_loose")
_nut20_url, _nut20_prim = _factory("factory_nut_m20_loose")
_bolt16_url, _bolt16_prim = _factory("factory_bolt_m16_loose")

FASTENER = PartSet(
    name="fastener",
    # Sizes and origin offsets measured by isaac/probe_parts.py (Oct 12): x across the
    # flats, y across the corners. This folder's M16 bolt has a round socket head.
    correct=PartDef("nut_m16", "hex", (0.024, 0.0277, 0.013), _nut16_url, _nut16_prim, (0.0, 0.0, 0.0225)),
    wrong=(PartDef("bolt_m16", "cylinder", (0.024, 0.024, 0.041), _bolt16_url, _bolt16_prim, (0.0, 0.0, 0.0205)),
           PartDef("nut_m12", "hex", (0.019, 0.0219, 0.010), _nut12_url, _nut12_prim, (0.0, 0.0, 0.017)),
           PartDef("nut_m20", "hex", (0.030, 0.0346, 0.016), _nut20_url, _nut20_prim, (0.0, 0.0, 0.028))),
    defects=("chip", "crack", "rust"),
    pocket_shape="round", pocket_size=0.030, fixture_outer=0.08,
    fixture_base_top=0.012, fixture_wall_top=0.020,
    nest_inner=0.032,
    gap_open=0.045, gap_holding=0.024, gap_empty=0.004, gripper_scale=0.5,
    # Close enough that a 4 mm chip is ~35 px, far enough that every scene's part stays in
    # frame. Searched over 1,375 pre-grasp scenes with camera jitter (tests/test_fastener.py):
    # the first, tighter framing cropped the 41 mm bolt's top in 88 scenes and displaced
    # nuts in 7; aiming 14 mm up at 1.3x this distance keeps all of them inside (worst 0.96).
    closeup_offset=(0.1105, -0.039, 0.091),
    inspect_aim_dz=0.014,
    # A 27.7 mm across-corners nut in a 30 mm counterbore has 1.15 mm of clearance.
    seated_jitter_mm=0.8,
    closeup_offset_handling=(0.17, -0.06, 0.14),
    material="steel",
    # The close-up is about 4x tighter than the cube's: a 15-40 mm shift pushed the nut
    # partly out of frame in 36% of displaced scenes (code review). 8-20 mm still puts a
    # 27.7 mm nut clearly over the 32 mm outline.
    f1_offset_mm=(8.0, 20.0),
    chip_mm=(3.0, 5.0),
    crack_mm=(6.0, 10.0),  # a 13 mm-tall flat holds about 10.4 mm at steep angles
    # Corners 4 (330 deg) and 5 (30 deg) face the camera (~340 deg). Corner 3 (270 deg)
    # was visible but foreshortened (facet facing 0.33-0.54 toward the camera, vs 0.86).
    chip_corners=(4, 5),
    held_yaw_deg=30.0,
)

PART_SETS = {"cube": CUBE, "fastener": FASTENER}


def get(name: str) -> PartSet:
    try:
        return PART_SETS[name]
    except KeyError:
        raise ValueError(f"unknown part set {name!r}; choose from {sorted(PART_SETS)}") from None
