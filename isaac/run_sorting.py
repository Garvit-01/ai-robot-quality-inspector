#!/usr/bin/env python3
"""The sorting line: a UR10 sorts a tray of six nuts using the fine-tuned verifier.

Per part (isaac/trays.py says what each one really is):
  1. pick it from the feed tray, set it on the inspection nest (the carrier the verifier was
     trained on), park the arm out of view;
  2. top photos (overview + close-up) — the same views as the training data;
  3. pick it again and hold it over an upward-looking camera: underside photo;
  4. verifier: the trained pre-grasp question on the top photos, a damage question on the
     underside photo; fused into accept / reject (any "wrong" or "damaged" = reject);
  5. accept → good-parts rack, reject → reject bin.
Afterwards every part's final position is read back from physics, so a part the arm dropped
on the way is counted as a handling failure, not silently as "sorted".

The model runs in its own process (finetune/serve_verifier.py) and is called over HTTP on
localhost; the simulation waits for the answer. `--verifier oracle` uses the ground truth
instead, to test the arm on its own.

Arm moves are closed-loop on position: a waypoint is reached when the end effector is within
tolerance, not after a fixed time (NVIDIA's PickPlaceController is time-based, which hid the
first probe's frozen arm).

Usage (Isaac Sim's Python):
    ~/isaacsim/python.sh isaac/run_sorting.py --trays 20000 --verifier oracle --out runs/sorting/test
    ~/isaacsim/python.sh isaac/run_sorting.py --trays dev --out runs/sorting/dev      # needs serve_verifier.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))  # selfheal.sorting: pure Python, shared with eval/

import faults  # noqa: E402
import layout as L  # noqa: E402
import partsets  # noqa: E402
import trays  # noqa: E402
from selfheal.sorting import decide  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trays", default="20000", help="comma-separated seeds, or a split name: dev | eval")
ap.add_argument("--out", type=Path, default=REPO / "runs" / "sorting" / "test")
ap.add_argument("--verifier", default="http://127.0.0.1:8765", help="service URL, or 'oracle'")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--move-steps", type=int, default=600, help="step budget per waypoint")
ap.add_argument("--debug", action="store_true", help="first part only; print every waypoint")
args, _ = ap.parse_known_args()

from isaacsim.simulation_app import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui, "renderer": "RayTracedLighting", "width": 1024, "height": 768})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot.manipulators import SingleManipulator  # noqa: E402
from isaacsim.robot.manipulators.examples.universal_robots.controllers.rmpflow_controller import (  # noqa: E402
    RMPFlowController)
from isaacsim.robot.manipulators.grippers import SurfaceGripper  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

from cell import Cell, _visible  # noqa: E402
from dr import make_dr  # noqa: E402
from geometry import (crack_ribbon, hex_corner_cut, hex_corner_cut_underside, hex_flat_frame,  # noqa: E402
                      mask_to_underside, place_on_face, rust_pattern, underside_crack)

PS = partsets.FASTENER
PARTS = {PS.correct.name: PS.correct, **{d.name: d for d in PS.wrong}}
CFG = yaml.safe_load((REPO / "config" / "selfheal_fastener.yaml").read_text())
ROBOT_USD = f"{partsets.ASSET_ROOT}/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

# --- layout of the line (metres, table top z = 0, robot base at the origin) ---------------
NEST = (L.CARRIER_SLOT[0], L.CARRIER_SLOT[1])
# Feed tray: clear of the carrier, and far enough from the robot's base for RMPflow; at
# x = 0.17-0.27 the arm stalled ~13 cm short of the slots and its descent swept the nuts away.
FEED_SLOTS = [(x, y) for y in (-0.42, -0.36) for x in (0.30, 0.35, 0.40)]   # feed tray, 3 x 2
RACK_SLOTS = [(x, y) for y in (0.25, 0.31) for x in (0.35, 0.40, 0.45)]     # good parts, 3 x 2
PLATE_TOP = 0.006
UNDER_CAM = (0.30, 0.02, 0.004)        # upward-looking camera on the table
HOVER_Z = 0.13                          # part bottom this high above the table when photographed
PARK = np.array([0.10, 0.30, 0.45])     # arm out of the inspection views
EE_DOWN = euler_angles_to_quat(np.array([0, np.pi / 2.0, 0]))  # suction cup pointing down
# Suction-cup tip this far above the part's TOP face when gripping. Measured from the top, not
# the centre: the 10 mm M12 nut was missed when the tip sat 20 mm above its centre (15 mm gap).
GRIP_GAP = 0.007
SAFE_Z = 0.20
# Long moves go through a point above the middle of the table: straight from PARK (+y) to the
# feed tray or nest (-y) RMPflow stalled ~15 cm short (a local minimum swinging the base round).
VIA = np.array([0.40, 0.0, 0.32])

UNDERSIDE_QUESTION = """Checkpoint: CP_PRE_GRASP. The image is a close-up photo taken from BELOW of
the part the robot's suction gripper is holding: {part}. You see the part's underside against
the robot's blue suction cup. Decide only whether the part is wrong or physically damaged.
- A bolt, or a nut clearly smaller or larger than an M16 nut: wrong_part.
- A corner cut off, a dark crack line, or brown rust on the underside: defective_part.
- Otherwise: step_ok true, failure_type none.
location_hint is gripper in every case.

You are given 1 image(s): image 1 is a close-up of the part's underside.

"""


# --- scene ------------------------------------------------------------------------------
def collide(prim, approximation: str | None = None) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)
    if approximation and prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)


def box(stage, path, size, centre, rgb):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(*centre))
    UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    collide(cube.GetPrim())
    return cube


def first_mesh(root) -> UsdGeom.Mesh:
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(p)
    raise RuntimeError(f"no mesh under {root.GetPath()}")


class Part:
    """One physics part: rigid body, invisible cylinder collider, NVIDIA's visual mesh with
    this part's damage applied (top or underside), origin at the part's centre."""

    def __init__(self, stage, cell, tp: trays.TrayPart, xy):
        self.tp, self.pd = tp, PARTS[tp.part]
        ax, ay, az = self.pd.dims
        path = f"/World/Sort/part{tp.slot}"
        self.body = UsdGeom.Xform.Define(stage, path)
        UsdGeom.XformCommonAPI(self.body).SetTranslate(Gf.Vec3d(xy[0], xy[1], PLATE_TOP + az / 2 + 0.001))
        UsdGeom.XformCommonAPI(self.body).SetRotate(Gf.Vec3f(0, 0, tp.feed_yaw_deg))
        UsdPhysics.RigidBodyAPI.Apply(self.body.GetPrim())
        UsdPhysics.MassAPI.Apply(self.body.GetPrim()).CreateMassAttr(0.03 if self.pd.shape == "hex" else 0.07)
        col = UsdGeom.Cylinder.Define(stage, f"{path}/collider")
        col.CreateRadiusAttr((ay if self.pd.shape == "hex" else ax) / 2)
        col.CreateHeightAttr(az)
        col.CreateAxisAttr("Z")
        UsdGeom.Imageable(col).CreatePurposeAttr("guide")
        collide(col.GetPrim())
        asset = UsdGeom.Xform.Define(stage, f"{path}/asset")
        asset.GetPrim().GetReferences().AddReference(self.pd.usd_url, self.pd.usd_prim)
        c = self.pd.usd_center
        UsdGeom.XformCommonAPI(asset).SetTranslate(Gf.Vec3d(-c[0], -c[1], -c[2]))
        material = cell.m_rust if tp.defect == "rust" else cell.m_steel
        UsdShade.MaterialBindingAPI.Apply(asset.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        self.asset, self.stage, self.cell = asset, stage, cell
        self._damage()

    def _damage(self):
        tp, (ax, ay, az), c = self.tp, self.pd.dims, self.pd.usd_center
        if tp.defect is None:
            return
        if tp.defect in ("chip", "rust"):
            mesh = first_mesh(self.asset.GetPrim())
            pts = [tuple(v) for v in mesh.GetPointsAttr().Get()]
        if tp.defect == "chip":
            if tp.side == "top":
                new, _ = hex_corner_cut(pts, tp.chip_corner, ay, c[2] + az / 2, tp.chip_mm / 1000.0)
            else:
                new, _ = hex_corner_cut_underside(pts, tp.chip_corner, ay, c[2], az / 2, tp.chip_mm / 1000.0)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*q) for q in new]))
        elif tp.defect == "rust":
            cols, metal, rough = rust_pattern(pts, 7000 + tp.slot, tp.rust_coverage)
            if tp.side == "underside":
                cols, metal, rough = mask_to_underside(pts, cols, metal, rough, c[2])
            pv = UsdGeom.PrimvarsAPI(mesh)
            pv.CreatePrimvar("rustColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec3fArray([Gf.Vec3f(*q) for q in cols]))
            pv.CreatePrimvar("rustMetal", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(Vt.FloatArray(metal))
            pv.CreatePrimvar("rustRough", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(Vt.FloatArray(rough))
        else:  # crack: a dark ribbon, child of the body (centred frame)
            if tp.side == "top":  # on the flat facing the inspection camera, as in the dataset
                pts2d, counts, idx = crack_ribbon(tp.crack_mm / 1000.0, tp.crack_angle_deg, ay / 4, 7000 + tp.slot,
                                                  width=0.0004, segments=6, margin=0.0012, half_height=az / 2,
                                                  wobble=0.0005)
                centre, u, v, n = hex_flat_frame(300.0, ax, 0.0)
                pts = place_on_face(pts2d, centre, u, v)
            else:
                pts, counts, idx = underside_crack(tp.crack_mm / 1000.0, tp.crack_angle_deg, 7000 + tp.slot,
                                                   bottom_z=-az / 2, ring_radius=0.010)
            crack = UsdGeom.Mesh.Define(self.stage, f"{self.body.GetPath()}/crack")
            crack.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            crack.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*q) for q in pts]))
            crack.GetFaceVertexCountsAttr().Set(counts)
            crack.GetFaceVertexIndicesAttr().Set(idx)
            UsdShade.MaterialBindingAPI.Apply(crack.GetPrim()).Bind(self.cell.m_crack,
                                                                     UsdShade.Tokens.strongerThanDescendants)

    def pos(self) -> np.ndarray:
        t = UsdGeom.Xformable(self.body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return np.array([t[0], t[1], t[2]])

    def remove(self):
        self.stage.RemovePrim(self.body.GetPath())


def build(world):
    stage = omni.usd.get_context().get_stage()
    cell = Cell(stage, HERE / "assets" / "tag36h11_id0_1000px.png", (1024, 768), part_set="fastener")
    for _ in range(150):
        app.update()
    seed = 1032  # a clean pre-grasp scene: lights, materials and the cameras aimed at the nest
    cell.apply(faults.make_scene(seed, "fastener"), make_dr(seed, True, "fastener"))
    for prim in cell.parts.values():
        _visible(prim, False)
    for p in ("/World/Gripper", "/World/RobotBase", "/World/Fixture"):
        _visible(stage.GetPrimAtPath(p), False)
    for p in ("/World/Floor", "/World/Table", "/World/Carrier"):
        collide(stage.GetPrimAtPath(p))
    for p in stage.Traverse():
        if str(p.GetPath()).startswith("/World/Bin/") and p.IsA(UsdGeom.Gprim):
            collide(p)
    box(stage, "/World/Sort/feed_tray", (0.17, 0.11, PLATE_TOP), (0.35, -0.39, PLATE_TOP / 2), (0.25, 0.27, 0.30))
    box(stage, "/World/Sort/rack", (0.17, 0.11, PLATE_TOP), (0.40, 0.28, PLATE_TOP / 2), (0.20, 0.45, 0.25))

    # Upward-looking camera: a camera's view axis is its -Z, so turn it 180 degrees about X.
    cam = UsdGeom.Camera.Define(stage, "/World/Cameras/Underside")
    cam.CreateFocalLengthAttr(40.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 768 / 1024)
    # USD's default near clip is 1 unit = 1 m here: the part 12.5 cm above was clipped away and
    # the first underside photos showed only the ceiling. Same range as the dataset cameras.
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(*UNDER_CAM))
    UsdGeom.XformCommonAPI(cam).SetRotate(Gf.Vec3f(180.0, 0.0, 0.0))
    ring = UsdLux.SphereLight.Define(stage, "/World/Lights/UndersideRing")
    ring.CreateRadiusAttr(0.01)
    ring.CreateIntensityAttr(30000.0)
    UsdGeom.XformCommonAPI(ring).SetTranslate(Gf.Vec3d(UNDER_CAM[0] + 0.05, UNDER_CAM[1], 0.03))

    robot = add_reference_to_stage(usd_path=ROBOT_USD, prim_path="/World/UR10")
    robot.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")
    gripper = SurfaceGripper(end_effector_prim_path="/World/UR10/ee_link",
                             surface_gripper_path="/World/UR10/ee_link/SurfaceGripper")
    ur10 = world.scene.add(SingleManipulator(prim_path="/World/UR10", name="ur10",
                                             end_effector_prim_path="/World/UR10/ee_link", gripper=gripper))
    ur10.set_joints_default_state(positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]))
    return stage, cell, ur10


# --- motion --------------------------------------------------------------------------
class Arm:
    def __init__(self, world, ur10):
        self.world, self.ur10 = world, ur10
        self.rmp = RMPFlowController(name="rmp", robot_articulation=ur10, attach_gripper=True)
        self.ctrl = ur10.get_articulation_controller()
        # RMPflow steers the suction-cup TIP; the articulation's end effector is the wrist
        # flange (ee_link) above it. Measure the offset once, by settling on a target: checking
        # the flange against tip targets read as 161.7 mm short of every waypoint (first runs).
        self.tip_offset = np.zeros(3)
        self.go(PARK, tol=0.0, label="calibrate", steps=900)
        self.tip_offset = self.flange() - PARK
        print(f"[arm] flange-to-tip offset {np.round(self.tip_offset, 4)} m", flush=True)

    def step(self, n: int = 1, render: bool = False):
        # Motion steps don't render (3 ray-traced cameras per step made a part take ~10 min);
        # captures render on their own, without pausing the timeline.
        for _ in range(n):
            self.world.step(render=render or args.gui)

    def flange(self) -> np.ndarray:
        return np.asarray(self.ur10.end_effector.get_world_pose()[0])

    def ee(self) -> np.ndarray:
        """The suction-cup tip: what RMPflow steers and what the waypoints are for."""
        return self.flange() - self.tip_offset

    def go(self, target, tol: float = 0.004, label: str = "", steps: int | None = None) -> bool:
        target = np.asarray(target, dtype=float)
        here = self.ee()
        if label not in ("calibrate", "via") and np.linalg.norm(here[:2] - target[:2]) > 0.30 and np.linalg.norm(here[:2] - VIA[:2]) > 0.05:
            self.go(VIA, tol=0.02, label="via")
        for i in range(steps or args.move_steps):
            self.ctrl.apply_action(self.rmp.forward(target_end_effector_position=target,
                                                    target_end_effector_orientation=EE_DOWN))
            self.step()
            if np.linalg.norm(self.ee() - target) < tol:
                break
        err = float(np.linalg.norm(self.ee() - target))
        if args.debug:
            print(f"[move] {label:<14} target {np.round(target, 3)} ee {np.round(self.ee(), 3)} "
                  f"err {err * 1000:.1f} mm steps {i + 1}", flush=True)
        return err < tol

    def grip(self, on: bool):
        (self.ur10.gripper.close if on else self.ur10.gripper.open)()
        self.step(20)
        closed = bool(self.ur10.gripper.is_closed())
        if args.debug:
            print(f"[grip] {'close' if on else 'open'} -> closed={closed}", flush=True)
        return closed

    def pick(self, part: Part) -> bool:
        p = part.pos()
        if not self.go([p[0], p[1], SAFE_Z], label="above part"):
            return False  # never descend from somewhere else: the sweep pushes parts away
        self.go([p[0], p[1], p[2] + part.pd.dims[2] / 2 + GRIP_GAP], tol=0.003, label="down to part")
        held = self.grip(True)
        self.go([p[0], p[1], SAFE_Z], label="lift")
        if args.debug:
            print(f"[pick] part {part.tp.slot} was at {np.round(p, 3)}, now {np.round(part.pos(), 3)}", flush=True)
        return held

    def place(self, xy, part: Part, surface_z: float, drop_height: float = 0.002):
        """Lower the part until its bottom is `drop_height` above `surface_z`, then release."""
        az = part.pd.dims[2]
        self.go([xy[0], xy[1], SAFE_Z])
        self.go([xy[0], xy[1], surface_z + drop_height + az + GRIP_GAP], tol=0.003)
        self.grip(False)
        self.go([xy[0], xy[1], SAFE_Z])

    def drop_in_bin(self):
        self.go([L.BIN_CENTER[0], L.BIN_CENTER[1], 0.18])
        self.grip(False)
        self.step(30)


# --- verifier --------------------------------------------------------------------------
def strip_to_json(text: str) -> str:
    out = text
    for tag in ("</think>", "</thinking>"):
        if tag in out:
            out = out.rsplit(tag, 1)[1]
    out = out.strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[1] if "\n" in out else out[3:]
        if "```" in out:
            out = out.rsplit("```", 1)[0]
    return out.strip()


def ask(system: str, user: str, images: list[Path]) -> dict:
    req = json.dumps({"system": system, "user": user, "images": [str(p.resolve()) for p in images]}).encode()
    t0 = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(args.verifier + "/verify", data=req,
                                                       headers={"Content-Type": "application/json"}),
                                timeout=180) as r:
        res = json.loads(r.read())
    try:
        v = json.loads(strip_to_json(res["raw"]))
        ft = v.get("failure_type") if isinstance(v, dict) else None
    except json.JSONDecodeError:
        v, ft = None, None
    return {"failure_type": ft or "unclear", "raw": res.get("raw", "")[:600],
            "latency_ms": round((time.monotonic() - t0) * 1000)}


def oracle(tp: trays.TrayPart, view: str) -> dict:
    return {"failure_type": tp.expected(view), "raw": "oracle", "latency_ms": 0}


# --- main -------------------------------------------------------------------------------
def main() -> int:
    seeds = list(trays.SPLITS[args.trays]) if args.trays in trays.SPLITS else [int(s) for s in args.trays.split(",")]
    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
    stage, cell, ur10 = build(world)
    world.reset()
    world.play()
    ur10.gripper.set_default_state(opened=True)
    arm = Arm(world, ur10)

    res = (1024, 768)
    annot = {}
    for name, path in (("overview", cell.views.overview), ("closeup", cell.views.closeup),
                       ("underside", "/World/Cameras/Underside")):
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        a.attach([rep.create.render_product(path, res)])
        annot[name] = a

    def shoot(names, stem):
        arm.step(3, render=True)  # settle: RTX temporal history would ghost the previous view
        rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
        paths = []
        for n in names:
            p = out / "images" / f"{stem}_{n}.png"
            Image.fromarray(np.asarray(annot[n].get_data())[:, :, :3]).save(p)
            paths.append(p)
        return paths

    system, schema = CFG["prompts"]["system"], CFG["prompts"]["schema_block"]
    top_q = (CFG["prompts"]["checkpoints"]["CP_PRE_GRASP"].format(**CFG["part_reference"])
             + "\n\nYou are given 2 image(s): image 1 is an overview of the whole cell; image 2 is a close-up "
               "of the region this checkpoint is about.\n\n" + schema)
    under_q = UNDERSIDE_QUESTION.format(part=CFG["part_reference"]["correct"]) + schema

    log = (out / "parts.jsonl").open("a")
    for seed in seeds:
        tray = trays.make_tray(seed)
        parts = [Part(stage, cell, tp, FEED_SLOTS[tp.slot]) for tp in tray.parts]
        if args.debug:
            parts = parts[:1]
        arm.step(60)
        rack_i = 0
        t_tray = time.time()
        for part in parts:
            tp, t0 = part.tp, time.time()
            rec = {"tray": seed, "slot": tp.slot, "part": tp.part, "kind": tp.kind, "defect": tp.defect,
                   "side": tp.side, "correct_action": tp.correct_action}
            held1 = arm.pick(part)
            arm.place(NEST, part, L.CARRIER_TOP)
            arm.go(PARK, tol=0.02)
            top_imgs = shoot(["overview", "closeup"], f"{seed}_{tp.slot}_top")
            held2 = arm.pick(part)
            p = part.pos()
            arm.go([UNDER_CAM[0], UNDER_CAM[1], HOVER_Z + part.pd.dims[2] + GRIP_GAP], tol=0.004)
            arm.step(20)
            under_imgs = shoot(["underside"], f"{seed}_{tp.slot}_under")
            if args.verifier == "oracle":
                vt, vu = oracle(tp, "top"), oracle(tp, "underside")
            else:
                vt = ask(system, top_q, top_imgs)
                vu = ask(system, under_q, under_imgs)
            action = decide(vt["failure_type"], vu["failure_type"])  # the robot acts on 'ours'
            if action == "accept":
                arm.place(RACK_SLOTS[rack_i % len(RACK_SLOTS)], part, PLATE_TOP)
                rack_i += 1
            else:
                arm.drop_in_bin()
            rec.update(held_from_feed=held1, held_from_nest=held2, verdict_top=vt, verdict_underside=vu,
                       action=action, decision_correct=action == tp.correct_action,
                       seconds=round(time.time() - t0, 1))
            print(f"[sort] tray {seed} slot {tp.slot} {tp.kind:<9} {tp.part:<8} {str(tp.defect):<5} {str(tp.side):<9} "
                  f"top={vt['failure_type']:<15} under={vu['failure_type']:<15} -> {action:<6} "
                  f"{'OK' if rec['decision_correct'] else 'WRONG'}", flush=True)
            part.rec = rec
        arm.go(PARK, tol=0.02)
        arm.step(90)
        for part in parts:  # where did each part physically end up?
            x, y, z = part.pos()
            in_bin = abs(x - L.BIN_CENTER[0]) < L.BIN_SIZE[0] / 2 and abs(y - L.BIN_CENTER[1]) < L.BIN_SIZE[1] / 2
            on_rack = abs(x - 0.40) < 0.085 and abs(y - 0.28) < 0.055 and z < 0.05
            where = "bin" if in_bin else "rack" if on_rack else "elsewhere"
            part.rec.update(final_xyz=[round(float(v), 4) for v in (x, y, z)], ended_in=where,
                            handled_ok=where == ("rack" if part.rec["action"] == "accept" else "bin"))
            log.write(json.dumps(part.rec) + "\n")
            part.remove()
        log.flush()
        print(f"[sort] tray {seed} done in {time.time() - t_tray:.0f}s", flush=True)
    log.close()
    rep.orchestrator.wait_until_complete()
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
