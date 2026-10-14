#!/usr/bin/env python3
"""Feasibility probe for the sorting cell: can a real arm move a physics nut in our cell?

Builds the same cell as the dataset (cell.py, fastener set), hides the stand-in gripper and
robot-base cylinder, adds colliders to the furniture, drops in a UR10 with NVIDIA's suction
gripper at the robot-base position, spawns ONE physics M16 nut on the inspection nest, and
runs NVIDIA's pick-and-place controller (RMPflow) to carry it into the reject bin.

Answers, before building the full sorting line on top of it:
  - does the UR10 reach the nest and the bin from the origin?
  - does the suction gripper hold a 24 mm nut (top face has a 16 mm hole)?
  - does the nut end up in the bin?

The nut's visual is NVIDIA's mesh (instanceable, so it can't take physics APIs); its
collision shape is an invisible cylinder of the same size on the rigid body.

Usage (Isaac Sim's Python; GPU must be free):
    ~/isaacsim/python.sh isaac/probe_arm.py --out runs/probe_arm            # headless, saves PNGs
    ~/isaacsim/python.sh isaac/probe_arm.py --gui                           # watch it (DCV desktop)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import faults  # noqa: E402
import layout as L  # noqa: E402
import partsets  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gui", action="store_true")
ap.add_argument("--out", type=Path, default=Path("runs/probe_arm"))
ap.add_argument("--steps", type=int, default=2400, help="physics steps at 60 Hz")
ap.add_argument("--pick-offset-mm", type=float, default=0.0,
                help="pick this far off the nut's centre (the top face has a hole in the middle)")
args, _ = ap.parse_known_args()

from isaacsim.simulation_app import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui, "renderer": "RayTracedLighting", "width": 1024, "height": 768})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot.manipulators import SingleManipulator  # noqa: E402
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import (  # noqa: E402
    PickPlaceController)
from isaacsim.robot.manipulators.grippers import SurfaceGripper  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

from cell import Cell, _visible  # noqa: E402
from dr import make_dr  # noqa: E402

ROBOT_USD = f"{partsets.ASSET_ROOT}/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
NUT = partsets.FASTENER.correct


def collide(prim, approximation: str | None = None) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)
    if approximation and prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)


def physics_nut(stage, path: str, xyz) -> UsdGeom.Xform:
    """Rigid body: invisible cylinder collider + NVIDIA's visual mesh, origin at the nut's centre."""
    body = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(body).SetTranslate(Gf.Vec3d(*xyz))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.03)
    col = UsdGeom.Cylinder.Define(stage, f"{path}/collider")
    col.CreateRadiusAttr(NUT.dims[1] / 2)  # across the corners
    col.CreateHeightAttr(NUT.dims[2])
    col.CreateAxisAttr("Z")
    UsdGeom.Imageable(col).CreatePurposeAttr("guide")  # not rendered, still collides
    collide(col.GetPrim())
    asset = UsdGeom.Xform.Define(stage, f"{path}/asset")
    asset.GetPrim().GetReferences().AddReference(NUT.usd_url, NUT.usd_prim)
    c = NUT.usd_center
    UsdGeom.XformCommonAPI(asset).SetTranslate(Gf.Vec3d(-c[0], -c[1], -c[2]))
    return body


def main() -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
    stage = omni.usd.get_context().get_stage()

    # The dataset cell, posed for a clean pre-grasp scene (lights, cameras, materials).
    cell = Cell(stage, HERE / "assets" / "tag36h11_id0_1000px.png", (1024, 768), part_set="fastener")
    for _ in range(150):  # remote assets load once
        app.update()
    seed = 1032  # a clean CP_PRE_GRASP scene
    cell.apply(faults.make_scene(seed, "fastener"), make_dr(seed, True, "fastener"))
    for prim in cell.parts.values():
        _visible(prim, False)
    _visible(cell.gripper, False)
    _visible(stage.GetPrimAtPath("/World/RobotBase"), False)

    # Furniture collides; meshes with holes keep their exact shape (static triangle mesh).
    for p in ("/World/Floor", "/World/Table", "/World/Carrier"):
        collide(stage.GetPrimAtPath(p))
    for p in stage.Traverse():
        path = str(p.GetPath())
        if path.startswith(("/World/Fixture/", "/World/Bin/")) and p.IsA(UsdGeom.Gprim):
            collide(p, "none")

    # The arm, where the stand-in base was. Short suction cup variant, as in NVIDIA's example.
    robot = add_reference_to_stage(usd_path=ROBOT_USD, prim_path="/World/UR10")
    robot.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")
    gripper = SurfaceGripper(end_effector_prim_path="/World/UR10/ee_link",
                             surface_gripper_path="/World/UR10/ee_link/SurfaceGripper")
    ur10 = world.scene.add(SingleManipulator(prim_path="/World/UR10", name="ur10",
                                             end_effector_prim_path="/World/UR10/ee_link", gripper=gripper))
    ur10.set_joints_default_state(positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]))

    nest = (L.CARRIER_SLOT[0], L.CARRIER_SLOT[1], L.CARRIER_TOP + NUT.dims[2] / 2 + 0.002)
    nut = physics_nut(stage, "/World/Sort/nut0", nest)
    world.reset()

    ur10.gripper.set_default_state(opened=True)
    controller = PickPlaceController(name="pick_place", gripper=ur10.gripper, robot_articulation=ur10)
    arm = ur10.get_articulation_controller()

    rp = rep.create.render_product(cell.views.overview, (1024, 768))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

    def nut_pos():
        m = UsdGeom.Xformable(nut).ComputeLocalToWorldTransform(0)
        t = m.ExtractTranslation()
        return [round(t[0], 4), round(t[1], 4), round(t[2], 4)]

    bx, by = L.BIN_CENTER
    place = np.array([bx, by, L.BIN_SIZE[2] + 0.06])  # release above the open bin
    off = args.pick_offset_mm / 1000.0
    log, shots, done_at = [], 0, None
    for step in range(args.steps):
        world.step(render=args.gui or step % 200 == 0)
        pick = np.array(nut_pos()) + np.array([off, 0.0, 0.0])
        action = controller.forward(picking_position=pick, placing_position=place,
                                    current_joint_positions=ur10.get_joint_positions(),
                                    end_effector_offset=np.array([0, 0, 0.02]))
        arm.apply_action(action)
        if step % 200 == 0:
            rep.orchestrator.step(rt_subframes=4)
            Image.fromarray(np.asarray(rgb.get_data())[:, :, :3]).save(args.out / f"step_{step:05d}.png")
            shots += 1
            log.append({"step": step, "event": controller.get_current_event(), "nut": nut_pos(),
                        "gripper_closed": bool(ur10.gripper.is_closed())})
            print(f"[probe] step {step} event {controller.get_current_event()} nut {nut_pos()} "
                  f"gripper_closed {ur10.gripper.is_closed()}", flush=True)
        if controller.is_done() and done_at is None:
            done_at = step
            print(f"[probe] controller done at step {step}", flush=True)
        if done_at is not None and step > done_at + 120:  # let the nut settle in the bin
            break

    final = nut_pos()
    hx, hy = L.BIN_SIZE[0] / 2, L.BIN_SIZE[1] / 2
    in_bin = abs(final[0] - bx) < hx and abs(final[1] - by) < hy and final[2] < L.BIN_SIZE[2]
    result = {"controller_done_step": done_at, "nut_final": final, "nut_start": list(nest),
              "in_bin": in_bin, "pick_offset_mm": args.pick_offset_mm, "log": log}
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[probe] RESULT in_bin={in_bin} final={final} done_at={done_at} -> {args.out}", flush=True)
    rep.orchestrator.wait_until_complete()
    app.close()
    return 0 if in_bin else 1


if __name__ == "__main__":
    sys.exit(main())
