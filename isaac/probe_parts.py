#!/usr/bin/env python3
"""Preflight for the industrial-parts branch: can we load NVIDIA's nut and bolt assets by
URL, give them a real steel material, and render them close up?

The assets are referenced from NVIDIA's asset server, never copied into this repo: the
Omniverse licence allows publishing renders and video of them, not redistributing the
files.

Usage:
    ~/isaacsim/python.sh isaac/probe_parts.py --out runs/probe_parts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("runs/probe_parts"))
ap.add_argument("--only", default=None, help="render just this part (debugging)")
ap.add_argument("--no-material", action="store_true", help="keep the asset's own material")
args, _ = ap.parse_known_args()

from isaacsim.simulation_app import SimulationApp  # noqa: E402

app = SimulationApp(launch_config={"headless": True, "renderer": "RayTracedLighting",
                                   "width": 1024, "height": 768})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
# Reference ONLY each asset's visual mesh. The full assets are physics bodies (a nut is a
# free-floating articulation root), and a floating body isn't drawn until physics runs,
# which staged renders never do: the first probe rendered the bolt (fixed joint) and
# silently dropped both nuts. Props/Factory keeps a physics-free visual mesh for every
# part at <name>/instanceable_meshes.usd -> /<name>/<name>/visuals.
def factory_part(name: str) -> tuple[str, str]:
    return (f"{ROOT}/Isaac/Props/Factory/{name}/instanceable_meshes.usd", f"/{name}/{name}/visuals")


PARTS = {
    "nut_m16": factory_part("factory_nut_m16_loose"),
    "nut_m12": factory_part("factory_nut_m12_loose"),
    "bolt_m16": factory_part("factory_bolt_m16_loose"),
    "nut_m20": factory_part("factory_nut_m20_loose"),
}
STEEL = f"{ROOT}/NVIDIA/Materials/Base/Metals/Steel_Stainless.mdl"
# Metal only looks like metal with surroundings to reflect: an indoor workshop HDR.
HDR = f"{ROOT}/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr"

report: dict = {}
code = 0
try:
    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(5):
        app.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    omni.kit.commands.execute("CreateMdlMaterialPrimCommand", mtl_url=STEEL,
                              mtl_name="Steel_Stainless", mtl_path="/World/Looks/Steel")
    steel = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Steel"))

    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(table).SetScale(Gf.Vec3f(0.4, 0.3, 0.02))
    UsdGeom.XformCommonAPI(table).SetTranslate(Gf.Vec3d(0, 0, -0.01))

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    x = -0.05
    for name, (url, prim_path) in PARTS.items():
        if args.only and name != args.only:
            continue
        # The referenced asset brings its own transform ops onto whatever prim holds the
        # reference, so we pose a PARENT prim instead (the first probe found all three
        # parts piled on one spot: the reposition was silently ignored).
        holder = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        asset = UsdGeom.Xform.Define(stage, f"/World/{name}/asset").GetPrim()
        asset.GetReferences().AddReference(url, prim_path)
        for _ in range(30):
            app.update()
        r = cache.ComputeWorldBound(holder).ComputeAlignedRange()
        size = r.GetSize()
        lo = r.GetMin()
        # Stand each part on the table, side by side.
        UsdGeom.XformCommonAPI(holder).SetTranslate(Gf.Vec3d(x - (lo[0] + size[0] / 2), -(lo[1] + size[1] / 2), -lo[2]))
        if not args.no_material:
            UsdShade.MaterialBindingAPI.Apply(holder).Bind(steel, UsdShade.Tokens.strongerThanDescendants)
        for _ in range(5):
            app.update()
        after = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(holder).ComputeAlignedRange()
        report.setdefault("_placement", {})[name] = {
            "before_min_mm": [round(1000 * v, 1) for v in lo], "after_min_mm": [round(1000 * v, 1) for v in after.GetMin()],
            "after_max_mm": [round(1000 * v, 1) for v in after.GetMax()]}
        report[name] = {"loaded": holder.IsValid() and size[0] > 0,
                        "size_mm": [round(1000 * v, 1) for v in size],
                        "children": len(list(Usd.PrimRange(holder)))}
        x += 0.05

    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(600.0)
    dome.CreateTextureFileAttr(HDR)
    key = UsdLux.DistantLight.Define(stage, "/World/Key")
    key.CreateIntensityAttr(700.0)
    UsdGeom.XformCommonAPI(key).SetRotate(Gf.Vec3f(-40, 0, 30))

    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateFocalLengthAttr(35.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 0.75)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.13, -0.16, 0.14), Gf.Vec3d(0.0, 0.0, 0.01), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())

    rp = rep.create.render_product("/World/Cam", (1024, 768))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    # Remote meshes stream to the renderer asynchronously: geometry can be in the USD
    # stage (so bounds are right) before the GPU has it. Pump the app until every part
    # has had time to arrive, then settle the temporal history.
    for _ in range(120):
        app.update()
    for _ in range(6):
        rep.orchestrator.step(rt_subframes=8)
    arr = np.asarray(annot.get_data())
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr[:, :, :3].astype(np.uint8)).save(args.out / "parts.png")
    report["image"] = str(args.out / "parts.png")
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    report["error"] = f"{type(exc).__name__}: {exc}"
    code = 1
finally:
    print("=== PARTS PROBE ===\n" + json.dumps(report, indent=2), flush=True)
    rep.orchestrator.wait_until_complete()
    app.close()
sys.exit(code)
