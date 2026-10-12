#!/usr/bin/env python3
"""Open ONE scene in Isaac Sim WITH its window, so you can look around it in 3D.

The batch renderer (stage_scene.py) runs headless — no window, it just writes images.
This builds exactly the same cell, poses exactly the same scene for a seed, and then
keeps Isaac Sim open so you can orbit, zoom, and switch cameras.

Run it from a terminal on the DCV desktop (it needs a screen), when no render is running:

    ~/isaacsim/python.sh isaac/view_scene.py --part-set fastener --seed 1031
    ~/isaacsim/python.sh isaac/view_scene.py --part-set cube --seed 1005 --camera overview

  or simply:  make view SEED=1031

In the window:
  - Mouse: right-drag = look around, middle-drag = pan, scroll = zoom (or press F to frame
    the selected object).
  - Top-left of the viewport, the camera menu switches between "Overview", "Closeup"
    (the two cameras the dataset is rendered from) and "Perspective" (free camera).
  - The Stage panel (right) lists everything in the scene: /World/Parts, /World/Fixture, ...
  - Close the window to quit.

Interesting seeds (fastener set): 1002 good nut seated in the fixture, 1012 bolt (wrong
part), 1006 M20 nut (too big), 1000 M12 (too small), 1031 chipped corner, 1001 crack,
1007 rust, 1005 nut seated crooked (14 mm off, 11 deg tilt), 1004 dropped on the floor,
1003 empty grasp, 1014 nut held in the gripper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import faults  # noqa: E402
from dr import make_dr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--part-set", default="fastener", choices=("cube", "fastener"))
ap.add_argument("--seed", type=int, default=1031)
ap.add_argument("--camera", default="closeup", choices=("closeup", "overview", "free"))
ap.add_argument("--check", action="store_true", help="build the scene without a window and exit (self-test)")
args, _ = ap.parse_known_args()

from isaacsim.simulation_app import SimulationApp  # noqa: E402

app = SimulationApp(launch_config={"headless": args.check, "renderer": "RayTracedLighting",
                                   "width": 1280, "height": 960})

import omni.usd  # noqa: E402

from cell import Cell  # noqa: E402

spec = faults.make_scene(args.seed, args.part_set)
print(f"\n[view] seed {args.seed} ({args.part_set}): fault {spec.fault} at {spec.checkpoint}\n"
      f"[view] correct answer: {spec.expected.failure_type} @ {spec.expected.location_hint}"
      f"   details: {spec.truth or '-'}\n", flush=True)

ctx = omni.usd.get_context()
ctx.new_stage()
for _ in range(5):
    app.update()
cell = Cell(ctx.get_stage(), HERE / "assets" / "tag36h11_id0_1000px.png", (1024, 768), part_set=args.part_set)
for _ in range(20 if args.part_set == "cube" else 150):  # remote assets load once
    app.update()
cell.apply(spec, make_dr(args.seed, True, args.part_set))

if args.camera != "free":
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = getattr(cell.views, args.camera)
    except Exception as exc:  # the viewport API differs between Kit versions; not fatal
        print(f"[view] could not switch camera automatically ({exc}); use the viewport's camera menu", flush=True)

if args.check:
    for _ in range(10):
        app.update()
    print("[view] check OK: scene built and posed", flush=True)
    app.close()
    sys.exit(0)

print("[view] window open. Close it to quit.", flush=True)
while app.is_running():
    app.update()
app.close()
