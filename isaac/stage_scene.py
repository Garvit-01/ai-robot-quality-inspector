#!/usr/bin/env python3
"""Render fault scenes with ground truth. The data factory for evaluation and Tier 2.

For each seed: derive the scene (faults.make_scene), pose the cell, apply domain
randomisation, render one frame from two cameras, and write the images next to a JSON
file holding the correct answer. Nothing is labelled by hand: we placed the part, so we
know the answer.

Imperative on purpose (see SPEC.md §7.1): NVIDIA's SDG sample uses a declarative
`rep.trigger.on_frame` graph, which cannot express "frame N is fault F3 at 11 degrees,
and the label says so". Here Python sets the state, then `rep.orchestrator.step()`
captures exactly one frame of it.

Usage (Isaac Sim's Python, headless):
    ~/isaacsim/python.sh isaac/stage_scene.py --start 1000 --count 12 --out runs/scenes/smoke

Output:
    <out>/images/<seed>_overview.png
    <out>/images/<seed>_closeup.png
    <out>/labels/<seed>.json
    <out>/manifest.jsonl          one line per seed, including failures
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import faults  # noqa: E402  (pure Python, safe before SimulationApp)

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--start", type=int, default=1000, help="first seed")
parser.add_argument("--count", type=int, default=12, help="number of consecutive seeds")
parser.add_argument("--seeds", type=str, default=None,
                    help="explicit comma-separated seeds; overrides --start/--count")
parser.add_argument("--out", type=Path, default=REPO / "runs" / "scenes" / "smoke")
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=768)
parser.add_argument("--rt-subframes", type=int, default=8,
                    help="RTX render subframes per capture")
parser.add_argument("--settle-steps", type=int, default=3,
                    help="frames rendered and discarded after each re-pose, before the capture. "
                         "RTX keeps temporal history (anti-aliasing, denoiser); without settling, a "
                         "faint ghost of the PREVIOUS scene shows in the new frame")
parser.add_argument("--no-dr", action="store_true", help="disable domain randomisation")
args, _unknown = parser.parse_known_args()

from isaacsim.simulation_app import SimulationApp  # noqa: E402

simulation_app = SimulationApp(launch_config={
    "headless": True,
    "renderer": "RayTracedLighting",
    "width": args.width,
    "height": args.height,
})

# Everything below needs the running app.
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402

from cell import Cell  # noqa: E402

TAG_TEXTURE = HERE / "assets" / "tag36h11_id0_1000px.png"

TABLE_COLOURS = [(0.55, 0.56, 0.58), (0.36, 0.42, 0.38), (0.62, 0.60, 0.55), (0.32, 0.34, 0.38)]


def make_dr(seed: int, enabled: bool) -> dict:
    """Domain randomisation for one scene, from its own seed-derived stream.

    Deliberately moderate. The goal is that a model learns "misaligned seat", not "the
    table that is beige". Part colour varies within one grey family across ALL part
    kinds, so colour is never a shortcut to the label.
    """
    if not enabled:
        return {"table_rgb": TABLE_COLOURS[0], "part_rgb": (0.62, 0.62, 0.64),
                "dome_intensity": 180.0, "key_intensity": 600.0, "color_temp_k": 5600.0,
                "key_elevation_deg": 60.0, "key_azimuth_deg": 210.0, "cam_jitter": (0.0, 0.0, 0.0)}
    rng = random.Random(seed * 7919 + 3)
    base = rng.choice(TABLE_COLOURS)
    table = tuple(min(1.0, max(0.0, c + rng.uniform(-0.04, 0.04))) for c in base)
    g = rng.uniform(0.52, 0.74)
    return {
        "table_rgb": table,
        "part_rgb": (g, g, g + rng.uniform(0.0, 0.03)),
        # First render pass was ~4x overexposed (a 0.18-albedo floor came out near white),
        # so these are the original ranges scaled by roughly 0.22.
        "dome_intensity": rng.uniform(100.0, 260.0),
        "key_intensity": rng.uniform(350.0, 850.0),
        "color_temp_k": rng.uniform(4800.0, 6500.0),
        "key_elevation_deg": rng.uniform(40.0, 75.0),
        "key_azimuth_deg": rng.uniform(0.0, 360.0),
        "cam_jitter": (rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)),
    }


def save_rgb(annot, path: Path) -> tuple[int, int]:
    data = annot.get_data()
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.size == 0:
        raise RuntimeError(f"empty frame from annotator (shape {arr.shape})")
    Image.fromarray(arr[:, :, :3].astype(np.uint8)).save(path)
    return arr.shape[1], arr.shape[0]


def main() -> int:
    out: Path = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    manifest = (out / "manifest.jsonl").open("w")  # "a" duplicated rows on every re-run

    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(5):
        simulation_app.update()
    stage = ctx.get_stage()

    cell = Cell(stage, TAG_TEXTURE)
    for _ in range(20):
        simulation_app.update()

    res = (args.width, args.height)
    rp_overview = rep.create.render_product(cell.views.overview, res)
    rp_closeup = rep.create.render_product(cell.views.closeup, res)
    a_overview = rep.AnnotatorRegistry.get_annotator("rgb")
    a_overview.attach([rp_overview])
    a_closeup = rep.AnnotatorRegistry.get_annotator("rgb")
    a_closeup.attach([rp_closeup])

    # Warm-up: the first captures render before textures and shaders finish loading.
    # Pose a real scene and throw a few frames away so seed #1 isn't a broken image.
    first = int(args.seeds.split(",")[0]) if args.seeds else args.start
    cell.apply(faults.make_scene(first), make_dr(first, not args.no_dr))
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=args.rt_subframes)

    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds
             else list(range(args.start, args.start + args.count)))
    ok = failed = 0
    t_run = time.time()
    for seed in seeds:
        t0 = time.time()
        record = {"seed": seed}
        try:
            spec = faults.make_scene(seed)
            dr = make_dr(seed, not args.no_dr)
            placed = cell.apply(spec, dr)
            for _ in range(args.settle_steps):
                rep.orchestrator.step(rt_subframes=args.rt_subframes)
            rep.orchestrator.step(rt_subframes=args.rt_subframes)

            ov = out / "images" / f"{seed}_overview.png"
            cu = out / "images" / f"{seed}_closeup.png"
            size = save_rgb(a_overview, ov)
            save_rgb(a_closeup, cu)

            label = {
                "seed": seed,
                "images": {"overview": str(ov.relative_to(out)), "closeup": str(cu.relative_to(out))},
                "resolution": list(size),
                "scene": spec.to_json(),
                "placed": placed,
                "domain_randomisation": dr,
            }
            (out / "labels" / f"{seed}.json").write_text(json.dumps(label, indent=2))
            record.update(ok=True, fault=spec.fault, checkpoint=spec.checkpoint,
                          failure_type=spec.expected.failure_type, secs=round(time.time() - t0, 2))
            ok += 1
        except Exception as exc:  # one bad scene must not kill a 2000-frame run
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                          trace=traceback.format_exc(limit=3))
            failed += 1
        manifest.write(json.dumps(record) + "\n")
        manifest.flush()
        print(f"[stage] seed {seed}: {'ok' if record.get('ok') else 'FAILED'} "
              f"{record.get('fault', '')} {record.get('checkpoint', '')} "
              f"{record.get('secs', record.get('error', ''))}", flush=True)

    elapsed = time.time() - t_run
    print(f"[stage] done: {ok} ok, {failed} failed, {elapsed:.1f}s "
          f"({elapsed / max(1, ok + failed):.2f}s per scene) -> {out}", flush=True)
    manifest.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        traceback.print_exc()
    finally:
        rep.orchestrator.wait_until_complete()
        simulation_app.close()
    sys.exit(code)
