#!/usr/bin/env python3
"""Preflight check for the Isaac Sim side of the cell. Run before M1 work.

Answers four questions that otherwise cost an afternoon of guessing:
  1. Does a headless SimulationApp launch at all on this GPU?
  2. Which namespace works — `isaacsim.*` (5.x) or the deprecated `omni.isaac.*`?
  3. Is Replicator importable, and at what version?
  4. Is the Nucleus asset root reachable? NVIDIA's SDG samples pull props from it;
     if it is unreachable from this box we must build the cell from local primitives.

Usage:
    ~/isaacsim/python.sh isaac/probe_env.py
"""
from __future__ import annotations

import json
import sys
import traceback

REPORT: dict[str, object] = {}


def note(key: str, value: object) -> None:
    REPORT[key] = value
    print(f"  {key}: {value}", flush=True)


print("[1/4] launching headless SimulationApp...", flush=True)
try:
    # Isaac Sim 5.x namespace. The 2022-era samples use `omni.isaac.kit`, which now
    # lives in extsDeprecated — prefer this and fall back only to report the fact.
    from isaacsim.simulation_app import SimulationApp

    launcher = "isaacsim.simulation_app"
except ImportError:
    from omni.isaac.kit import SimulationApp  # type: ignore[no-redef]

    launcher = "omni.isaac.kit (DEPRECATED)"

simulation_app = SimulationApp(
    launch_config={"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480}
)
note("launcher", launcher)

exit_code = 0
try:
    print("[2/4] probing core namespace...", flush=True)
    try:
        import isaacsim.core.api  # noqa: F401
        from isaacsim.core.utils.stage import create_new_stage  # noqa: F401

        note("core_namespace", "isaacsim.core.* OK")
    except Exception as exc:
        note("core_namespace", f"FAILED: {type(exc).__name__}: {exc}")
        exit_code = 1

    print("[3/4] probing Replicator...", flush=True)
    try:
        import omni.replicator.core as rep

        note("replicator", getattr(rep, "__version__", "importable, version attr absent"))
        note("replicator_has_writer_registry", hasattr(rep, "WriterRegistry"))
        # The custom writer is how we emit Verdict JSON labels instead of KITTI boxes.
        note("replicator_has_annotator_registry", hasattr(rep, "AnnotatorRegistry"))
    except Exception as exc:
        note("replicator", f"FAILED: {type(exc).__name__}: {exc}")
        exit_code = 1

    print("[4/4] probing Nucleus asset root (needs network)...", flush=True)
    try:
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
        note("assets_root", root or "None — Nucleus unreachable")
        note("assets_root_usable", bool(root))
    except Exception as exc:
        note("assets_root", f"FAILED: {type(exc).__name__}: {exc}")
        note("assets_root_usable", False)

except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    print("\n=== PROBE REPORT ===", flush=True)
    print(json.dumps(REPORT, indent=2), flush=True)
    print("closing simulation app...", flush=True)
    simulation_app.close()

sys.exit(exit_code)
