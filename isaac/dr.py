"""Domain randomisation: lighting, colours, camera jitter and environment per scene.

Pure Python, so its statistical properties can be tested without Isaac Sim — in
particular that nothing it draws is correlated with the fault (see tests/test_dr.py).
Everything is derived from the seed through its own random streams, so a scene's look is
reproducible and independent of the placement stream in faults.py.
"""
from __future__ import annotations

import random

import partsets

TABLE_COLOURS = [(0.55, 0.56, 0.58), (0.36, 0.42, 0.38), (0.62, 0.60, 0.55), (0.32, 0.34, 0.38)]


def make_dr(seed: int, enabled: bool, part_set: str = "cube") -> dict:
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
        **(_fastener_dr(seed) if part_set == "fastener" else {}),
    }


def _fastener_dr(seed: int) -> dict:
    """Extra randomisation for metal parts, from its own stream so cube scenes are
    untouched. Steel reflects its surroundings, so the environment matters as much as the
    lights: rotate through four workshop HDRs. A textured dome is brighter per unit of
    intensity than a plain one, and the key light is softened to keep highlights in range
    (values from the parts probe, which looked right)."""
    rng = random.Random(seed * 6271 + 5)
    # Drawn from the stream, NOT from seed % 4: the fault is seed % 6, and the shared
    # factor 2 tied the environment to the fault class (a code review caught every defect
    # rendered in two environments and every wrong part in the other two — a label leak a
    # model could learn). tests/test_dr.py checks the independence.
    return {"hdr": partsets.WORKSHOP_HDRS[rng.randrange(len(partsets.WORKSHOP_HDRS))],
            "dome_intensity": rng.uniform(450.0, 800.0),
            "key_intensity": rng.uniform(300.0, 700.0)}
