#!/usr/bin/env python3
"""Render the README diagrams (docs/diagrams.html) to PNG images in docs/.

    .venv/bin/python scripts/render_diagrams.py      # needs: pip install playwright; playwright install chromium
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent.parent / "docs"
SECTIONS = {"overview": "arch_overview.png", "live": "arch_live_loop.png", "build": "arch_where_it_runs.png",
            "models": "model_comparison.png"}

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1700, "height": 1200}, device_scale_factor=1.5)
    page.goto((DOCS / "diagrams.html").as_uri())
    page.wait_for_load_state("networkidle")
    page.evaluate("document.fonts.ready")
    for sec, name in SECTIONS.items():
        page.locator(f"#{sec}").screenshot(path=str(DOCS / name), omit_background=True)
        print("[diagrams]", DOCS / name)
    browser.close()
