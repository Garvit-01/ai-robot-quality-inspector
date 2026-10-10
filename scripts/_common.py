"""Shared config/env loading and the verifier output schema.

Kept in scripts/ for M0. Moves into src/selfheal_cell/ once the ROS2 package exists;
the smoke scripts are meant to be throwaway-ish, the schema is not.
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError  # noqa: F401  (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "selfheal.yaml"

FailureType = Literal[
    "none",
    "empty_grasp",
    "part_displaced",
    "part_dropped_recoverable",
    "part_dropped_unrecoverable",
    "misaligned_seat",
    "wrong_part",
    "defective_part",
    "unclear",
]

LocationHint = Literal[
    "gripper",
    "carrier",
    "table_in_reach",
    "table_out_of_reach",
    "off_table",
    "fixture",
    "reject_bin",
    "unknown",
]


class Verdict(BaseModel):
    """Strict schema for a verifier response. Anything else is a parse failure."""

    checkpoint: str
    step_ok: bool
    failure_type: FailureType
    evidence: str = Field(max_length=500)
    location_hint: LocationHint
    confidence: float = Field(ge=0.0, le=1.0)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(f"Config not found: {CONFIG_PATH}")
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def load_dotenv() -> None:
    """Minimal .env loader. Avoids a python-dotenv dependency for two smoke scripts.

    Real environment variables always win, so you can override per-invocation.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(name: str) -> str:
    """Fail loudly and usefully. A missing key should never look like a model error."""
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(
            f"\nMissing required environment variable: {name}\n"
            f"  1. cp {REPO_ROOT}/.env.example {REPO_ROOT}/.env\n"
            f"  2. fill in {name} (Token Factory keys start with 'v1.')\n"
        )
    return value


def client_for(section: dict):
    """Build an OpenAI-compatible client from a config section's *_env indirection."""
    from openai import OpenAI

    return OpenAI(
        base_url=require_env(section["base_url_env"]),
        api_key=require_env(section["api_key_env"]),
        timeout=section.get("timeout_s", 20),
    )


def strip_to_json(text: str) -> str:
    """Reasoning models wrap answers in <think> blocks and markdown fences. Strip both.

    Takes everything after the LAST closing think tag, since nested or repeated blocks
    show up in practice, then peels one code fence if present.
    """
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


def image_data_uri(path: Path, longest_side: int, quality: int) -> str:
    """Encode an image as a base64 data URI, downscaled to longest_side.

    Falls back to raw bytes if Pillow is absent, so the smoke test still runs.
    """
    try:
        import io

        from PIL import Image

        img = Image.open(path).convert("RGB")
        if max(img.size) > longest_side:
            scale = longest_side / max(img.size)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        payload = buf.getvalue()
    except ImportError:
        payload = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode()
