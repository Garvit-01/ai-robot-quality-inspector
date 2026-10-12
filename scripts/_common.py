"""Shared config/env loading and the verifier output schema.

Kept in scripts/ for M0. Moves into src/selfheal_cell/ once the ROS2 package exists;
the smoke scripts are meant to be throwaway-ish, the schema is not.
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "selfheal.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so `import selfheal` works from scripts/ and eval/

from selfheal.verdict import FailureType, LocationHint, Verdict, strip_to_json  # noqa: E402  (moved; re-exported)

__all__ = ["FailureType", "LocationHint", "Verdict", "strip_to_json", "load_config", "load_dotenv",
           "require_env", "client_for", "image_data_uri", "REPO_ROOT", "SHARED_KEY_ENV"]


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


#: One Token Factory key serves both the planner and the verifier in Tier 1. Asking for it
#: three times in .env is three chances to paste it wrong, so the per-role variables are
#: optional and fall back to this. They matter in Tier 2, when the verifier moves to its own
#: fine-tuned endpoint with its own credentials.
SHARED_KEY_ENV = "NEBIUS_TOKEN_FACTORY_KEY"


def client_for(section: dict):
    """Build an OpenAI-compatible client from a config section's *_env indirection."""
    from openai import OpenAI

    api_key = os.environ.get(section["api_key_env"], "").strip()
    if not api_key:
        api_key = require_env(SHARED_KEY_ENV)

    return OpenAI(
        base_url=require_env(section["base_url_env"]),
        api_key=api_key,
        timeout=section.get("timeout_s", 20),
    )


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
