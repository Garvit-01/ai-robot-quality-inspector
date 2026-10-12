"""The verifier's output contract: the Verdict schema and how to parse a model reply.

Shared by the real pipeline, the benchmark and the smoke tests, so there is exactly one
definition of what a valid verdict is.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
