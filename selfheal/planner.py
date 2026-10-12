"""The recovery planner's action space and the code-level guardrails around it.

The five tools ARE the action space: the model can never move the robot directly and
never emits a coordinate. The guardrail runs after the model answers, in code, because a
prompt is a request and this is a constraint.
"""
from __future__ import annotations

# The whitelist. Exactly one call per decision. See SPEC.md §11.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "continue_cycle",
            "description": (
                "The verdict is acceptable or was a false alarm. Proceed with the cycle "
                "unchanged. Unsafe if the part is confirmed wrong, defective, or missing."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "relocalize_and_regrasp",
            "description": (
                "Re-find the part using the depth camera in the given region, then grasp it "
                "again. Use when the part was dropped or is displaced but still reachable. "
                "Unsafe if the part is off the table or out of reach."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "enum": ["carrier", "table"],
                        "description": "Where to look for the part.",
                    },
                    "reason": {"type": "string", "description": "One sentence why."},
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reseat_part",
            "description": (
                "Pick the part back out of the fixture and place it again. Use only for a "
                "misaligned seat of an otherwise correct, undamaged part."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "One sentence why."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_to_bin",
            "description": (
                "Pick the current part from wherever it is and drop it in the reject bin. "
                "This is the ONLY correct action for a wrong or defective part."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "One sentence why."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_and_alert",
            "description": (
                "Stop the cell and raise an operator alert. Use when the situation is "
                "unclear, the part is unreachable, or recoveries have been exhausted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "What the operator needs to know.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]

WHITELIST = {t["function"]["name"] for t in TOOLS}

SYSTEM_PROMPT = """You are the recovery planner for an industrial pick-and-place cell.
You receive a verifier report about a failed or uncertain step.
Choose exactly ONE tool. You may never move the robot directly.
Rules:
- Prefer the least invasive recovery likely to succeed.
- Wrong or defective parts must go to the reject bin, never to the fixture.
- If a part is off the table, out of reach, or the situation is unclear, call stop_and_alert.
- If two recoveries have already failed this cycle, call stop_and_alert.
Give a one-sentence reason inside the tool arguments where available."""

# Code-enforced, not prompt-enforced. SPEC.md §11.
FORCED_REJECT = {"wrong_part", "defective_part"}


def guardrail(tool_name: str, failure_type: str) -> tuple[str, str | None]:
    """Return (final_tool, override_note). Overrides are a reported metric."""
    if tool_name not in WHITELIST:
        return "stop_and_alert", f"tool {tool_name!r} not in whitelist"
    if failure_type in FORCED_REJECT and tool_name != "reject_to_bin":
        return "reject_to_bin", f"{failure_type} must go to the bin, model chose {tool_name}"
    return tool_name, None
