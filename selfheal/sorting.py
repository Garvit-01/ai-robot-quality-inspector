"""Sorting-line decisions and how they are judged. Pure Python: imported by the simulator
(isaac/run_sorting.py, Isaac Sim's own Python) and by the benchmark (eval/bench_sorting.py).

A part is accepted (to the good-parts rack) or rejected (to the bin). Three policies are
compared on the very same parts, because each one only needs what the camera(s) saw:

  B         today's factory: a gripper sensor knows a part is held, nothing about what it
            is. Everything it can pick up is accepted.
  top_only  ours with the single top camera (the original cell).
  ours      ours with both cameras: top + upward-looking.

The robot in the simulation acts on `ours`. The other two are computed from the same logged
verdicts, so the comparison is exact, not a separate run with different luck.
"""
from __future__ import annotations

REJECT_IF = frozenset({"wrong_part", "defective_part", "unclear"})


def decide(top: str, underside: str = "none") -> str:
    """The lookup table: reject anything either view calls wrong or damaged. An unclear view
    is also rejected: never ship a part the inspector couldn't vouch for. (Every Nemotron size
    matched this table on 180/180 scenes, so the robot uses it directly; see decisions D27.)"""
    return "reject" if {top, underside} & REJECT_IF else "accept"


def policy_actions(top: str, underside: str) -> dict[str, str]:
    return {"B": "accept", "top_only": decide(top), "ours": decide(top, underside)}


def outcome(action: str, good: bool) -> str:
    """shipped_bad   a bad part reached the good-parts rack (the expensive mistake)
    scrapped_good    a good part went to the bin
    correct          good accepted, or bad rejected"""
    if action == "accept":
        return "correct" if good else "shipped_bad"
    return "scrapped_good" if good else "correct"
