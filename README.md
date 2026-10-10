# Self-Healing Work Cell

**Apache-2.0** · Nebius x NVIDIA Global AI Hackathon 2026 · **Physical AI** track

A robot work cell that understands *what* went wrong, not just *that* something went wrong —
and a measurement of whether a small fine-tuned vision model does that job better than a
large zero-shot one.

> **Status: in development (started Oct 10, 2026).** Results sections below are
> placeholders until the eval harness runs. Nothing in this README will claim a number
> that is not reproducible from `runs/`.

---

## The problem

An industrial pick-and-place cell is effectively blind. It has one sensor that matters: a
gripper switch reporting "something is / isn't between my fingers." If a part slips out
mid-transport, the arm completes the cycle and places nothing. If the wrong part is loaded,
it goes into the fixture. If the part lands crooked, it ships.

Anything the gripper switch can't detect ends the same way: **stop the cell, call a
technician over.**

Adding real checks means hand-engineering a vision program per part number, per failure
mode. High-mix low-volume shops — job shops, machine tending — run hundreds of part
numbers and cannot pay for that. So they don't, and they eat the stoppages.

## The approach

Replace N hand-engineered checks with **one general verifier plus a bounded
decision-maker.**

At four checkpoints in the cycle, capture frames and ask a vision-language model one
question. It answers in strict JSON with a failure type drawn from a fixed enum. Hard
sensor rules are then fused on top — the gripper sensor always wins on "is something in
the hand," because a VLM should not overrule a switch.

| Checkpoint | Question |
|---|---|
| `CP_PRE_GRASP` | Right part, undamaged, roughly where expected? |
| `CP_POST_LIFT` | Is the part actually in the gripper? |
| `CP_POST_TRANSPORT` | Still in the gripper? If not, where did it go? |
| `CP_POST_PLACE` | Correctly seated in the fixture? |

If a checkpoint fails, **Nemotron** picks exactly one action from a whitelist of five
skills we wrote and tested. It never moves the robot and never emits a coordinate — it
supplies a qualitative region, and classical depth-based geometry does the rest.

```
Sensor ──▶ Verifier (VLM) ──▶ fuse with gripper signal ──▶ Nemotron picks 1 skill
   ▲                                                              │
   └──────────── re-verify the same checkpoint ◀── MoveIt2 executes
```

**The guardrails are in code, not in the prompt.** A wrong or defective part can only ever
go to the reject bin; if the model picks anything else we override it and count the
override as a reported metric. Non-whitelisted tool, malformed arguments, no tool call, or
a timeout all collapse to `stop_and_alert`. Two failed recoveries forces a stop.

## Why fine-tune — the part that matters

No plant will stream floor camera frames to a third-party API every cycle. Data
sovereignty, latency, and per-cycle cost all forbid it. A small self-hosted specialist is
the only deployable form of this idea.

Isaac Sim makes testing that cheap: we *stage* each fault, so **the ground-truth label is
free by construction** — zero annotation. That turns the simulator into a training-data
factory and lets us run the same evaluation twice:

- **Tier 1** — hosted zero-shot vision model.
- **Tier 2** — [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
  LoRA fine-tuned on staged faults, trained as a Nebius Serverless Job and served on a
  Nebius endpoint.

Same held-out eval split for both, so the columns are comparable. We report accuracy,
p50/p90 latency, and cost per 100 cycles for each.

## How Nebius and NVIDIA models are used

| Component | What | Where it runs |
|---|---|---|
| Recovery planner | NVIDIA **Nemotron**, tool calling, temperature 0 | **Nebius Token Factory** |
| Verifier (Tier 1) | hosted NVIDIA vision model, zero-shot | **Nebius Token Factory** |
| Verifier (Tier 2) | **Cosmos-Reason1-7B** + our LoRA adapter | trained in a **Nebius AI Cloud Serverless Job**, served on a **Nebius Serverless Endpoint** |
| Fault-scene rendering | Isaac Sim 5.1.0 | local GPU (Tesla T4) |

Model IDs and base URLs are **never hardcoded** — they resolve from environment variables
via [`config/selfheal.yaml`](config/selfheal.yaml), which is what makes the Tier 1 → Tier 2
swap a config change. Always re-pin IDs from the live catalogue before a run:

```bash
.venv/bin/python scripts/smoke_nemotron.py --list
```

## Setup

Requires ROS2 Jazzy, MoveIt2, and Isaac Sim 5.1.0 for the cell itself; the model-facing
pieces need only Python.

```bash
git clone git@github.com:Garvit-01/selfheal-workcell.git
cd selfheal-workcell

python3 -m venv .venv
.venv/bin/pip install openai pydantic pyyaml pillow

cp .env.example .env
$EDITOR .env          # add your Nebius Token Factory key (starts with "v1.")
```

Build the ROS2 packages by symlinking them into your workspace — this repo is never built
in place:

```bash
ln -s "$PWD/src/selfheal_cell" ~/ros2_ws/src/
ln -s "$PWD/src/selfheal_interfaces" ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select selfheal_cell selfheal_interfaces
```

## Run

Smoke-test the two model paths first. Neither needs Isaac Sim:

```bash
# planner: asserts exactly one whitelisted, well-formed tool call
.venv/bin/python scripts/smoke_nemotron.py

# ...and that the code guardrail overrides a bad choice
.venv/bin/python scripts/smoke_nemotron.py --failure-type wrong_part

# verifier: asserts a schema-valid verdict on a synthetic frame
.venv/bin/python scripts/smoke_verifier.py --selftest
```

Full trial runs and metrics (harness in progress):

```bash
.venv/bin/python -m selfheal_cell.trial_runner --condition C --trials 25
.venv/bin/python eval/compute_metrics.py runs/<run_id>
```

## Experimental conditions

| Cond | Name | Checks | On failure |
|---|---|---|---|
| A | no recovery | none | proceeds blindly |
| B | **industrial baseline** | gripper signal at post-lift and post-transport only | retry pick at original location 2x, then stop |
| C | **ours** | verifier at all 4 checkpoints + sensor fusion | Nemotron picks one whitelisted skill, max 2 recoveries |
| D | ablation | same verifier as C | fixed `failure_type → skill` dict, no LLM |

Condition D exists to answer the obvious objection: *why an LLM, when a switch statement
could pick from a five-item menu?* We will report that result honestly **even if D ties C.**

Condition B is the realistic industrial standard, not a straw man. Note that it detects
**zero** wrong-part, defective-part, and misaligned-seat faults — not because it is tuned
badly, but **by construction**: a gripper switch cannot perceive identity or alignment.

## Results

Pending. The metrics being collected, per condition and per fault type:

- Verifier precision and recall, per checkpoint and per failure type
- Rate at which the planner selects the correct recovery
- Guardrail overrides, and how often the verifier answered `unclear`
- False alarms on clean (`F0`) scenes, and unnecessary recoveries
- Verifier latency, p50 and p90
- Cost per 100 cycles — tokens x price, plus endpoint hours x rate

Tier 2 adds the same table for the fine-tuned model alongside the zero-shot one,
plus accuracy as a function of training-set size.

## Limitations

Stated up front because they are the honest frame for every number here.

- **Sim only.** Trained and evaluated entirely in Isaac Sim. Sim-to-real transfer is
  unproven and this is not evidence of real-world reliability.
- **No closed-loop physics.** The arm is not yet a physics-driven articulation in Isaac
  Sim; fault scenes are staged and rendered with ground truth, and recovery skills execute
  for real on the MoveIt2 stack. We therefore **do not** claim end-to-end cycle success
  rates, operator calls per 100 cycles, or quality-escape rates — those need the closed
  loop, which is future work. We report detection and decision quality, which is what this
  setup can honestly measure.
- **Self-reported VLM confidence is poorly calibrated.** It is logged but treated as a soft
  signal only, never as a gate.
- **Latency.** Verification happens only at checkpoints, never inside motion control.
  Added cycle time is reported, not hidden.

## Provenance

The underlying UR5e + Robotiq 2F-85 pick-and-place cell — MoveIt2 config, combined URDF,
`apriltag_ros` perception off an Isaac Sim camera, TF-based pose lookup — predates this
hackathon (July–September 2026) and lives in
[`ur5e-vision-pick-place`](https://github.com/Garvit-01/ur5e-vision-pick-place). Scene,
launch, and config files reused here are noted where they appear.

Everything else is new work for this hackathon: fault staging and ground truth, the
verifier and its fusion rules, the recovery planner and guardrails, the recovery skills,
the fine-tuning pipeline, the eval harness, and the dashboard.
