# Self-Healing Work Cell

**Apache-2.0** · Nebius x NVIDIA Global AI Hackathon 2026 · **Physical AI** track

A robot work cell that understands *what* went wrong, not just *that* something went wrong —
and a measurement of whether a small fine-tuned vision model does that job better than a
large zero-shot one.

> **Headline:** on 120 held-out simulated scenes, NVIDIA **Cosmos-Reason1-7B** fine-tuned on
> our rendered faults is right **94.2%** of the time (21.7% before fine-tuning, 75.8% for the best
> untrained frontier model), and with it **0 of 60 bad parts ship**, where today's
> gripper-sensor-only cell ships all 60. Every number here is reproducible from `runs/`;
> limitations are stated below.

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
(continue, regrasp, reseat, reject to bin, stop and alert). It never moves the robot and never
emits a coordinate; at most it names a region ("carrier" or "table").

```
Camera ──▶ Verifier (VLM) ──▶ fuse with gripper signal ──▶ Nemotron picks 1 action ──▶ guardrails
```

In the benchmark each chosen action is scored against what should have happened. Executing
the actions with a physics-driven arm is being added (a sorting line in Isaac Sim).

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
  LoRA fine-tuned on 1,800 staged faults on a **Nebius AI Cloud** H100 VM (97 minutes), then
  compressed to 4-bit so it also runs on a 15 GB T4, the kind of GPU a plant could host.

Same held-out eval split, same prompt and same scorer for every model, so the columns are
comparable.

## How Nebius and NVIDIA models are used

| Component | What | Where it runs |
|---|---|---|
| Recovery planner | NVIDIA **Nemotron**, tool calling, temperature 0 | **Nebius Token Factory** |
| Verifier (Tier 1) | Gemma 3 27B, zero-shot (the baseline we built the pipeline on) | **Nebius Token Factory** |
| Verifier (Tier 2) | NVIDIA **Cosmos-Reason1-7B** + our LoRA adapter | trained and evaluated on a **Nebius AI Cloud** VM (1x H100); runs 4-bit on a local T4 |
| Comparison verifiers | every per-token vision model on Token Factory, plus NVIDIA Nemotron-Nano-12B-v2-VL | **Nebius Token Factory**; Nemotron-VL on the Nebius VM |
| Fault-scene rendering | Isaac Sim 5.1.0 | local GPU (Tesla T4) |

Model IDs and base URLs are **never hardcoded** — they resolve from environment variables
via [`config/selfheal.yaml`](config/selfheal.yaml), which is what makes the Tier 1 → Tier 2
swap a config change. Always re-pin IDs from the live catalogue before a run:

```bash
.venv/bin/python scripts/smoke_nemotron.py --list
```

## Setup

What each part needs:

| Part | Needs |
|---|---|
| Calling the models, scoring, reports, tests | Python 3.10+ and a Nebius Token Factory key. Any computer. |
| Rendering scenes (`isaac/`) | Isaac Sim 5.1 and an NVIDIA RTX GPU (we used a T4) |
| Fine-tuning (`finetune/`) | a GPU VM; we used one H100 80 GB on Nebius AI Cloud |

```bash
git clone git@github.com:Garvit-01/nvidia_nebius_challenge_ws.git
cd nvidia_nebius_challenge_ws

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
$EDITOR .env          # add your Nebius Token Factory key and the model IDs
```

## Run

`make help` lists everything, with time and cost. The main entry points:

```bash
make test        # all unit tests, no key or GPU needed
make smoke       # one live call each to the planner (Nemotron) and the verifier
make demo        # render one scene of each fault type (Isaac Sim)
make view SEED=1031   # open one scene in Isaac Sim with a window
make eval        # render 120 held-out scenes, score the verifier, compare policies A-D, PDF
```

The fine-tune is driven end to end by `finetune/run_finetune.sh`; see `finetune/README.md`.

## Experimental conditions

| Cond | Name | Checks | On failure |
|---|---|---|---|
| A | no recovery | none | proceeds blindly |
| B | **industrial baseline** | gripper signal at post-lift and post-transport only | retry pick at original location 2x, then stop |
| C | **ours** | verifier at all 4 checkpoints + sensor fusion | Nemotron picks one whitelisted action, then guardrails |
| D | ablation | same verifier as C | fixed `failure_type → action` table, no LLM |

Condition D exists to answer the obvious objection: *why an LLM, when a switch statement
could pick from a five-item menu?* We will report that result honestly **even if D ties C.**

Condition B is the realistic industrial standard, not a straw man. Note that it detects
**zero** wrong-part, defective-part, and misaligned-seat faults — not because it is tuned
badly, but **by construction**: a gripper switch cannot perceive identity or alignment.

## Results

All on the **120 held-out scenes** (seeds 5000-5119): M16 nuts and their faults, never used for
training or for tuning prompts. 60 of them hold a bad part (wrong size, M16 bolt, chipped,
cracked, rusty, or seated crooked), 20 are clean, the rest are handling faults (displaced,
dropped, missed grasp). Same prompt, same two camera views, same parser and scorer for every row.

**The verifier: is it right about what's wrong?**

| Verifier | Correct | False alarms on clean parts | Faults missed |
|---|---|---|---|
| **Cosmos-Reason1-7B + our LoRA** (H100) | **94.2%** | 25% | **0%** |
| … same, 4-bit on a T4 | 93.3% | 20% | 2% |
| MiniMax-M3, zero-shot | 75.8% | 5% | 16% |
| Kimi-K3, zero-shot | 72.5% | 5% | 16% |
| Qwen3.8-27B, zero-shot | 66.7% | 0% | 22% |
| Gemma 3 27B, zero-shot | 48.3% | 0% | 52% |
| NVIDIA Nemotron-Nano-12B-v2-VL, zero-shot | 38.3% | 55% | 38% |
| NVIDIA Cosmos-Reason1-7B, zero-shot | 21.7% | 50% | 30% |

Untrained models cannot tell an M16 nut from an M12 or M20, and miss 4 mm chips; a second
prompt version aimed at exactly those changed nothing (0/6 wrong-size nuts, 0/4 chips on dev),
so it is a perception gap, not a wording one. Fine-tuning on
1,800 rendered scenes (51.5M trainable parameters, 0.6% of the model) closes it.

**The whole loop: how many bad parts reach the fixture?** (60 bad parts in the 120 scenes)

| Condition | Correct decisions | Bad parts shipped |
|---|---|---|
| A · no checks | 16.7% | 60 |
| B · industrial baseline (gripper sensor only) | 21.7% | 60 |
| C · ours, Gemma as verifier | 48.3% | 44 |
| **C · ours, fine-tuned Cosmos as verifier** | **95.8%** | **0** |

The cost: 5 of 20 clean nuts were flagged and sent to the bin.

**What didn't help:** the planner. Four Nemotron sizes (Nano 30B, Super 120B, Ultra 550B,
3.5-Lightning) made the same decision as a fixed lookup table (condition D) on all 180 scenes
(dev + held-out). The verifier's accuracy is what moves the result; we report that plainly.

## Limitations

Stated up front because they are the honest frame for every number here.

- **Sim only.** Trained and evaluated on rendered scenes; train and test come from the same
  simulator, camera and parts. 94% shows the model can learn the cues untrained models miss;
  it is not evidence of real-world reliability. We checked: on 275 real factory photos (MVTec
  AD, metal nuts and screws) the fine-tuned model calls almost everything defective (91-100% of
  good parts flagged), while untrained general models catch about half the real damage with
  few false alarms. Closing that gap needs real images in the training mix.
- **Visible damage only.** The single inspection camera only sees damage on the faces it
  looks at, so the benchmark only places damage there. Underside damage needs a second camera
  (in progress).
- **Snapshots, not a moving robot.** Each scene is one staged moment: the recovery tools are
  chosen and scored, not executed. We therefore **do not** claim cycle success rates or
  operator calls per 100 cycles. A physics-driven sorting arm is being added.
- **Self-reported VLM confidence is poorly calibrated.** It is logged but treated as a soft
  signal only, never as a gate.
- **Latency.** Verification happens only at checkpoints, never inside motion control.
  Added cycle time is reported, not hidden.

## Provenance

This project builds on an **existing** robot cell rather than starting from nothing, and the
split matters, so here it is explicitly.

**Pre-existing.** The UR5e + Robotiq 2F-85 pick-and-place cell — MoveIt2 configuration,
combined URDF, `apriltag_ros` perception off an Isaac Sim camera, and TF-based pose lookup —
was built between **July and September 2026**, beginning before this hackathon's submission
period opened on August 26, 2026. It lives in a separate repository,
[`ur5e-vision-pick-place`](https://github.com/Garvit-01/ur5e-vision-pick-place). Scene,
launch, and configuration files reused here are noted where they appear.

**New for this hackathon**, all of it after October 10, 2026: the fault staging and
ground-truth generation (cubes, then NVIDIA's M16 nuts and bolts with chips, cracks and rust),
the VLM verifier and its sensor-fusion rules, the recovery planner and the code-level
guardrails, the LoRA fine-tuning pipeline, and the evaluation harness. The dated commit history of both repositories
reflects this.
