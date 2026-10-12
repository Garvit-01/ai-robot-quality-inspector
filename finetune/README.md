# How the fine-tune works

The verifier is a vision-language model: it looks at two camera images and answers one
question as JSON. Zero-shot, NVIDIA's Cosmos-Reason1-7B got **21.7%** of held-out fastener
scenes right (Gemma 3 27B: 48.3%). Fine-tuning teaches it this cell, these parts and these
faults, from scenes the simulator labelled for free.

**Run it:** `./finetune/run_finetune.sh all` — or one step at a time. The script is
commented step by step; read it top to bottom.

## The pieces

| File | Where it runs | What it does |
|---|---|---|
| `run_finetune.sh` | here | the whole procedure, step by step |
| `build_dataset.py` | here | rendered scenes → training examples (JSONL) |
| `setup_vm.sh` | GPU VM | install PyTorch + Hugging Face libraries, download the model |
| `train_lora.py` | GPU VM | the training loop |
| `generate.py` | GPU VM | run the model (with or without the fine-tune) on held-out scenes |
| `score_generations.py` | here | score the replies with the benchmark's own scorer |

## 1. What one training example is

```json
{"system": "You are a quality and process verifier for a robot work cell...",
 "user":   "Checkpoint: CP_PRE_GRASP. The robot is about to grasp the part... (the full question)",
 "images": ["ftrain_v1/images/10004_overview.png", "ftrain_v1/images/10004_closeup.png"],
 "target": "{\"checkpoint\": \"CP_PRE_GRASP\", \"step_ok\": false, \"failure_type\": \"wrong_part\",
             \"evidence\": \"The nut is too big: it covers the printed outline on several sides.\",
             \"location_hint\": \"carrier\", \"confidence\": 0.9}"}
```

- `system` and `user` are **exactly** what the verifier is sent in production
  (`eval/bench_verifier.py`): the model is trained on the real task.
- `target` is the correct answer. The simulator placed the M20 nut, so it *knows*.
  The `evidence` sentence is generated from the ground truth (`build_dataset.py:evidence`)
  so the model learns to name the cue that matters, then decide.

Splits never overlap: training seeds 10000–11799 (1,800 scenes), validation 2000–2059,
held-out evaluation 5000–5119. The held-out scenes are never trained on.

## 2. What LoRA is, and what gets trained

The model has 8.3 billion parameters. Updating all of them needs ~4× the GPU memory and
lots of data. **LoRA** (Low-Rank Adaptation) freezes them all and adds, next to chosen
weight matrices `W`, two small matrices `A` (r × d) and `B` (d × r) with rank `r = 16`.
The layer computes `W·x + B·A·x`; only `A` and `B` are trained. Result here: **51.5 M
trainable parameters, 0.6%** of the model, and the original weights are untouched — the
"fine-tune" is a ~200 MB adapter file.

Where the adapters go (`train_lora.py`, `TARGETS`):
- the **language model**'s attention and MLP projections (how it reasons and answers);
- the **vision encoder**'s blocks too — the zero-shot failures (a nut's size, a 4 mm chip,
  a nut caught on a rim) are *seeing* problems, so the vision side must be able to adapt.

## 3. The training loop (`train_lora.py`)

For each example:
1. Build the conversation (system, user with two images, assistant answer) with the
   model's own chat template, and turn images into visual tokens at full 1024×768.
2. **Mask the prompt**: labels are `-100` (ignored) for every token before the answer, so
   the loss only measures how well it writes the correct verdict.
3. Forward pass → cross-entropy loss → backward.
4. Every 8 examples (gradient accumulation = an effective batch of 8): clip gradients,
   AdamW step, learning-rate schedule (1e-4 peak, warm-up then cosine decay).

Logged every 10 examples: loss, seconds per example, ETA, peak GPU memory. Validation loss
and a checkpoint after each epoch. bf16 precision and gradient checkpointing keep memory
at ~22 GB of the H100's 80 GB.

This is **supervised fine-tuning** — one correct answer per image, known for free — not
RLHF: there are no preferences to learn, and no reward model is needed.

## 4. Evaluation — same test, same scorer

`generate.py` runs the model greedily (no randomness) on the 120 held-out scenes, once
without the adapter (baseline) and once with it. `score_generations.py` parses the replies
with the **same** parser and scores them with the **same** code as every Gemma run, so the
numbers sit side by side in `reports/` and in the A/B/C/D policy comparison.

## Hardware and cost

1× NVIDIA H100 80 GB on Nebius AI Cloud ($4.52/h, eu-north1). Setup ~5 min; ~2.5 s per
training example. Not a competition requirement — the runtime Nebius calls are what's
required — but training on Nebius AI Cloud is part of the project's Nebius story.
