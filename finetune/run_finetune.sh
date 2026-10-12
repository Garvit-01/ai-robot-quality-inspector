#!/usr/bin/env bash
# =========================================================================================
#  Tier 2: fine-tune NVIDIA Cosmos-Reason1-7B as the work-cell verifier — end to end.
# =========================================================================================
#
# Run from the project root on THIS machine (the one with Isaac Sim). The heavy lifting
# happens on a rented GPU VM reached over SSH; this machine renders scenes and scores.
#
#   ./finetune/run_finetune.sh all            # every step below, in order
#   ./finetune/run_finetune.sh train          # or any single step by name
#
# Steps:
#   data      build training / validation / evaluation examples from rendered scenes
#   push      copy the examples, their images and the scripts to the VM
#   setup     install PyTorch + Hugging Face libraries on the VM, download the model
#   baseline  score the model BEFORE fine-tuning on the held-out scenes
#   train     LoRA fine-tune on the training scenes
#   dev       score the fine-tuned model on the 60 DEV scenes — use this to compare runs
#   eval      score the fine-tuned model on the same held-out scenes — only for the final pick
#   pull      bring the adapter and the raw replies back here
#   score     parse + score both runs exactly like every other benchmark, A/B/C/D, PDF
#   nemotron  (optional) NVIDIA Nemotron-Nano-12B-v2-VL zero-shot on the held-out scenes, scored
#             the same way. Own environment on the VM (setup_nemotron_vl.sh), ~25 GB download,
#             ~30 s per scene on one H100
#
# What you need first:
#   - rendered scenes:  runs/scenes/ftrain_v1 (training), fdev_v2 (validation), feval_v1 (held out)
#   - a GPU VM you can SSH into as $VM_USER@$VM_IP (Nebius AI Cloud, 1x H100 80 GB here)
#
# Cost control: the first thing `setup` does on the VM is schedule an automatic shutdown,
# so a stuck job can't run up a bill. Delete the VM when finished.
set -euo pipefail

VM_IP="${VM_IP:-89.169.114.255}"            # the VM's public IP
VM_USER="${VM_USER:-ubuntu}"
VM_DIR="${VM_DIR:-selfheal}"              # working directory on the VM (under ~)
SHUTDOWN_MIN="${SHUTDOWN_MIN:-240}"       # safety cap: VM powers off after this many minutes
CONFIG="config/selfheal_fastener.yaml"    # the verifier prompt the model is trained on
EPOCHS="${EPOCHS:-1.5}"                   # passes over the training set

TRAIN=runs/scenes/ftrain_v1   # 1,800 scenes, seeds 10000-11799
VAL=runs/scenes/fdev_v2       #    60 scenes, seeds  2000-2059   (validation loss only)
EVAL=runs/scenes/feval_v1     #   120 scenes, seeds  5000-5119   (never trained on)
DATA=finetune/data
PY=.venv/bin/python

SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 $VM_USER@$VM_IP"
RSYNC="rsync -a -e 'ssh -o BatchMode=yes'"
vm() { $SSH "cd ~/$VM_DIR && $*" < /dev/null; }   # run a command on the VM, in the work dir
say() { echo; echo "=== $(date -u +%H:%M:%S) $* ==="; }

# Long GPU jobs run detached on the VM (nohup), so a dropped SSH connection can't kill them;
# this machine polls the log until the job's process is gone, showing progress lines.
# Usage: vm_job <log file> <command>. Returns non-zero if there is no log or it shows a
# Python traceback. The command is passed as plain words: no quoted arguments.
vm_job() {
  local log="$1"; shift
  vm "nohup $* > $log 2>&1 &"
  sleep 15   # let the process start before the first check
  local last=""
  # [p]ython: the bracket stops pgrep matching this polling command's own shell.
  while vm "pgrep -f '[p]ython.* $(echo "$*" | awk '{print $2}')' > /dev/null"; do
    sleep 60
    local now; now=$(vm "grep -E '^\[(train|gen)\]' $log | tail -1" || true)
    [ "$now" != "$last" ] && echo "  $now" && last="$now"
  done
  vm "grep -E '^\[(train|gen)\]' $log | tail -2"
  vm "test -s $log && ! grep -q Traceback $log"   # the job ran and didn't crash
}

# -----------------------------------------------------------------------------------------
# 1. DATA — one JSON line per scene: the exact prompt the verifier sees, the two image
#    paths, and the correct answer as Verdict JSON. Nothing is hand-labelled: the
#    simulator placed every part, so it knows the answer. (finetune/build_dataset.py)
# -----------------------------------------------------------------------------------------
step_data() {
  say "data: building examples"
  $PY finetune/build_dataset.py "$TRAIN" --config "$CONFIG" --out "$DATA/train.jsonl"
  $PY finetune/build_dataset.py "$VAL"   --config "$CONFIG" --out "$DATA/dev.jsonl"
  $PY finetune/build_dataset.py "$EVAL"  --config "$CONFIG" --out "$DATA/eval.jsonl"
  echo "one training example's ANSWER (what the model learns to write):"
  head -1 "$DATA/train.jsonl" | $PY -c "import json,sys; print('  ', json.loads(sys.stdin.read())['target'])"
}

# -----------------------------------------------------------------------------------------
# 2. PUSH — the VM needs no project code: just the examples, their PNGs, and the three
#    scripts. Images keep their relative paths (<scene set>/images/<seed>_<view>.png).
# -----------------------------------------------------------------------------------------
step_push() {
  say "push: examples, images and scripts -> $VM_IP"
  $SSH "mkdir -p ~/$VM_DIR/scenes" < /dev/null
  eval $RSYNC finetune/setup_vm.sh finetune/train_lora.py finetune/generate.py \
       "$DATA/train.jsonl" "$DATA/dev.jsonl" "$DATA/eval.jsonl" "$VM_USER@$VM_IP:~/$VM_DIR/"
  for d in "$TRAIN" "$VAL" "$EVAL"; do   # PNGs only, not labels: answers live in the JSONL
    eval $RSYNC --include="'*/'" --include="'*.png'" --exclude="'*'" "$d" "$VM_USER@$VM_IP:~/$VM_DIR/scenes/"
  done
  vm "du -sh scenes; ls"
}

# -----------------------------------------------------------------------------------------
# 3. SETUP — safety shutdown first, then Python + PyTorch + transformers/peft and the model
#    (~16 GB). finetune/setup_vm.sh is idempotent: re-running skips what is there.
# -----------------------------------------------------------------------------------------
step_setup() {
  say "setup: safety shutdown in $SHUTDOWN_MIN min, then libraries + model"
  $SSH "sudo shutdown -h +$SHUTDOWN_MIN 'selfheal safety cap'" < /dev/null
  vm "WORK=~/$VM_DIR bash setup_vm.sh"
}

# -----------------------------------------------------------------------------------------
# 4. BASELINE — the same model with NO fine-tuning, on the held-out scenes. Without this,
#    an improvement could just mean "NVIDIA's model was already good". (generate.py)
# -----------------------------------------------------------------------------------------
step_baseline() {
  say "baseline: base Cosmos-Reason1-7B on $(wc -l < "$DATA/eval.jsonl") held-out scenes"
  vm ".venv/bin/python generate.py --data eval.jsonl --images scenes --out gen_base.jsonl"
}

# -----------------------------------------------------------------------------------------
# 5. TRAIN — LoRA: the 7B model's own weights stay frozen; small low-rank "adapter"
#    matrices (0.6% of the parameters) are added to the language model AND the vision
#    encoder and trained. The loss counts only the answer tokens — the prompt and images
#    are context. Logged every 10 examples: loss, seconds per example, ETA, GPU memory.
#    A checkpoint is saved after every full epoch. (finetune/train_lora.py)
# -----------------------------------------------------------------------------------------
step_train() {
  say "train: LoRA, $EPOCHS epochs over $(wc -l < "$DATA/train.jsonl") examples"
  vm_job train.log .venv/bin/python train_lora.py --data train.jsonl --val dev.jsonl --images scenes \
        --out adapter --epochs "$EPOCHS"
}

# -----------------------------------------------------------------------------------------
# 6a. DEV — the fine-tuned model on the 60 validation scenes. If you train more than once
#     (different epochs, rank, data), pick the winner HERE. Picking by the held-out score
#     would turn it into "best of N tries on the test" and overstate the result.
# -----------------------------------------------------------------------------------------
step_dev() {
  say "dev: fine-tuned model on the dev scenes (for choosing between runs)"
  vm_job gen_tuned_dev.log .venv/bin/python generate.py --data dev.jsonl --images scenes --adapter adapter --out gen_tuned_dev.jsonl
  eval $RSYNC "$VM_USER@$VM_IP:~/$VM_DIR/gen_tuned_dev.jsonl" "$DATA/gen/"
  $PY finetune/score_generations.py "$DATA/dev.jsonl" "$DATA/gen/gen_tuned_dev.jsonl" \
      --scene-dir "$VAL" --model "Cosmos-Reason1-7B + LoRA (dev)" --out runs/bench/fdev_v2__cosmos_tuned
}

# -----------------------------------------------------------------------------------------
# 6b. EVAL — the fine-tuned model (base + adapter) on the SAME held-out scenes.
# -----------------------------------------------------------------------------------------
step_eval() {
  say "eval: fine-tuned model on the held-out scenes"
  vm_job gen_tuned.log .venv/bin/python generate.py --data eval.jsonl --images scenes --adapter adapter --out gen_tuned.jsonl
}

# -----------------------------------------------------------------------------------------
# 7. PULL — the adapter (~200 MB) and the raw replies come home; the VM can then go.
# -----------------------------------------------------------------------------------------
step_pull() {
  say "pull: adapter + replies"
  mkdir -p "$DATA/gen" finetune/adapter
  eval $RSYNC "$VM_USER@$VM_IP:~/$VM_DIR/gen_*.jsonl" "$VM_USER@$VM_IP:~/$VM_DIR/train.log" "$DATA/gen/"
  eval $RSYNC "$VM_USER@$VM_IP:~/$VM_DIR/adapter/" finetune/adapter/
  ls -la "$DATA/gen" finetune/adapter | head -20
}

# -----------------------------------------------------------------------------------------
# 8. SCORE — parse and score both runs with the benchmark's own code, then run the A/B/C/D
#    recovery-policy comparison on the fine-tuned verdicts and rebuild the PDF report.
# -----------------------------------------------------------------------------------------
step_score() {
  say "score"
  $PY finetune/score_generations.py "$DATA/eval.jsonl" "$DATA/gen/gen_base.jsonl" \
      --scene-dir "$EVAL" --model "nvidia/Cosmos-Reason1-7B (base, zero-shot)" --out runs/bench/feval_v1__cosmos_base
  $PY finetune/score_generations.py "$DATA/eval.jsonl" "$DATA/gen/gen_tuned.jsonl" \
      --scene-dir "$EVAL" --model "nvidia/Cosmos-Reason1-7B + LoRA (ours)" --out runs/bench/feval_v1__cosmos_tuned
  $PY eval/bench_decisions.py "$EVAL" runs/bench/feval_v1__cosmos_tuned --config "$CONFIG" \
      --out runs/decisions/feval_v1__cosmos_tuned
  $PY eval/make_report.py runs/bench/feval_v1__fastener-v1 runs/bench/feval_v1__cosmos_base \
      runs/bench/feval_v1__cosmos_tuned --decisions runs/decisions/feval_v1__cosmos_tuned \
      --out reports/evaluation_report_finetune.pdf
}

# -----------------------------------------------------------------------------------------
# 9. NEMOTRON (optional) — NVIDIA's vision Nemotron, zero-shot, on the same held-out scenes.
#    Token Factory's per-token Nemotrons are text-only, so it runs on this VM. Needs `push`.
# -----------------------------------------------------------------------------------------
step_nemotron() {
  say "nemotron: Nemotron-Nano-12B-v2-VL zero-shot on the held-out scenes"
  eval $RSYNC finetune/setup_nemotron_vl.sh finetune/generate_nemotron_vl.py "$VM_USER@$VM_IP:~/$VM_DIR/"
  vm "bash setup_nemotron_vl.sh"
  vm_job gen_nemotron_vl.log ~/nvl/.venv/bin/python generate_nemotron_vl.py --data eval.jsonl --images scenes \
      --out gen_nemotron_vl.jsonl
  mkdir -p "$DATA/gen"
  eval $RSYNC "$VM_USER@$VM_IP:~/$VM_DIR/gen_nemotron_vl.jsonl" "$DATA/gen/"
  $PY finetune/score_generations.py "$DATA/eval.jsonl" "$DATA/gen/gen_nemotron_vl.jsonl" \
      --scene-dir "$EVAL" --model "nvidia/Nemotron-Nano-12B-v2-VL (zero-shot)" --out runs/bench/feval_v1__nemotron_vl
}

case "${1:-}" in
  all) step_data; step_push; step_setup; step_baseline; step_train; mkdir -p "$DATA/gen"; step_dev; step_eval; step_pull; step_score ;;
  data|push|setup|baseline|train|dev|eval|pull|score|nemotron) "step_$1" ;;
  *) sed -n '2,31p' "$0"; exit 1 ;;
esac
