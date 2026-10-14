#!/usr/bin/env bash
# =========================================================================================
#  Compare vision models as the verifier, on the same scenes, prompt and scorer.
# =========================================================================================
#
# Runs every model you list through Nebius Token Factory on:
#   sim    the 120 held-out rendered scenes   (eval/bench_verifier.py)   -> runs/bench/feval_v1__<tag>
#   real   the 275 real MVTec AD photos        (eval/bench_realphoto.py)  -> runs/bench/mvtec__<tag>
#
#   ./eval/compare_models.sh                      # the default list below, sim + real
#   ./eval/compare_models.sh sim moonshotai/Kimi-K3 MiniMaxAI/MiniMax-M3
#   ./eval/compare_models.sh real Qwen/Qwen3.8-27B
#
# Costs Token Factory credit (roughly cents to ~$1 per model per set). Reasoning models think
# before answering, so replies get a 1,500-token budget; unreadable replies count as wrong.
# Model IDs: `.venv/bin/python scripts/smoke_nemotron.py --list`. Only "Public" vision models
# work per token; "Dedicated" ones need an endpoint (billed per hour).
#
# Our own fine-tuned model is not on Token Factory; run it locally on the T4:
#   ~/cosmos_t4/.venv/bin/python finetune/generate.py --data finetune/data/eval.jsonl \
#       --images runs/scenes --adapter finetune/adapter --quant 4bit --out <file>
#   then finetune/score_generations.py (sim) or eval/bench_realphoto.py score (real).
set -euo pipefail
cd "$(dirname "$0")/.."

WHAT="${1:-both}"; [ $# -gt 0 ] && shift
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(google/gemma-3-27b-it moonshotai/Kimi-K3 MiniMaxAI/MiniMax-M3 Qwen/Qwen3.8-27B
          deepseek-ai/DeepSeek-V4.1-Flash zai-org/GLM-5.3-Flash moonshotai/Kimi-K2.6)
fi
PY=.venv/bin/python

for m in "${MODELS[@]}"; do
  tag=$(echo "${m#*/}" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]/_/g')
  if [ "$WHAT" = sim ] || [ "$WHAT" = both ]; then
    echo "=== sim held-out: $m $(date -u +%H:%M) UTC"
    $PY eval/bench_verifier.py runs/scenes/feval_v1 --config config/selfheal_fastener.yaml --model "$m" \
        --max-tokens 1500 --workers 3 --out "runs/bench/feval_v1__$tag" 2>&1 | tail -1
  fi
  if [ "$WHAT" = real ] || [ "$WHAT" = both ]; then
    [ -f runs/mvtec/examples.jsonl ] || $PY eval/bench_realphoto.py build --mvtec runs/mvtec --out runs/mvtec/examples.jsonl
    echo "=== real photos: $m $(date -u +%H:%M) UTC"
    $PY eval/bench_realphoto.py api --examples runs/mvtec/examples.jsonl --images runs/mvtec --model "$m" \
        --max-tokens 1500 --out "runs/mvtec/gen_$tag.jsonl" 2>&1 | tail -1
    $PY eval/bench_realphoto.py score --examples runs/mvtec/examples.jsonl --generations "runs/mvtec/gen_$tag.jsonl" \
        --model "$m (zero-shot)" --out "runs/bench/mvtec__$tag" | tail -3
  fi
done
echo "=== done $(date -u +%H:%M) UTC"
