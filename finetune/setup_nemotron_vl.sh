#!/usr/bin/env bash
# Prepare the GPU VM to run NVIDIA Nemotron-Nano-12B-v2-VL zero-shot. Run ON the VM.
#
# Why a separate environment (~/nvl/.venv): the model's remote code pins transformers 4.53,
# while the Cosmos fine-tune environment (setup_vm.sh) uses a newer transformers.
#
# What it took to get working on the Nebius "Ubuntu 22.04 for NVIDIA GPUs (CUDA 12)" image,
# driver 550 / CUDA 12.4 (Oct 12, 2026):
#   - vLLM, the model card's fast path, needs a newer driver; transformers is used instead.
#   - causal-conv1d's newest release has no prebuilt wheel for torch 2.5 and failed to
#     compile from source, so both GPU kernels come as prebuilt wheels from GitHub releases.
#   - Triton compiles a small C helper at first use and needs the Python headers
#     (python3.10-dev), which the image doesn't ship.
#
# Downloads ~25 GB of weights from Hugging Face (public, no token). Idempotent.
set -euo pipefail

WORK="${WORK:-$HOME/nvl}"
MODEL="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
CONV1D="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
MAMBA="https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

mkdir -p "$WORK"
cd "$WORK"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "[nvl] $(date -u +%H:%M:%S) system: Python headers for Triton"
dpkg -s python3.10-dev >/dev/null 2>&1 || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.10-dev

echo "[nvl] $(date -u +%H:%M:%S) venv + packages"
[ -d .venv ] || uv venv --python 3.10 .venv
P=.venv/bin/python
uv pip install --python $P --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple "torch==2.5.1" "torchvision==0.20.1"
uv pip install --python $P "transformers>4.53,<4.54" timm accelerate open_clip_torch numpy pillow \
  einops "huggingface_hub>=0.25" hf_transfer
uv pip install --python $P --no-deps "$CONV1D" "$MAMBA"

echo "[nvl] $(date -u +%H:%M:%S) model download: $MODEL"
HF_HUB_ENABLE_HF_TRANSFER=1 $P - <<PY
from huggingface_hub import snapshot_download
print("model at", snapshot_download("$MODEL", allow_patterns=["*.json", "*.safetensors", "*.py", "*.jinja", "*.txt"]))
PY

$P -c "import torch, transformers, mamba_ssm, causal_conv1d; print('[nvl] OK', torch.__version__, transformers.__version__, mamba_ssm.__version__, causal_conv1d.__version__)"
echo "[nvl] $(date -u +%H:%M:%S) DONE"
