#!/usr/bin/env bash
# Prepare a fresh GPU VM for the Tier 2 fine-tune. Run ON the VM, as the login user.
#
# Tested on Nebius AI Cloud: 1x H100 80 GB, "Ubuntu 22.04 LTS for NVIDIA GPUs (CUDA 12)",
# which ships CUDA 12.4 and drivers but no Python packages (no torch, not even pip).
#
# Installs uv, a Python venv with PyTorch + Hugging Face libraries, and downloads
# nvidia/Cosmos-Reason1-7B (~16 GB, not gated on Hugging Face). Idempotent: re-running
# skips what is already there.
set -euo pipefail

WORK="${WORK:-$HOME/selfheal}"
MODEL="${MODEL:-nvidia/Cosmos-Reason1-7B}"
mkdir -p "$WORK"
cd "$WORK"

echo "[setup] $(date -u +%H:%M:%S) uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] $(date -u +%H:%M:%S) venv + packages"
[ -d .venv ] || uv venv --python 3.10 .venv
# torch built for CUDA 12.4, matching the image's driver stack.
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple \
  "torch==2.5.1" "torchvision==0.20.1"
uv pip install --python .venv/bin/python \
  "transformers>=4.51,<4.56" "peft>=0.13" "accelerate>=1.0" "qwen-vl-utils>=0.0.8" \
  "pillow" "pydantic>=2" "huggingface_hub>=0.25" "hf_transfer"

echo "[setup] $(date -u +%H:%M:%S) model download: $MODEL"
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
print("model at", p)
PY

.venv/bin/python - <<'PY'
import torch, transformers, peft
print(f"[setup] torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"gpu={torch.cuda.get_device_name(0)} | transformers {transformers.__version__} | peft {peft.__version__}")
PY
echo "[setup] $(date -u +%H:%M:%S) DONE"
