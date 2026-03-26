#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv4"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo ">>> Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo ">>> Python: $(which python) ($(python --version))"
python -m pip install --upgrade pip "setuptools<81" wheel

echo ">>> Installing PyTorch nightly cu128 (Blackwell)"
python -m pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

echo ">>> Installing inference dependencies"
python -m pip install \
  "vllm>=0.16.0" \
  "transformers>=4.57.0" \
  "openai>=1.0.0" \
  "huggingface_hub>=0.36.0" \
  "decord>=0.6.0"

echo ">>> Verifying environment"
python - <<'PY'
import importlib
import torch

print(f"torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}")
for pkg in ["vllm", "transformers", "openai", "huggingface_hub", "decord"]:
    mod = importlib.import_module(pkg)
    print(f"{pkg}={getattr(mod, '__version__', 'unknown')}")
PY

echo ">>> Environment .venv4 is ready"
