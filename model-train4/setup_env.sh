#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv5"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo ">>> Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo ">>> Python: $(which python) ($(python --version))"
python -m pip install --upgrade pip "setuptools<81" wheel

TRL_VERSION="0.24.0"
DATASETS_VERSION="4.3.0"
PYARROW_SPEC="pyarrow>=21,<24"
TRANSFORMERS_VERSION="5.2.0"

echo ">>> Installing PyTorch nightly cu128 for Blackwell"
python -m pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

echo ">>> Installing Unsloth + RL/VLM stack"
python -m pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
python -m pip install --upgrade \
  "trl==${TRL_VERSION}" \
  "transformers==${TRANSFORMERS_VERSION}" \
  accelerate \
  "datasets==${DATASETS_VERSION}" \
  "${PYARROW_SPEC}" \
  peft \
  bitsandbytes \
  sentencepiece \
  protobuf \
  pillow \
  numpy \
  scipy \
  mergekit \
  orjson \
  tqdm \
  psutil \
  boto3 \
  ffmpeg-python \
  librosa \
  soundfile \
  faster-whisper \
  inaSpeechSegmenter \
  diffusers

echo ">>> Verifying environment"
python - <<'PY'
import importlib
import torch

print(f"torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}")
for pkg in [
    "unsloth",
    "trl",
    "accelerate",
    "datasets",
    "peft",
    "bitsandbytes",
    "librosa",
    "faster_whisper",
    "inaSpeechSegmenter",
]:
    mod = importlib.import_module(pkg)
    print(f"{pkg}={getattr(mod, '__version__', 'unknown')}")
PY
python -m pip check

echo ">>> Environment is ready"
