#!/bin/bash
# =============================================================================
# setup_env.sh — установка окружения для GSPO + Qwen3-Omni
# RTX PRO 6000 Blackwell (sm_120) — требует PyTorch nightly + cu128
# =============================================================================
# Использование:
#   bash setup_env.sh          # автоопределение
#   bash setup_env.sh cu128    # принудительно cu128 (Blackwell)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQS_FILE="${SCRIPT_DIR}/requirements2.txt"
VENV_DIR="${SCRIPT_DIR}/.venv3"

# ---------------------------------------------------------------------------
# 0. Активация venv3
# ---------------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  echo ">>> Создание виртуального окружения ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
fi

echo ">>> Активация ${VENV_DIR}..."
source "${VENV_DIR}/bin/activate"
echo "    Python: $(which python) — $(python --version)"

CUDA_TAG="${1:-}"

# ---------------------------------------------------------------------------
# 1. Найти CUDA_HOME и добавить nvcc в PATH
# ---------------------------------------------------------------------------
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _dir in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.4; do
    if [[ -f "${_dir}/bin/nvcc" ]]; then
      export CUDA_HOME="${_dir}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
      echo ">>> CUDA_HOME найден: ${CUDA_HOME}"
      break
    fi
  done
fi

# ---------------------------------------------------------------------------
# 2. Определить GPU capability и CUDA_TAG
# ---------------------------------------------------------------------------
if [[ -z "${CUDA_TAG}" ]]; then
  # Определяем compute capability GPU
  GPU_SM=$(python3 -c "
import subprocess, re
try:
    out = subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'], text=True).strip()
    major, minor = out.split('.')
    print(f'sm_{major}{minor}')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

  echo ">>> GPU compute capability: ${GPU_SM}"

  # Blackwell (sm_120+) требует nightly PyTorch
  if [[ "${GPU_SM}" == "sm_12"* || "${GPU_SM}" == "sm_13"* ]]; then
    echo ">>> Обнаружен Blackwell GPU (${GPU_SM}) — требует PyTorch nightly cu128"
    CUDA_TAG="cu128"
    USE_NIGHTLY=true
  elif command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    MINOR=$(echo "${CUDA_VER}" | cut -d. -f2)
    if   [[ "${MAJOR}" -ge 12 && "${MINOR}" -ge 8 ]]; then CUDA_TAG="cu128"
    elif [[ "${MAJOR}" -ge 12 && "${MINOR}" -ge 4 ]]; then CUDA_TAG="cu124"
    elif [[ "${MAJOR}" -ge 12 && "${MINOR}" -ge 1 ]]; then CUDA_TAG="cu121"
    else CUDA_TAG="cu124"; fi
    USE_NIGHTLY=false
  elif command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    if [[ "${MAJOR}" -ge 13 || "${MAJOR}" -ge 12 ]]; then CUDA_TAG="cu128"; else CUDA_TAG="cu124"; fi
    USE_NIGHTLY=false
  else
    echo "GPU не найден. CPU-режим."
    CUDA_TAG="cpu"
    USE_NIGHTLY=false
  fi
else
  # Тег передан явно — проверяем нужен ли nightly
  if [[ "${CUDA_TAG}" == "cu128" ]]; then
    USE_NIGHTLY=true
  else
    USE_NIGHTLY=false
  fi
fi

echo "========================================"
echo "  CUDA tag    : ${CUDA_TAG}"
echo "  Nightly      : ${USE_NIGHTLY}"
echo "  CUDA_HOME    : ${CUDA_HOME:-не задан}"
echo "========================================"

# ---------------------------------------------------------------------------
# 3. Обновить pip / setuptools
# ---------------------------------------------------------------------------
python -m pip install --upgrade pip "setuptools<81" wheel

# ---------------------------------------------------------------------------
# 4. PyTorch
# ---------------------------------------------------------------------------
echo ">>> Установка PyTorch..."
if [[ "${USE_NIGHTLY:-false}" == "true" ]]; then
  # Blackwell sm_120 поддерживается только в nightly
  echo ">>> Используем PyTorch nightly (единственная сборка с поддержкой sm_120)..."
  python -m pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
else
  python -m pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# Проверка что sm_120 теперь поддерживается
python -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'    GPU capability: sm_{cap[0]}{cap[1]} — поддерживается: {torch.cuda.is_available()}')
" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Flash-Attention
# ---------------------------------------------------------------------------
if [[ "${CUDA_TAG}" != "cpu" ]]; then
  # Убедимся что nvcc доступен
  if ! command -v nvcc &>/dev/null; then
    echo ">>> nvcc не найден — устанавливаем cuda-toolkit..."
    sudo apt-get install -y cuda-toolkit-12-8 2>/dev/null || true
    for _dir in /usr/local/cuda /usr/local/cuda-12.8; do
      if [[ -f "${_dir}/bin/nvcc" ]]; then
        export CUDA_HOME="${_dir}"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
        break
      fi
    done
  fi

  python -m pip install psutil

  if [[ "${USE_NIGHTLY:-false}" == "true" ]]; then
    echo ">>> flash-attn для Blackwell — компиляция из исходников..."
    echo "    (это займёт 20-30 минут, но нужна поддержка sm_120)"
    MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
  else
    echo ">>> flash-attn prebuilt wheel (torch 2.6.0 + cu124)..."
    FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu124torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    python -m pip install "${FLASH_WHEEL}" || {
      echo ">>> Prebuilt не подошёл, компилируем..."
      MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
    }
  fi
else
  echo ">>> Пропуск flash-attn (нет CUDA)."
fi

# ---------------------------------------------------------------------------
# 6. Остальные зависимости
# ---------------------------------------------------------------------------
echo ">>> Установка зависимостей из ${REQS_FILE}..."
python -m pip install -r "${REQS_FILE}"

# ---------------------------------------------------------------------------
# 7. bitsandbytes
# ---------------------------------------------------------------------------
echo ">>> Установка bitsandbytes..."
if [[ "${CUDA_TAG}" != "cpu" ]]; then
  python -m pip install "bitsandbytes>=0.44.0"
fi

# ---------------------------------------------------------------------------
# 8. vLLM (пропускаем если уже установлен)
# ---------------------------------------------------------------------------
if [[ "${CUDA_TAG}" != "cpu" ]]; then
  if python -c "import vllm" 2>/dev/null; then
    echo ">>> vLLM уже установлен — пропуск."
  else
    echo ">>> Установка vLLM..."
    python -m pip install "vllm>=0.8.0"
  fi
else
  echo ">>> Пропуск vLLM (нет CUDA)."
fi

# ---------------------------------------------------------------------------
# 9. Проверка
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Проверка установки"
echo "========================================"
python - <<'EOF'
import importlib, sys

checks = {
    "torch":         lambda m: f"v{m.__version__}, CUDA={m.cuda.is_available()}, GPUs={m.cuda.device_count()}, sm={m.cuda.get_device_capability() if m.cuda.is_available() else 'N/A'}",
    "transformers":  lambda m: f"v{m.__version__}",
    "swift":         lambda m: f"v{m.__version__}",
    "peft":          lambda m: f"v{m.__version__}",
    "deepspeed":     lambda m: f"v{m.__version__}",
    "accelerate":    lambda m: f"v{m.__version__}",
    "bitsandbytes":  lambda m: f"v{m.__version__}",
    "vllm":          lambda m: f"v{m.__version__}",
    "flash_attn":    lambda m: f"v{m.__version__}",
}

ok = True
for pkg, info_fn in checks.items():
    try:
        mod = importlib.import_module(pkg)
        print(f"  ✓  {pkg:<20} {info_fn(mod)}")
    except Exception as e:
        print(f"  ✗  {pkg:<20} ОШИБКА: {e}")
        if pkg not in ("flash_attn", "vllm"):
            ok = False

sys.exit(0 if ok else 1)
EOF

echo ""
echo "Окружение готово к обучению."