#!/bin/bash
# =============================================================================
# setup_env.sh — установка окружения для GSPO + Qwen3-Omni
# =============================================================================
# Использование:
#   bash setup_env.sh            # автоопределение CUDA
#   bash setup_env.sh cu118      # принудительно CUDA 11.8
#   bash setup_env.sh cu124      # принудительно CUDA 12.4
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQS_FILE="${SCRIPT_DIR}/requirements2.txt"

CUDA_TAG="${1:-}"

# ---------------------------------------------------------------------------
# 1. Определить версию CUDA
# ---------------------------------------------------------------------------
if [[ -z "${CUDA_TAG}" ]]; then
  if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    MINOR=$(echo "${CUDA_VER}" | cut -d. -f2)
    if   [[ "${MAJOR}" -ge 12 && "${MINOR}" -ge 8 ]]; then CUDA_TAG="cu128"
    elif [[ "${MAJOR}" -ge 12 && "${MINOR}" -ge 4 ]]; then CUDA_TAG="cu124"
    elif [[ "${MAJOR}" -ge 12 && "${MINOR}" -ge 1 ]]; then CUDA_TAG="cu121"
    elif [[ "${MAJOR}" -ge 11 && "${MINOR}" -ge 8 ]]; then CUDA_TAG="cu118"
    else
      echo "Предупреждение: CUDA ${CUDA_VER} не протестирована. Используем cu124."
      CUDA_TAG="cu124"
    fi
  else
    echo "nvcc не найден. Устанавливаем CPU-версию PyTorch (только для теста)."
    CUDA_TAG="cpu"
  fi
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
echo "========================================"
echo "  CUDA tag : ${CUDA_TAG}"
echo "  Index URL: ${TORCH_INDEX}"
echo "========================================"

# ---------------------------------------------------------------------------
# 2. Обновить pip / setuptools
# ---------------------------------------------------------------------------
python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. PyTorch (должен идти ДО остальных пакетов)
# ---------------------------------------------------------------------------
echo ">>> Установка PyTorch..."
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"

# ---------------------------------------------------------------------------
# 4. Flash-Attention (требует уже установленного PyTorch + CUDA headers)
# ---------------------------------------------------------------------------
if [[ "${CUDA_TAG}" != "cpu" ]]; then
  echo ">>> Установка flash-attn..."
  pip install flash-attn --no-build-isolation
else
  echo ">>> Пропуск flash-attn (нет CUDA)."
fi

# ---------------------------------------------------------------------------
# 5. Остальные зависимости
# ---------------------------------------------------------------------------
echo ">>> Установка зависимостей из ${REQS_FILE}..."
pip install -r "${REQS_FILE}"

# ---------------------------------------------------------------------------
# 5а. bitsandbytes — отдельно, нужна версия совместимая с CUDA
# ---------------------------------------------------------------------------
echo ">>> Установка bitsandbytes для QLoRA..."
if [[ "${CUDA_TAG}" == "cu128" ]]; then
  pip install "bitsandbytes>=0.44.0"
elif [[ "${CUDA_TAG}" != "cpu" ]]; then
  pip install "bitsandbytes>=0.43.0"
fi

# ---------------------------------------------------------------------------
# 6. Проверка
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Проверка установки"
echo "========================================"
python - <<'EOF'
import importlib, sys

checks = {
    "torch":         lambda m: f"v{m.__version__}, CUDA={m.cuda.is_available()}, GPUs={m.cuda.device_count()}",
    "transformers":  lambda m: f"v{m.__version__}",
    "swift":         lambda m: f"v{m.__version__}",
    "peft":          lambda m: f"v{m.__version__}",
    "deepspeed":     lambda m: f"v{m.__version__}",
    "accelerate":    lambda m: f"v{m.__version__}",
    "bitsandbytes":  lambda m: f"v{m.__version__}",
    "flash_attn":    lambda m: f"v{m.__version__}",
}

ok = True
for pkg, info_fn in checks.items():
    try:
        mod = importlib.import_module(pkg)
        print(f"  ✓  {pkg:<20} {info_fn(mod)}")
    except Exception as e:
        print(f"  ✗  {pkg:<20} ОШИБКА: {e}")
        if pkg not in ("flash_attn",):   # flash_attn опционален на CPU
            ok = False

sys.exit(0 if ok else 1)
EOF

echo ""
echo "Окружение готово к обучению."
