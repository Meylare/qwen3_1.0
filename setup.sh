#!/bin/bash
# =============================================================================
# setup.sh — полная установка окружения для virality-grpo на AWS
# Ubuntu 22.04 + NVIDIA GPU (A100 / H100 / A10G)
#
# Запуск:
#   chmod +x setup.sh && ./setup.sh
#
# Или по шагам: bash -x setup.sh  (с выводом каждой команды)
# =============================================================================

set -e  # останавливаемся при любой ошибке
echo "=== [0] Проверка GPU ==="
nvidia-smi || { echo "ОШИБКА: nvidia-smi не найден. Проверь драйверы."; exit 1; }
nvcc --version || echo "nvcc не в PATH — это нормально для DL AMI (CUDA всё равно есть)"

CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
echo "Обнаружен CUDA: $CUDA_VER"

# =============================================================================
echo ""
echo "=== [1] Системные пакеты ==="
# =============================================================================
sudo apt-get update -q
sudo apt-get install -y \
    ffmpeg \
    git \
    git-lfs \
    wget \
    curl \
    tmux \
    htop \
    nvtop \
    python3-pip \
    python3-venv \
    build-essential

git lfs install
echo "✓ Системные пакеты установлены"

# =============================================================================
echo ""
echo "=== [2] Python venv ==="
# =============================================================================
# Создаём venv в домашней директории — не загрязняем system python
python3 -m venv /home/ubuntu/venv
source /home/ubuntu/venv/bin/activate

# Обновляем pip внутри venv
pip install --upgrade pip setuptools wheel
echo "✓ venv создан: /home/ubuntu/venv"
echo "  Чтобы активировать: source /home/ubuntu/venv/bin/activate"

# Добавляем автоактивацию в .bashrc (только если ещё нет)
if ! grep -q "venv/bin/activate" /home/ubuntu/.bashrc; then
    echo "" >> /home/ubuntu/.bashrc
    echo "# Auto-activate virality-grpo venv" >> /home/ubuntu/.bashrc
    echo "source /home/ubuntu/venv/bin/activate" >> /home/ubuntu/.bashrc
    echo "✓ Автоактивация добавлена в .bashrc"
fi

# =============================================================================
echo ""
echo "=== [3] PyTorch ==="
# =============================================================================
# cu124 покрывает CUDA 12.x (большинство современных AWS инстансов)
# Если у тебя CUDA 11.8 — замени cu124 на cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
echo "✓ PyTorch установлен"

# =============================================================================
echo ""
echo "=== [4] Transformers (dev ветка — нужна для Qwen3-Omni) ==="
# =============================================================================
# Сначала удаляем stable версию если была
pip uninstall transformers -y 2>/dev/null || true

# Dev версия из main — единственная что поддерживает Qwen3OmniMoeThinker
pip install git+https://github.com/huggingface/transformers

python -c "import transformers; print(f'transformers {transformers.__version__}')"
echo "✓ transformers (dev) установлен"

# =============================================================================
echo ""
echo "=== [5] Основные ML библиотеки ==="
# =============================================================================
pip install \
    "accelerate>=0.30.0" \
    "trl>=0.12.0" \
    "peft>=0.10.0"

# bitsandbytes — для QLoRA NF4. Версия 0.43+ нужна для bfloat16 + 4bit
pip install "bitsandbytes>=0.43.0"

python -c "import bitsandbytes as bnb; print(f'bitsandbytes {bnb.__version__}')"
echo "✓ accelerate, trl, peft, bitsandbytes установлены"

# =============================================================================
echo ""
echo "=== [6] Qwen3-Omni утилиты + decord ==="
# =============================================================================
pip install "qwen-omni-utils[decord]>=0.2.0"

# decord отдельно — иногда в qwen-omni-utils идёт старая версия
pip install "decord>=0.6.0"

python -c "import decord; print(f'decord {decord.__version__}')"
echo "✓ qwen-omni-utils и decord установлены"

# =============================================================================
echo ""
echo "=== [7] Вспомогательные пакеты ==="
# =============================================================================
pip install \
    ffmpeg-python \
    "numpy>=1.24" \
    tqdm \
    wandb \
    huggingface_hub

echo "✓ Вспомогательные пакеты установлены"

# =============================================================================
echo ""
echo "=== [8] Проверка всего окружения ==="
# =============================================================================
python - <<'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.0f} GB VRAM")

import transformers
print(f"transformers: {transformers.__version__}")

# Проверяем что Qwen3-Omni классы доступны
from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor
print("✓ Qwen3OmniMoeThinkerForConditionalGeneration — OK")

import peft
print(f"peft: {peft.__version__}")

import bitsandbytes as bnb
print(f"bitsandbytes: {bnb.__version__}")

from qwen_omni_utils import process_mm_info
print("✓ qwen_omni_utils.process_mm_info — OK")

import decord
print(f"decord: {decord.__version__}")

import trl
print(f"trl: {trl.__version__}")

import accelerate
print(f"accelerate: {accelerate.__version__}")
EOF

echo "✓ Все зависимости в порядке"

# =============================================================================
echo ""
echo "=== [9] Скачивание модели ==="
# =============================================================================
MODEL_DIR="/home/ubuntu/models"
MODEL_NAME="Qwen/Qwen3-Omni-30B-A3B-Thinking"  # <- если нет доступа к 30B, замени на 7B ниже
MODEL_PATH="$MODEL_DIR/Qwen3-Omni-30B-A3B-Thinking"

mkdir -p "$MODEL_DIR"

# Если модель уже скачана — пропускаем
if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
    echo "Модель уже существует: $MODEL_PATH — пропускаем скачивание"
else
    echo "Скачиваем $MODEL_NAME в $MODEL_PATH ..."
    echo "⚠️  Это займёт время (модель ~60GB). Запусти в tmux если нужно."
    echo ""

    # huggingface-cli download надёжнее чем git clone для больших моделей:
    #   - resume при обрыве
    #   - параллельная загрузка шардов
    #   - не тащит весь git history
    huggingface-cli download \
        "$MODEL_NAME" \
        --local-dir "$MODEL_PATH" \
        --local-dir-use-symlinks False \
        --resume-download

    echo "✓ Модель скачана в $MODEL_PATH"
fi

# Быстрая проверка что config.json есть
python - <<EOF
from transformers import Qwen3OmniMoeProcessor
proc = Qwen3OmniMoeProcessor.from_pretrained("$MODEL_PATH", trust_remote_code=True)
print(f"✓ Процессор загружается корректно (vocab_size={proc.tokenizer.vocab_size})")
EOF

# =============================================================================
echo ""
echo "=== [10] Структура директорий проекта ==="
# =============================================================================
PROJECT_DIR="/home/ubuntu/virality-grpo"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/checkpoints/virality_grpo"
mkdir -p "$PROJECT_DIR/logs"

echo "Скопируй свои файлы в $PROJECT_DIR:"
echo "  scp -r ./your_code/* ubuntu@<IP>:$PROJECT_DIR/"
echo ""
echo "Данные (train.jsonl / eval.jsonl) клади в:"
echo "  $PROJECT_DIR/data/"

# =============================================================================
echo ""
echo "============================================================"
echo "  ВСЁ ГОТОВО"
echo "============================================================"
echo ""
echo "Активировать окружение:"
echo "  source /home/ubuntu/venv/bin/activate"
echo ""
echo "Тестовый запуск (debug режим, 20 примеров):"
echo "  cd $PROJECT_DIR"
echo "  python train.py --debug --train_data ./data/train.jsonl"
echo ""
echo "Полный запуск:"
echo "  python train.py \\"
echo "      --train_data ./data/train.jsonl \\"
echo "      --eval_data  ./data/eval.jsonl \\"
echo "      --output_dir ./checkpoints/virality_grpo \\"
echo "      --wandb"
echo ""
echo "Мониторинг GPU во время обучения:"
echo "  watch -n 1 nvidia-smi"
echo "  # или:"
echo "  nvtop"
echo ""