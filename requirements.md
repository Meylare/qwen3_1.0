## Зависимости

```
# Core
torch>=2.3.0
transformers @ git+https://github.com/huggingface/transformers  # нужна dev версия для Qwen3-Omni
accelerate>=0.30.0
trl>=0.12.0

# Qwen3-Omni utils — обязательно
qwen-omni-utils[decord]>=0.2.0   # decord для быстрой загрузки видео

# QLoRA
peft>=0.10.0
bitsandbytes>=0.43.0

# Видео обработка
decord>=0.6.0      # быстрее torchvision для видео
ffmpeg-python      # конвертация форматов если нужно

# Утилиты
numpy>=1.24
tqdm
```

### Установка
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers
pip install accelerate trl peft bitsandbytes
pip install "qwen-omni-utils[decord]"
pip install ffmpeg-python tqdm
sudo apt install ffmpeg  # системный ffmpeg обязателен
```
