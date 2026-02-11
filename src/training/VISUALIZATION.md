# Визуализация важности признаков Qwen модели

Скрипты для создания визуализаций важности признаков обученной Qwen2.5-Omni модели. Показывают, какие части входных данных (токены текста, кадры видео) влияют на предсказание виральности.

## 🚀 Быстрый старт

### Тестирование без модели
```bash
conda activate qwen3-project
# Из корня проекта:
python src/training/visualize_qwen_features.py --test_mode

# Или как модуль:
python -m src.training.visualize_qwen_features --test_mode
```

### Запуск на реальных данных
```bash
# Из корня проекта:
python src/training/visualize_qwen_features.py \
    --checkpoint_path checkpoints \
    --num_samples 5

# Или как модуль:
python -m src.training.visualize_qwen_features \
    --checkpoint_path checkpoints \
    --num_samples 5
```

## 📊 Что создается

Для каждого примера создаются **4 визуализации**:

1. **`attention_heatmap.png`** - Тепловая карта attention weights между токенами
2. **`token_importance.png`** - Топ-20 важных токенов текста (gradient-based)
3. **`frame_importance.png`** - Важность кадров видео (график)
4. **`video_token_heatmap_on_frames.png`** - Тепловая карта важности на кадрах ⭐

### Структура результатов
```
visualizations/qwen_features/
├── sample_1/
│   ├── attention_heatmap.png
│   ├── token_importance.png
│   ├── frame_importance.png
│   └── video_token_heatmap_on_frames.png
└── sample_2/...
```

## 📋 Требования

**Для запуска нужна обученная модель:**
- LoRA адаптеры: `checkpoints/epoch_X/lora_adapter/`
- MLP голова: `checkpoints/epoch_X/viral_head.pt`
- Данные: `data/processed/train.jsonl`

**Зависимости:**
```bash
pip install matplotlib seaborn Pillow
# Опционально: opencv-python (для улучшенного наложения)
```

## ⚙️ Параметры

- `--checkpoint_path` - Директория с checkpoints (по умолчанию: `checkpoints`)
- `--data_path` - Путь к данным (по умолчанию: `data/processed/train.jsonl`)
- `--output_dir` - Директория для результатов (по умолчанию: `visualizations/qwen_features`)
- `--num_samples` - Количество примеров (по умолчанию: 5)
- `--epoch` - Номер эпохи (по умолчанию: последняя)
- `--test_mode` - Тестовый режим без модели

## 🎨 Описание визуализаций

### 1. Attention Heatmap
Матрица внимания между токенами. Показывает связи между частями текста.

### 2. Token Importance
Топ-20 важных токенов для предсказания. Вычисляется через градиенты.

### 3. Frame Importance
График важности каждого кадра. Выделены топ-3 кадра.

### 4. Пространственная тепловая карта видео-токенов на кадрах ⭐
**Пространственная** тепловая карта важности, наложенная на оригинальные кадры. Показывает, **какие регионы каждого кадра** важны для модели.

**Особенности:**
- Тепловая карта показывает **пространственное распределение** важности по кадру
- Красные/желтые области = высоко важные регионы
- Черные области = низко важные регионы
- Позволяет визуально определить, на какие объекты или части кадра модель обращает внимание

## 🍎 macOS специфика

Скрипт автоматически определяет macOS и настраивается:
- Использует **MPS** (Metal Performance Shaders) для Apple Silicon
- Отключает quantization (не поддерживается на macOS)
- Использует numpy fallback если cv2 недоступен

**Проверка:**
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

## 🔧 Устранение проблем

### "Checkpoint не найден"
- Убедитесь, что модель обучена
- Проверьте путь к checkpoint

### "CUDA/MPS out of memory"
- Уменьшите `--num_samples`
- Закройте другие процессы

### "cv2 not found"
- Нормально, используется numpy fallback
- Для установки: `pip install opencv-python`

### "ModuleNotFoundError: No module named 'src'"
- Убедитесь, что запускаете из корня проекта: `C:\Projects\qwen3_1.0`
- Или используйте запуск как модуль: `python -m src.training.visualize_qwen_features`
- Скрипт автоматически добавляет путь, но лучше запускать из корня

### Тестирование перед запуском
```bash
# Из корня проекта:
python src/training/visualize_qwen_features.py --test_mode

# Или как модуль:
python -m src.training.visualize_qwen_features --test_mode
```

## 💡 Использование в коде

```python
from src.training.visualize_qwen_features import visualize_qwen_features

visualize_qwen_features(
    checkpoint_path="checkpoints",
    data_path="data/processed/train.jsonl",
    output_dir="visualizations/qwen_features",
    num_samples=5
)
```

## 📚 Методы анализа

1. **Gradient-based Importance** - градиенты по входным данным
2. **Attention Weights** - извлечение из последнего слоя трансформера
3. **Video Token Importance** - сопоставление токенов с кадрами через `video_grid_thw`
4. **Hidden State Analysis** - fallback если градиенты недоступны
