# Qwen3-VL Video Virality Prediction System

Система для предсказания вирусности видео на основе мультимодального анализа с использованием Qwen3-VL и XGBoost.

## 🏗️ Архитектура

Проект состоит из двух основных компонентов:

### 🤖 Нейронная сеть G2V2 (Grok Video Virality)
- **Модель:** Qwen3-VL-8B-Instruct с LoRA адаптацией
- **Вход:** Видео, аудио, текст описания
- **Выход:** Оценка вирусности 

### 📊 XGBoost Baseline
- **Признаки:** Исторические метрики аккаунта + статические признаки
- **Цель:** Предсказание логарифма просмотров
- **Метрики:** RMSE, R², SMAPE

## 📦 Установка

```bash

# Установка зависимостей
pip install torch torchvision torchaudio transformers peft pillow
pip install xgboost pandas scikit-learn tqdm joblib pytest
```

## 🚀 Быстрый старт

### 1. Подготовка данных

```bash
cd src/baseline

# 1. Загрузка сырых данных
python ingestion.py

# 2. Создание исторических признаков
python feature_engineering.py

# 3. Финальная сборка датасета
python merge_dataset.py
```

### 2. Обучение baseline модели

```bash
# Обучение XGBoost на исторических данных
python train.py
```

### 3. Генерация таргетов для LLM

```bash
# Создание датасета для обучения G2V2
python generate_viral_index.py
```

### 4. Тестирование компонентов

```bash
# Запуск всех тестов
python -m pytest tests/ -v
```

## 📋 Data Contract

### Идентификаторы (Meta)
- `account_id` (str): ID аккаунта
- `video_id` (str): ID видео
- `upload_date` (datetime): Дата загрузки
- `video_text` (str): Текст описания

### Таргет (Target)
- `target_views_log` (float): `log1p(current_video_views)`

### Статические признаки (Static Features)
- `st_log_followers` (float): Логарифм подписчиков
- `st_ratio_ff` (float): Отношение подписок к подписчикам
- `st_is_verified` (int): Флаг верификации (0/1)
- `st_is_business` (int): Флаг бизнес-аккаунта (0/1)
- `st_bio_length` (int): Длина биографии
- `st_hist_total_posts` (int): Общее количество постов

### Исторические признаки (History Features)
- `hist_window_size` (int): Количество найденных видео в истории
- `hist_median_views` (float): Медиана просмотров в истории
- `hist_mean_views` (float): Среднее просмотров в истории
- `hist_std_views` (float): Стандартное отклонение просмотров
- `hist_trend_views` (float): Тренд просмотров (последние 3 / последние 12)
- `hist_median_likes` (float): Медиана лайков в истории
- `hist_avg_er` (float): Средний engagement rate
- `hist_days_since_last` (float): Дни с момента последнего поста
- `hist_post_freq` (float): Частота постов

## 🧠 G2V2 Model Architecture

```
Input: Video + Text + Audio
       ↓
Qwen3-VL-8B (заморожен) → Vision Encoder
       ↓
Audio Projector (512→4096) → CLAP embeddings
       ↓
Fusion Layer → объединение модальностей
       ↓
LoRA адаптация → fine-tuning для задачи
       ↓
Predictor Head → Viral Index (0-1)
```

### Ключевые компоненты

- **QwenVideoProcessor:** Подготовка видео для Qwen3-VL
- **Projector:** MLP адаптер для CLAP эмбеддингов
- **PredictorHead:** Регрессионная голова для предсказания
- **RMSNorm:** Нормализация по Qwen стандартам

## 📊 Метрики качества

### XGBoost Baseline
- **RMSE:** < 0.8 (на логарифмах просмотров)
- **R²:** > 0.75
- **SMAPE:** < 25%

### G2V2 Model
- **Точность предсказания:** > 80% (вирусность > медианы)
- **Корреляция с реальными просмотрами:** > 0.7

## 🧪 Тестирование

```bash
# Тесты компонентов
python tests/test_projector.py    # Projector/MLPAdapter
python tests/test_predictor.py    # PredictorHead/ViralPredictor
python tests/test_processor.py    # QwenVideoProcessor

# Все тесты
python -m pytest tests/ -v
```

## 📁 Структура проекта

```
├── src/
│   ├── data/
│   │   └── processors.py          # QwenVideoProcessor
│   ├── models/
│   │   ├── __init__.py           # Экспорты компонентов
│   │   ├── components.py          # RMSNorm, Projector, PredictorHead
│   │   └── g2v2_model.py          # G2V2Model с LoRA
│   └── baseline/                  # XGBoost пайплайн
│       ├── ingestion.py           # Загрузка данных
│       ├── feature_engineering.py # Создание признаков
│       ├── merge_dataset.py       # Сборка датасета
│       ├── train.py              # Обучение модели
│       ├── generate_viral_index.py # Генерация таргетов
│       └── statistic/
│           └── features_lib.py    # Статические признаки
├── tests/                         # Модульные тесты
├── data_analyze/                  # Анализ данных
└── work_data/                     # Рабочие скрипты
```

## 🔧 Конфигурация

### Требования к окружению
- Python 3.10+
- CUDA 11.8+ (для GPU)
- 16GB+ RAM
- 100GB+ диск (для моделей)

### Переменные окружения
```bash
export CUDA_VISIBLE_DEVICES=0,1    # GPUs для обучения
export HF_HOME=/path/to/cache       # Кэш HuggingFace
```

## 🤝 Contributing

1. Активируйте conda окружение перед запуском
2. Запускайте тесты перед коммитом
3. Соблюдайте data contract для новых признаков
4. Добавляйте тесты для новых компонентов

