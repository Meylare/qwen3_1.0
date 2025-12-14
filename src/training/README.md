# Stage 1 Training - Audio Alignment

Этот скрипт обучает `audio_projector` генерировать описания аудио, используя teacher forcing.

## Что делает скрипт

1. **Загружает G2V2Model** с Qwen-3-VL
2. **Замораживает всю модель** кроме `audio_projector`
3. **Обучает проектор** на генерацию текстовых описаний аудио
4. **Сохраняет** только веса проектора в `projector.bin`

## Запуск

### С реальными данными (после активации conda)

```bash
conda activate qwen3-project
python src/training/train_stage1.py \
    --embeddings_file data/wavcaps_embeddings.pt \
    --captions_file data/wavcaps_captions.json \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --save_path checkpoints/projector.bin
```

### С фейковыми данными для теста (CPU)

```bash
python src/training/train_stage1.py \
    --use_fake_data \
    --use_fake_model \
    --num_epochs 1 \
    --batch_size 2 \
    --save_path test_projector.bin
```

## Параметры

- `--embeddings_file`: Путь к .pt файлу с CLAP эмбеддингами [N, 512]
- `--captions_file`: Путь к .json файлу со списком описаний
- `--model_name`: Имя модели Qwen для загрузки
- `--batch_size`: Размер батча
- `--num_epochs`: Количество эпох
- `--learning_rate`: Скорость обучения
- `--save_path`: Куда сохранить обученный проектор
- `--device`: Устройство (cpu/cuda)
- `--use_fake_data`: Использовать синтетические данные для теста
- `--use_fake_model`: Использовать упрощенную модель для теста

## Логика обучения

### Teacher Forcing

- **Вход**: Аудио-вектор + промпт "Describe this sound. Assistant: {caption}"
- **Цель**: Модель учится продолжать текст правильным описанием
- **Loss**: Cross-Entropy только на токенах ответа (caption)
- **Заморозка**: Все параметры кроме audio_projector не обучаются

### Формат данных

```json
// wavcaps_captions.json
["jazz music with piano", "dog barking loudly", "classical violin solo"]

// wavcaps_embeddings.pt - torch.Tensor [N, 512]
```

### Формат промпта

```
User: <|audio|> Describe this sound. Assistant: {caption} <|im_end|>
```

Только токены после "Assistant:" используются для обучения (labels != -100).

## Результат

После обучения:
- `projector.bin` содержит веса обученного audio_projector
- Модель может проецировать CLAP векторы в пространство Qwen
- Loss должен уменьшаться в процессе обучения

## Следующие шаги

После Stage 1 перейти к Stage 2 - обучению на вирусности видео с замороженным проектором.