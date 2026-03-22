# Dataset Format — GSPO + Qwen3-Omni-30B, задача виральности видео

## Сырой формат (входные данные)

Файл `raw_pairs.jsonl` — то, что вы собираете:

```jsonc
{
  "video_a": "C:\\Projects\\data\\pair_vid\\b98c18fa.mp4",
  "video_b": "C:\\Projects\\data\\pair_vid\\84a7a5c3.mp4",
  "views_a": 6997,
  "views_b": 1013,
  "author_context": "Розничный магазин товаров (одежда или аксессуары) с физическими точками продаж в торговых центрах."
}
```

## Формат после prepare_dataset.py (для обучения)

Файлы `data/train.jsonl` и `data/val.jsonl`:

```jsonc
{
  "messages": [
    {
      "role": "system",
      "content": "Ты эксперт по анализу виральности коротких видео. ..."
    },
    {
      "role": "user",
      "content": [
        {"type": "text",  "text": "Контекст канала: Розничный магазин...\n\nВидео A:"},
        {"type": "video", "video": "C:/Projects/data/pair_vid/b98c18fa.mp4"},
        {"type": "text",  "text": "Видео B:"},
        {"type": "video", "video": "C:/Projects/data/pair_vid/84a7a5c3.mp4"},
        {"type": "text",  "text": "Какое видео наберёт больше просмотров — A или B?"}
      ]
    }
  ],
  "solution": "A",
  "task_type": "video_virality",
  "views_a": 6997,
  "views_b": 1013
}
```

### Важные поля

| Поле | Тип | Назначение |
|---|---|---|
| `messages` | list | Промпт в формате ms-swift multimodal |
| `solution` | "A" или "B" | Ground-truth для virality_accuracy |
| `views_a` / `views_b` | int | Передаются в virality_calibration для оси B |
| `task_type` | str | Идентификатор задачи, зарезервирован |

### Что НЕ попадает в датасет

Поля с `_` префиксом вырезаются при записи — они только для аналитики:
- `_confidence` — high / medium / low по ratio просмотров
- `_swapped` — была ли пара зеркально перевёрнута

## Аугментация

prepare_dataset.py из каждой исходной пары делает 2 примера:
- оригинал: video_a=b98c..., video_b=84a7..., solution="A"
- swap:      video_a=84a7..., video_b=b98c..., solution="B"

Это защищает от позиционного смещения (модель не учится "всегда отвечать A").

## Запуск подготовки

```bash
python prepare_dataset.py \
    --input  raw_pairs.jsonl \
    --output_dir data/ \
    --val_ratio 0.1
```

## Ожидаемый формат ответа модели

```
<think>
Видео A: крючок слабый — первые 2 секунды статичная картинка...
Динамика монтажа — видео A: 1 переход каждые 4 сек, видео B: каждые 1.5 сек...
Эмоциональный отклик: видео B показывает реакцию покупателя...
Соответствие аудитории: оба релевантны для розничного магазина одежды...
Удержание: у видео B лучший темп для платформы...
</think>
ВЫБОР: B
```