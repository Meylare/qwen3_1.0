import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from decord import VideoReader, cpu
import json
import os
import numpy as np


class ViralVideoDataset(Dataset):
    def __init__(self, jsonl_path, processor_path="Qwen2.5-Omni-7B", max_frames=10):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        # trust_remote_code=True нужен для новых моделей Qwen
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            trust_remote_code=True
        )
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item['video_path']

        # 1. Загрузка кадров
        frames = self._load_video(video_path)

        # 2. Формирование сообщений для Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": item['system_prompt']},
                ],
            }
        ]

        # 3. Подготовка промпта
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 4. Токенизация и процессинг видео
        inputs = self.processor(
            text=[text_input],
            videos=[frames],
            padding=True,
            return_tensors="pt"
        )

        label = torch.tensor(item['targets']['viral_index'], dtype=torch.float)

        # Используем видео-ключи для Qwen2.5-Omni (важно для временной шкалы)
        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label
        }
        
        # Приоритет: pixel_values_videos > pixel_values
        if "pixel_values_videos" in inputs:
            result["pixel_values_videos"] = inputs["pixel_values_videos"].squeeze(0)
        elif "pixel_values" in inputs:
            result["pixel_values_videos"] = inputs["pixel_values"].squeeze(0)
        
        # Приоритет: video_grid_thw > image_grid_thw
        if "video_grid_thw" in inputs:
            result["video_grid_thw"] = inputs["video_grid_thw"].squeeze(0)
        elif "image_grid_thw" in inputs:
            result["video_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result

    def _load_video(self, path):
        if not os.path.exists(path):
            # Возвращаем черный квадрат, чтобы не крашить обучение, но логируем ошибку
            print(f"⚠️ Warning: Video not found at {path}")
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_frames)]

        try:
            vr = VideoReader(path, ctx=cpu(0))
            total_frames = len(vr)
            # Равномерный сэмплинг (начало, середина, конец)
            indices = np.linspace(0, total_frames - 1, self.max_frames).astype(int)
            frames = vr.get_batch(indices).asnumpy()
            return list(frames)
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_frames)]
