# src/data/processors.py
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Optional, Any
from transformers import AutoProcessor

class QwenVideoProcessor:
    """
    Процессор для подготовки видео и текста для моделей семейства Qwen3-VL.
    Поддерживает динамическое разрешение и нарезку на патчи.
    """
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct", 
        min_pixels: int = 224 * 224,
        max_pixels: int = 512 * 512,
        fps: float = 1.0 
    ):
        print(f"⚙️ Инициализация QwenVideoProcessor ({model_id})...")
        self.max_pixels = max_pixels
        self.fps = fps
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                trust_remote_code=True,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
        except Exception as e:
            print(f"⚠️ Ошибка загрузки AutoProcessor: {e}")
            self.processor = None

    def process(
        self, 
        text: str, 
        video_frames: List[Union[Image.Image, np.ndarray]], 
        system_prompt: str = "You are a helpful assistant."
    ) -> Dict[str, torch.Tensor]:
        
        if self.processor is None:
            raise RuntimeError("Processor not initialized correctly.")

        # 1. Нормализация кадров (numpy -> PIL)
        pil_frames = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                pil_frames.append(Image.fromarray(frame))
            else:
                pil_frames.append(frame)

        # --- FIX: Qwen3-VL требует минимум 2 кадра для видео ---
        if len(pil_frames) == 1:
            # Дублируем кадр, чтобы temporal_factor=2 не ронял код
            pil_frames.append(pil_frames[0])
        # -------------------------------------------------------

        # 2. Формирование сообщения (Chat Format)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": pil_frames,
                        "fps": self.fps,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        # 3. Подготовка текста
        text_prompt = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 4. Процессинг
        batch_inputs = self.processor(
            text=[text_prompt],
            videos=[pil_frames],
            padding=True,
            return_tensors="pt"
        )
        
        # 5. Формирование результата
        result = {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"]
        }

        # Поиск pixel_values
        if "pixel_values" in batch_inputs:
            result["pixel_values"] = batch_inputs["pixel_values"]
        elif "pixel_values_videos" in batch_inputs:
            result["pixel_values"] = batch_inputs["pixel_values_videos"]
        else:
            print(f"⚠️ WARNING: 'pixel_values' not found. Available keys: {list(batch_inputs.keys())}")
        
        # Поиск Grid THW
        if "image_grid_thw" in batch_inputs:
            result["image_grid_thw"] = batch_inputs["image_grid_thw"]
        elif "video_grid_thw" in batch_inputs:
            result["image_grid_thw"] = batch_inputs["video_grid_thw"]
        else:
             print("⚠️ WARNING: 'image_grid_thw' not found.")

        return result