# src/data/datasets.py
"""
Датасеты для обучения G2V2 модели.
Содержит три класса: ViralDataset, SFTDataset, DPODataset и функцию collate_fn.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import os
import json


class ViralDataset(Dataset):
    """
    Датасет для обучения виральности (Stage 2).
    Загружает тензоры видео/аудио и токенизирует текст.
    """

    def __init__(
        self,
        parquet_file: str,
        tokenizer: Any,
        max_len: int = 2048,
        data_root: str = "data"
    ):
        self.parquet_file = parquet_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_root = Path(data_root)

        # Загрузка данных
        self.df = pd.read_parquet(parquet_file)
        print(f"[INFO] Загружено {len(self.df)} сэмплов из {parquet_file}")

        # Проверка обязательных колонок
        required_cols = ['account_id', 'video_id', 'context_str', 'transcript_text', 'viral_index']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] Отсутствуют обязательные колонки: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_tensor(self, account_id: str, video_id: str, tensor_type: str) -> torch.Tensor:
        """
        Загрузка .pt файла с диска.
        tensor_type: 'video_embeds' или 'audio_embeds'
        """
        tensor_path = self.data_root / account_id / video_id / f"{tensor_type}.pt"

        if not tensor_path.exists():
            raise FileNotFoundError(f"[ERROR] Тензор не найден: {tensor_path}")

        tensor = torch.load(tensor_path, map_location='cpu', weights_only=True)
        return tensor

    def _process_transcript_text(self, transcript_text: str) -> str:
        """
        Обрабатывает текст транскрипта.
        Если это JSON с таймкодами - преобразует в читаемый формат.
        Если обычный текст - возвращает как есть.
        """
        if not isinstance(transcript_text, str) or not transcript_text.strip():
            return ""

        # Пробуем распарсить как JSON
        try:
            transcript_data = json.loads(transcript_text)

            # Если это список сегментов с таймкодами (стандартный формат Whisper)
            if isinstance(transcript_data, list):
                processed_segments = []
                for segment in transcript_data:
                    if isinstance(segment, dict) and 'text' in segment:
                        # Добавляем таймкод если есть
                        start_time = segment.get('start', 0)
                        text = segment['text'].strip()
                        if text:
                            # Форматируем таймкод в минуты:секунды
                            minutes = int(start_time // 60)
                            seconds = int(start_time % 60)
                            timestamp = f"[{minutes}:{seconds:02d}]"
                            processed_segments.append(f"{timestamp} {text}")

                if processed_segments:
                    return " ".join(processed_segments)

            # Если это объект с полем text
            elif isinstance(transcript_data, dict) and 'text' in transcript_data:
                return transcript_data['text']

            # Если это просто строка в JSON
            elif isinstance(transcript_data, str):
                return transcript_data

        except (json.JSONDecodeError, TypeError, KeyError):
            # Не JSON или неправильный формат - возвращаем как есть
            pass

        # Возвращаем оригинальный текст, если не смогли обработать как JSON
        return transcript_text

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Загрузка тензоров
        video_embeds = self._load_tensor(row['account_id'], row['video_id'], 'video_embeds')
        audio_embeds = self._load_tensor(row['account_id'], row['video_id'], 'audio_embeds')

        # Обработка транскрипта
        transcript_text = self._process_transcript_text(row['transcript_text'])

        # Формирование текста для токенизации
        text = f"{row['context_str']} {transcript_text}"

        # Токенизация с ChatML форматом
        messages = [
            {"role": "user", "content": text}
        ]

        # Применение chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback для токенизаторов без chat_template
            prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

        # Токенизация
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "video_embeds": video_embeds,          # [seq_v, hidden_v]
            "audio_embeds": audio_embeds,          # [seq_a, hidden_a]
            "input_ids": inputs["input_ids"].squeeze(0),     # [seq_t]
            "attention_mask": inputs["attention_mask"].squeeze(0),  # [seq_t]
            "label": torch.tensor(float(row['viral_index']), dtype=torch.float32)  # [1]
        }


class SFTDataset(Dataset):
    """
    Датасет для supervised fine-tuning (Stage 3).
    Текст-текст обучение с masking для loss.
    """

    def __init__(
        self,
        parquet_file: str,
        tokenizer: Any,
        max_len: int = 2048
    ):
        self.parquet_file = parquet_file
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Загрузка данных
        self.df = pd.read_parquet(parquet_file)
        print(f"[INFO] Загружено {len(self.df)} SFT сэмплов из {parquet_file}")

        # Проверка обязательных колонок
        required_cols = ['problem_json', 'advice_text']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] Отсутствуют обязательные колонки: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Парсинг JSON проблемы
        try:
            problem_data = json.loads(row['problem_json'])
            problem_text = json.dumps(problem_data, ensure_ascii=False)
        except:
            problem_text = str(row['problem_json'])

        # Формирование диалога
        messages = [
            {"role": "user", "content": problem_text},
            {"role": "assistant", "content": row['advice_text']}
        ]

        # Токенизация всего диалога
        if hasattr(self.tokenizer, 'apply_chat_template'):
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            full_text = f"<|im_start|>user\n{problem_text}<|im_end|>\n<|im_start|>assistant\n{row['advice_text']}<|im_end|>\n"

        # Токенизация
        inputs = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Создание labels с masking (заменяем токены пользователя на -100)
        labels = input_ids.clone()

        # Находим позицию assistant ответа
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Для токенизаторов с chat_template - сложная логика определения границ
            assistant_start = self._find_assistant_start(full_text, input_ids)
        else:
            # Простой fallback - ищем <|im_start|>assistant
            assistant_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
            if assistant_token:
                assistant_start = self._find_token_sequence(input_ids, assistant_token)
            else:
                # Если нет спец-токенов, маскируем первые 50% токенов
                assistant_start = len(input_ids) // 2

        # Маскировка токенов пользователя (все до assistant_start)
        if assistant_start > 0:
            labels[:assistant_start] = -100

        return {
            "input_ids": input_ids,              # [seq]
            "attention_mask": attention_mask,    # [seq]
            "labels": labels                     # [seq] с -100 для user токенов
        }

    def _find_assistant_start(self, full_text: str, input_ids: torch.Tensor) -> int:
        """Находит позицию начала ответа ассистента в токенах."""
        # Простая эвристика - ищем середину текста
        return len(input_ids) // 2

    def _find_token_sequence(self, input_ids: torch.Tensor, sequence: List[int]) -> int:
        """Находит позицию последовательности токенов."""
        seq_len = len(sequence)
        for i in range(len(input_ids) - seq_len + 1):
            if input_ids[i:i+seq_len].tolist() == sequence:
                return i + seq_len  # Возвращаем позицию после последовательности
        return len(input_ids) // 2  # Fallback


class DPODataset(Dataset):
    """
    Датасет для Direct Preference Optimization (Stage 3).
    Содержит тройки: prompt, chosen response, rejected response.
    """

    def __init__(
        self,
        parquet_file: str,
        tokenizer: Any,
        max_len: int = 2048
    ):
        self.parquet_file = parquet_file
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Загрузка данных
        self.df = pd.read_parquet(parquet_file)
        print(f"[INFO] Загружено {len(self.df)} DPO сэмплов из {parquet_file}")

        # Проверка обязательных колонок
        required_cols = ['prompt_text', 'chosen_text', 'rejected_text']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] Отсутствуют обязательные колонки: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Токенизация prompt
        prompt_inputs = self.tokenizer(
            row['prompt_text'],
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        # Токенизация chosen response
        chosen_inputs = self.tokenizer(
            row['chosen_text'],
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        # Токенизация rejected response
        rejected_inputs = self.tokenizer(
            row['rejected_text'],
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "prompt_input_ids": prompt_inputs["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_inputs["attention_mask"].squeeze(0),
            "chosen_input_ids": chosen_inputs["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_inputs["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_inputs["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_inputs["attention_mask"].squeeze(0)
        }


def viral_collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer=None) -> Dict[str, torch.Tensor]:
    """
    Функция сборки батча для ViralDataset.
    Обрабатывает паддинг тензоров разной длины.

    Args:
        batch: Список сэмплов из датасета
        tokenizer: Токенизатор для получения правильного pad_token_id
    """
    if not batch:
        return {}

    # Извлечение списков
    video_embeds = [item['video_embeds'] for item in batch]      # List of [seq_v, hidden_v]
    audio_embeds = [item['audio_embeds'] for item in batch]      # List of [seq_a, hidden_a]
    input_ids = [item['input_ids'] for item in batch]            # List of [seq_t]
    attention_masks = [item['attention_mask'] for item in batch] # List of [seq_t]
    labels = [item['label'] for item in batch]                   # List of [1]

    # Паддинг видео эмбеддингов
    if video_embeds:
        video_padded = torch.nn.utils.rnn.pad_sequence(
            video_embeds,
            batch_first=True,
            padding_value=0.0
        )  # [batch, max_seq_v, hidden_v]
        video_mask = _create_sequence_mask(video_embeds)  # [batch, max_seq_v]
    else:
        video_padded = torch.empty(0)
        video_mask = torch.empty(0)

    # Паддинг аудио эмбеддингов
    if audio_embeds:
        audio_padded = torch.nn.utils.rnn.pad_sequence(
            audio_embeds,
            batch_first=True,
            padding_value=0.0
        )  # [batch, max_seq_a, hidden_a]
        audio_mask = _create_sequence_mask(audio_embeds)  # [batch, max_seq_a]
    else:
        audio_padded = torch.empty(0)
        audio_mask = torch.empty(0)

    # Определение pad_token_id
    if tokenizer and hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = 0  # fallback

    # Паддинг текстовых последовательностей
    text_inputs = _collate_text_sequences(input_ids, attention_masks, pad_token_id)

    # Собираем батч
    return {
        "video_embeds": video_padded,                    # [batch, max_seq_v, hidden_v]
        "video_attention_mask": video_mask,              # [batch, max_seq_v]
        "audio_embeds": audio_padded,                    # [batch, max_seq_a, hidden_a]
        "audio_attention_mask": audio_mask,              # [batch, max_seq_a]
        "input_ids": text_inputs["input_ids"],          # [batch, max_seq_t]
        "attention_mask": text_inputs["attention_mask"], # [batch, max_seq_t]
        "labels": torch.stack(labels)                    # [batch, 1]
    }


def _create_sequence_mask(sequences: List[torch.Tensor]) -> torch.Tensor:
    """
    Создает маску для последовательностей.
    1 - где есть данные, 0 - где паддинг.
    """
    if not sequences:
        return torch.empty(0)

    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)

    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        mask[i, :seq_len] = 1.0

    return mask


def _collate_text_sequences(
    input_ids: List[torch.Tensor],
    attention_masks: List[torch.Tensor],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Паддинг текстовых последовательностей.

    Args:
        input_ids: Список тензоров с input_ids
        attention_masks: Список тензоров с attention_mask
        pad_token_id: ID токена для паддинга
    """
    max_len = max(ids.size(0) for ids in input_ids)

    # Паддинг input_ids
    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids, attention_masks):
        seq_len = ids.size(0)

        # Паддинг input_ids правильным pad_token_id
        if seq_len < max_len:
            pad_size = max_len - seq_len
            pad_tensor = torch.full((pad_size,), pad_token_id, dtype=ids.dtype)
            padded_ids = torch.cat([ids, pad_tensor])
            padded_mask = torch.cat([mask, torch.zeros(pad_size, dtype=mask.dtype)])
        else:
            padded_ids = ids
            padded_mask = mask

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)

    return {
        "input_ids": torch.stack(padded_input_ids),        # [batch, max_seq_t]
        "attention_mask": torch.stack(padded_attention_masks)  # [batch, max_seq_t]
    }