"""
lora_sync.py — синхронизация LoRA весов между HuggingFace тренером и vLLM-Omni сервером.

Схема работы:
  1. После каждого optimizer.step() тренер вызывает sync()
  2. LoRA адаптер сохраняется в /dev/shm (RAM диск) — мгновенно, ~34MB
  3. POST запрос к vLLM-Omni серверу на загрузку нового адаптера
  4. vLLM-Omni загружает адаптер — следующие generate() используют новые веса

Почему /dev/shm:
  - RAM диск, нет I/O на диск
  - 34MB (17M параметров × 2 байта) передаётся за <1ms
  - Оба процесса имеют доступ (один хост)

Почему не прямая передача через GPU:
  - Два отдельных процесса не могут читать GPU память друг друга
  - CPU → /dev/shm → CPU единственный безопасный путь
"""
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Путь в RAM диске — избегаем I/O на диск
LORA_SYNC_PATH = Path("/dev/shm/virality_lora_sync")

# Имя адаптера в vLLM-Omni — используется при каждом generate() запросе
LORA_ADAPTER_NAME = "virality_grpo"


class LoRASyncManager:
    """
    Менеджер синхронизации LoRA весов между HuggingFace и vLLM-Omni.

    Использование:
        sync_manager = LoRASyncManager(vllm_url="http://localhost:8091")
        # После каждого optimizer.step():
        sync_manager.sync(model)
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8091",
        sync_path: Path = LORA_SYNC_PATH,
        adapter_name: str = LORA_ADAPTER_NAME,
        timeout: float = 30.0,  # секунд на HTTP запрос
    ):
        self.vllm_url = vllm_url.rstrip("/")
        self.sync_path = Path(sync_path)
        self.adapter_name = adapter_name
        self.timeout = timeout
        self._sync_count = 0

        # Создаём директорию в /dev/shm при инициализации
        self.sync_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LoRASyncManager initialized: vllm={vllm_url}, sync_path={sync_path}")

    def sync(self, model) -> bool:
        """
        Синхронизирует LoRA веса из HuggingFace модели в vLLM-Omni.

        Args:
            model: PEFT модель с LoRA адаптером (PeftModel)

        Returns:
            True если синхронизация успешна, False если ошибка
        """
        t0 = time.time()

        # Шаг 1: сохраняем адаптер в /dev/shm
        try:
            self._save_adapter(model)
        except Exception as e:
            logger.error(f"LoRA sync: failed to save adapter: {e}")
            return False

        save_time = time.time() - t0

        # Шаг 2: отправляем запрос в vLLM-Omni
        try:
            success = self._load_adapter_in_vllm()
        except Exception as e:
            logger.error(f"LoRA sync: failed to load adapter in vLLM: {e}")
            return False

        total_time = time.time() - t0
        self._sync_count += 1

        if success:
            logger.info(
                f"LoRA sync #{self._sync_count}: "
                f"save={save_time:.2f}s, total={total_time:.2f}s"
            )
        return success

    def _save_adapter(self, model) -> None:
        """
        Сохраняет LoRA адаптер в /dev/shm в формате PEFT.

        Формат PEFT:
            /dev/shm/virality_lora_sync/
                adapter_config.json
                adapter_model.safetensors
        """
        # Сохраняем только адаптер (не base модель) — это ~34MB
        # save_pretrained с safe_serialization=True создаёт .safetensors
        model.save_pretrained(
            str(self.sync_path),
            safe_serialization=True,
        )
        logger.debug(f"Adapter saved to {self.sync_path}")

    def _load_adapter_in_vllm(self) -> bool:
        """
        Отправляет POST запрос к vLLM-Omni для загрузки нового адаптера.

        vLLM API для динамической загрузки LoRA:
        POST /v1/load_lora_adapter
        {
            "lora_name": "virality_grpo",
            "lora_path": "/dev/shm/virality_lora_sync",
            "base_model_name": "Qwen3-Omni-30B-A3B-AWQ"
        }

        Требует: VLLM_ALLOW_RUNTIME_LORA_UPDATING=True на сервере
        """
        url = f"{self.vllm_url}/v1/load_lora_adapter"
        payload = {
            "lora_name": self.adapter_name,
            "lora_path": str(self.sync_path),
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                logger.debug(f"vLLM-Omni loaded adapter: {response.json()}")
                return True
            else:
                logger.error(
                    f"vLLM-Omni load_lora_adapter failed: "
                    f"status={response.status_code}, body={response.text[:200]}"
                )
                return False
        except requests.exceptions.Timeout:
            logger.error(f"vLLM-Omni load_lora_adapter timeout ({self.timeout}s)")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"vLLM-Omni connection error: {self.vllm_url}")
            return False

    def get_lora_request(self) -> dict:
        """
        Возвращает параметры LoRA запроса для передачи в generate().

        Используется в vllm_server.py при каждом запросе генерации:
            extra_body={"lora_request": sync_manager.get_lora_request()}
        """
        return {
            "lora_name": self.adapter_name,
            "lora_path": str(self.sync_path),
        }

    def cleanup(self) -> None:
        """Удаляет временные файлы из /dev/shm."""
        if self.sync_path.exists():
            shutil.rmtree(self.sync_path, ignore_errors=True)
            logger.info(f"Cleaned up sync path: {self.sync_path}")


def wait_for_initial_sync(
    sync_manager: LoRASyncManager,
    model,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> None:
    """
    Начальная синхронизация — вызывается один раз перед обучением.

    Загружает начальные (случайные) LoRA веса в vLLM-Omni чтобы
    первая генерация тоже использовала текущий адаптер.
    """
    logger.info("Initial LoRA sync: uploading starting adapter to vLLM-Omni...")
    for attempt in range(max_retries):
        success = sync_manager.sync(model)
        if success:
            logger.info("Initial LoRA sync: success")
            return
        logger.warning(f"Initial LoRA sync attempt {attempt + 1}/{max_retries} failed, retrying...")
        time.sleep(retry_delay)
    raise RuntimeError(
        f"Initial LoRA sync failed after {max_retries} attempts. "
        "Check that vLLM-Omni server is running and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True"
    )