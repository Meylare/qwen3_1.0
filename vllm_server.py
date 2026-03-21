"""
vllm_server.py — управление vLLM-Omni сервером и генерация completions.

Функциональность:
  - Запуск vLLM-Omni сервера как subprocess
  - Health check — ожидание готовности сервера
  - Генерация G completions для пары видео через OpenAI-compatible API
  - Graceful shutdown

Почему vLLM-Omni а не HuggingFace generate():
  - CUDA graphs + async-chunk → TTFP уменьшен на 90% для Qwen3-Omni
  - FlashAttention 3 по умолчанию на Blackwell
  - 30-40 секунд вместо 346 секунд на одну пару видео

Конфигурация запуска:
  Используем qwen3_omni_moe_thinker_only stage config —
  только Thinker стадия (text output), без Talker и Code2Wav.
  Это наш случай: Qwen3-Omni-30B-A3B-Thinking.

API:
  vLLM-Omni использует OpenAI-compatible API.
  Видео передаётся как base64 в image_url поле (video_url).
  LoRA запрос передаётся через extra_body.
"""
import base64
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ── Константы ─────────────────────────────────────────────────────────────────

DEFAULT_PORT = 8091
DEFAULT_HOST = "localhost"
HEALTH_CHECK_INTERVAL = 2.0   # секунд между проверками
HEALTH_CHECK_TIMEOUT = 300.0  # максимум 5 минут на старт сервера

# Stage config для Thinking модели (только Thinker, без Talker/TTS)
# Этот конфиг верифицирован на 1x H100/A100 80GB
THINKER_ONLY_STAGE_CONFIG = """
# Stage config for Qwen3-Omni-MoE-Thinking (text-only output)
# Only Thinker stage — no Talker, no Code2Wav
# Verified on 1x H100-80G / RTX PRO 6000 Blackwell (96GB)

async_chunk: true

stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
      max_batch_size: 1
      engine_args:
        model_stage: thinker
        model_arch: Qwen3OmniMoeThinkerForConditionalGeneration
        worker_type: ar
        scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
        gpu_memory_utilization: 0.85
        enforce_eager: false
        trust_remote_code: true
        enable_lora: true
        max_lora_rank: 32
"""


class VLLMOmniServer:
    """
    Управляет жизненным циклом vLLM-Omni сервера и генерацией completions.

    Использование:
        server = VLLMOmniServer(
            model_path="/home/ubuntu/models/Qwen3-Omni-30B-A3B-AWQ",
            lora_adapter_name="virality_grpo",
        )
        server.start()
        completions = server.generate(video_a_path, video_b_path, prompt, G=2)
        server.stop()
    """

    def __init__(
        self,
        model_path: str,
        port: int = DEFAULT_PORT,
        host: str = DEFAULT_HOST,
        lora_adapter_name: str = "virality_grpo",
        stage_config_path: Optional[str] = None,  # None = автоматически создаём
        gpu_memory_utilization: float = 0.85,
    ):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.base_url = f"http://{host}:{port}"
        self.lora_adapter_name = lora_adapter_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self._process: Optional[subprocess.Popen] = None

        # Stage config — создаём временный файл если не передан
        if stage_config_path:
            self.stage_config_path = Path(stage_config_path)
        else:
            self.stage_config_path = Path("/dev/shm/vllm_omni_stage_config.yaml")
            self._write_stage_config()

    def _write_stage_config(self) -> None:
        """Записывает stage config в /dev/shm."""
        config = THINKER_ONLY_STAGE_CONFIG.replace(
            "gpu_memory_utilization: 0.85",
            f"gpu_memory_utilization: {self.gpu_memory_utilization}"
        )
        self.stage_config_path.write_text(config)
        logger.info(f"Stage config written to {self.stage_config_path}")

    def start(self, wait: bool = True) -> None:
        """
        Запускает vLLM-Omni сервер как subprocess.

        Args:
            wait: ждать ли пока сервер будет готов принимать запросы
        """
        if self._process and self._process.poll() is None:
            logger.warning("vLLM-Omni server already running")
            return

        cmd = [
            "vllm", "serve", self.model_path,
            "--omni",
            "--port", str(self.port),
            "--host", "0.0.0.0",
            "--stage-configs-path", str(self.stage_config_path),
            "--trust-remote-code",
            "--enable-lora",
            "--max-lora-rank", "32",
        ]

        env = os.environ.copy()
        # Разрешаем динамическую замену LoRA адаптеров без перезапуска
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"

        logger.info(f"Starting vLLM-Omni server: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if wait:
            self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        """
        Ждёт пока сервер готов принимать запросы.
        Проверяет /health endpoint каждые HEALTH_CHECK_INTERVAL секунд.
        """
        deadline = time.time() + HEALTH_CHECK_TIMEOUT
        logger.info(f"Waiting for vLLM-Omni server on {self.base_url}...")

        while time.time() < deadline:
            # Проверяем что процесс ещё жив
            if self._process.poll() is not None:
                stdout = self._process.stdout.read()
                raise RuntimeError(
                    f"vLLM-Omni server died during startup. "
                    f"Exit code: {self._process.returncode}\n{stdout[-2000:]}"
                )

            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    logger.info(f"vLLM-Omni server ready on port {self.port}")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass

            time.sleep(HEALTH_CHECK_INTERVAL)

        raise TimeoutError(
            f"vLLM-Omni server did not become ready within {HEALTH_CHECK_TIMEOUT}s"
        )

    def generate(
        self,
        video_a_path: str,
        video_b_path: str,
        system_prompt: str,
        user_text: str,
        G: int = 2,
        max_new_tokens: int = 600,
        temperature: float = 0.9,
        top_p: float = 0.95,
        lora_adapter_path: Optional[str] = None,
    ) -> List[str]:
        """
        Генерирует G completions для пары видео.

        Args:
            video_a_path: путь к первому видео
            video_b_path: путь ко второму видео
            system_prompt: системный промпт
            user_text: текстовый запрос пользователя
            G: количество completions
            max_new_tokens: максимум новых токенов
            temperature: температура сэмплирования
            top_p: top-p сэмплирования
            lora_adapter_path: путь к LoRA адаптеру (None = base model)

        Returns:
            Список из G строк completions
        """
        # Кодируем видео в base64
        video_a_b64 = self._encode_video(video_a_path)
        video_b_b64 = self._encode_video(video_b_path)

        # Строим сообщения в формате OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_a_b64}"},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b_b64}"},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        # Extra body для LoRA запроса
        extra_body = {}
        if lora_adapter_path:
            extra_body["lora_request"] = {
                "lora_name": self.lora_adapter_name,
                "lora_path": lora_adapter_path,
            }

        # model: если используется LoRA адаптер — передаём имя адаптера,
        # иначе vLLM маршрутизирует на base модель и игнорирует lora_request.
        payload = {
            "model": self.lora_adapter_name if lora_adapter_path else self.model_path,
            "messages": messages,
            "n": G,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            **extra_body,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=600.0,  # 10 минут на генерацию
            )
            response.raise_for_status()
            data = response.json()

            completions = [
                choice["message"]["content"]
                for choice in data["choices"]
            ]
            logger.info(
                f"vLLM-Omni generated {len(completions)} completions, "
                f"tokens: {data.get('usage', {}).get('completion_tokens', '?')}"
            )
            return completions

        except requests.exceptions.Timeout:
            logger.error("vLLM-Omni generate timeout (>600s)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM-Omni generate error: {e}")
            raise

    def _encode_video(self, video_path: str) -> str:
        """Кодирует видео файл в base64 строку."""
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def is_alive(self) -> bool:
        """Проверяет что сервер жив и отвечает."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def stop(self) -> None:
        """Graceful shutdown сервера."""
        if self._process is None:
            return

        if self._process.poll() is not None:
            logger.info("vLLM-Omni server already stopped")
            return

        logger.info("Stopping vLLM-Omni server...")
        self._process.send_signal(signal.SIGTERM)

        try:
            self._process.wait(timeout=30)
            logger.info("vLLM-Omni server stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("vLLM-Omni server didn't stop gracefully, killing...")
            self._process.kill()
            self._process.wait()

        self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()