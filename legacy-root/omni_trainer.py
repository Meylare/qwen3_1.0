"""
omni_trainer.py — GRPO тренер + коллатор для Qwen3-Omni Thinker.

Архитектура Qwen3-Omni:
  Thinker (MoE LM) → генерирует текст + <think>...</think>
  Talker (speech decoder) → НАМ НЕ НУЖЕН, используем Thinker-only класс

Коллатор здесь потому что он тесно связан с тренером:
ему нужен processor и те же флаги (use_audio_in_video, fps) что используются
при генерации и forward pass.

GRPO алгоритм (чистый, без ref модели):
  1. Для каждого промпта генерируем G completion'ов
  2. Считаем reward каждого через reward_fn
  3. Advantage = (r - mean_g) / std_g  [нормализация внутри группы]
  4. GRPO loss = -E[advantage * log π_policy(c|p)]
  5. Backward + update

Стабильность без ref модели обеспечивается:
  - нормализацией advantages (всегда ~[-2, +2])
  - clip_grad_norm_ перед каждым optimizer.step()
  - disable_dropout() при инициализации
"""
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_cosine_schedule_with_warmup

# vLLM-Omni интеграция — опциональная, используется если передан vllm_server
try:
    from vllm_server import VLLMOmniServer
    from lora_sync import LoRASyncManager
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


def disable_dropout(model: torch.nn.Module) -> None:
    """
    Отключает все Dropout слои в модели — приём из TRL GRPOTrainer.

    Зачем: при обучении generation и logprob computation должны использовать
    одно и то же распределение π_θ(c|p). С включённым dropout каждый forward
    pass применяет случайную маску → generation и logprob computation
    видят разные версии модели → policy gradient signal некорректен.

    TRL решает это отключением dropout раз и навсегда при инициализации,
    а не переключением eval/train mode на каждом шаге.
    Gradient checkpointing и LoRA при этом работают нормально.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def _expand_inputs(
    model_inputs: Dict[str, Any],
    G: int,
) -> Dict[str, Any]:
    """
    Расширяет входной батч из B примеров в B*G для GRPO generation/logprob.

    КРИТИЧНО: Qwen3-Omni содержит два типа тензоров с разной семантикой
    первой размерности:

    1. Batch-индексированные (первая dim = batch_size B):
       input_ids, attention_mask, position_ids, rope_deltas, token_type_ids
       Эти нужно повторить ПОЭЛЕМЕНТНО: repeat_interleave(G, dim=0)
       [a, b] → [a, a, a, b, b, b]  (для G=3)

    2. Мультимодальные (первая dim ≠ batch_size):
       pixel_values      [total_visual_tokens, C]  ← кол-во токенов всех видео
       video_grid_thw    [num_videos, 3]            ← T,H,W каждого видео
       audio_features    [total_audio_frames, D]
       audio_seqlens     [num_audios]
       Модель читает video_grid_thw чтобы нарезать pixel_values по видео.
       Эти нужно ТАЙЛИТЬ БЛОКОМ: cat([v]*G, dim=0)
       [a, b] → [a, b, a, b, a, b]  (для G=3)

    Используем явный whitelist batch-ключей вместо эвристики по shape[0].
    Эвристика хрупкая: при batch_size=1 любой тензор с shape[0]=1 попадал бы
    в repeat_interleave, что давало неверный результат для некоторых edge cases.
    """
    # Явный список ключей где первая размерность = batch_size.
    # Всё остальное — мультимодальные тензоры, тайлим блоком.
    #
    # Реальные ключи Qwen3OmniMoeProcessor (проверено на живом батче):
    #   BATCH_KEYS (repeat_interleave):
    #     input_ids [B, seq], attention_mask [B, seq]
    #   Блочное тайлирование (cat):
    #     pixel_values_videos [total_frames, 1536]
    #     video_grid_thw      [num_videos, 3]
    #     video_second_per_grid [num_videos]
    #     feature_attention_mask [num_videos, 1516]
    #     input_features      [num_videos, 128, 1516]
    BATCH_KEYS = {
        "input_ids", "attention_mask",
        "position_ids", "rope_deltas", "token_type_ids",
    }

    # Debug: логируем ключи и shapes при первом вызове для проверки
    # что все мультимодальные тензоры правильно классифицированы.
    # Можно включить через logging.DEBUG при первом запуске.
    if logger.isEnabledFor(logging.DEBUG):
        shapes = {
            k: (v.shape if isinstance(v, torch.Tensor) else type(v).__name__)
            for k, v in model_inputs.items()
        }
        logger.debug(f"_expand_inputs keys/shapes: {shapes}")

    expanded = {}
    for k, v in model_inputs.items():
        if not isinstance(v, torch.Tensor):
            if isinstance(v, list):
                # repeat_interleave для списков: [a,b] → [a,a,b,b] при G=2
                expanded[k] = [item for item in v for _ in range(G)]
            else:
                expanded[k] = v
        elif k in BATCH_KEYS:
            # Batch-индексированные: [a,b] → [a,a,b,b]
            expanded[k] = v.repeat_interleave(G, dim=0)
        else:
            # Мультимодальные тензоры (pixel_values, video_grid_thw и т.д.)
            # КРИТИЧНО: тоже repeat_interleave, не cat([v]*G).
            # cat даёт [a,b,a,b] — неверный порядок при batch_size>1.
            # repeat_interleave даёт [a,a,b,b] — совпадает с текстом.
            expanded[k] = v.repeat_interleave(G, dim=0)

    return expanded


# ══════════════════════════════════════════════════════════════════════════════
# КОЛЛАТОР
# ══════════════════════════════════════════════════════════════════════════════

class ViralityCollator:
    """
    Коллатор для батча пар видео.

    Pipeline:
      DataLoader item (dict с путями)
      -> process_mm_info (декодирует видео @ 1fps + аудио)
      -> apply_chat_template (строит ChatML строку)
      -> processor() (токенизирует + паддит + pixel_values + audio_features)
      -> тензоры на GPU

    КРИТИЧНО: use_audio_in_video=True должен быть одинаковым здесь и в generate().
    Qwen3-Omni при use_audio_in_video=True синхронизирует аудио-токены с видео-токенами
    на уровне positional encoding — рассинхрон ломает мультимодальный attention.
    """

    def __init__(
        self,
        processor,
        use_audio_in_video: bool = True,
        max_length: int = 65000,
        device: str = "cuda",
    ):
        # video_fps не нужен: fps вшит в conversation dict через build_conversation()
        self.processor = processor
        self.use_audio_in_video = use_audio_in_video
        self.max_length = max_length
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from qwen_omni_utils import process_mm_info
        import time

        texts, all_audios, all_images, all_videos = [], [], [], []

        for item in batch:
            conv = item["prompt"]

            # process_mm_info:
            # - декодирует видео из path -> list[PIL.Image] @ fps
            # - если use_audio_in_video=True: извлекает аудио -> np.array
            logger.info("Collator: начинаем process_mm_info...")
            _t0 = time.time()
            audios, images, videos = process_mm_info(
                conv,
                use_audio_in_video=self.use_audio_in_video,
            )
            logger.info(f"Collator: process_mm_info done — {time.time() - _t0:.1f}s")

            # Если аудио отключено — явно зануляем результат process_mm_info
            # и чистим аудио из conv чтобы processor() не нашёл его самостоятельно
            if not self.use_audio_in_video:
                audios = []
                for msg in conv:
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "audio":
                                part.pop("audio", None)

            logger.info("Collator: начинаем apply_chat_template...")
            _t1 = time.time()
            text = self.processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
                # Thinking модель: <think> всегда включён, флаг не нужен
            )
            logger.info(f"Collator: apply_chat_template done — {time.time() - _t1:.1f}s")

            texts.append(text)
            if audios:
                all_audios.extend(audios)
            if images:
                all_images.extend(images)
            if videos:
                all_videos.extend(videos)

        logger.info("Collator: начинаем processor()...")
        _t2 = time.time()
        model_inputs = self.processor(
            text=texts,
            audio=all_audios if (all_audios and self.use_audio_in_video) else None,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            return_tensors="pt",
            padding=True,
            # truncation и max_length намеренно не передаются.
            #
            # Qwen3OmniMoeProcessor вычисляет pixel_values и video_grid_thw
            # ДО токенизации и не знает о max_length. Если передать truncation=True,
            # tokenizer обрежет input_ids (вырезав vision placeholder tokens), но
            # pixel_values/video_grid_thw останутся полными → несоответствие размеров
            # → крэш или тихо неверный multimodal attention.
            #
            # Размер входа контролируется на уровне данных:
            # - max_video_duration=90s @ 1fps → ≤90 кадров
            # - два видео → ~34k визуальных токенов, хорошо в пределах 65k контекста
            # - если нужно уменьшить: processor.max_pixels (ресайз кадров)
            use_audio_in_video=self.use_audio_in_video,
        )
        logger.info(f"Collator: processor() done — {time.time() - _t2:.1f}s")

        # Если аудио отключено — удаляем audio фичи которые processor всё равно
        # создаёт из видео. Без этого audio tower обрабатывает их в generate()
        # и _compute_logprobs, загружая CPU на 100% при каждом токене.
        if not self.use_audio_in_video:
            model_inputs.pop("input_features", None)
            model_inputs.pop("feature_attention_mask", None)
            logger.info("Collator: audio features removed from batch (use_audio_in_video=False)")

        # Защитный assert: если вход всё-таки превысил контекст — лучше упасть явно,
        # чем молча кормить модель обрезанным промптом без мультимодальных токенов.
        seq_len = model_inputs["input_ids"].shape[1]
        if seq_len > self.max_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_length={self.max_length}. "
                f"Уменьши длину видео, fps, или processor.max_pixels."
            )

        # input_features (аудио) процессор возвращает в float32.
        # Веса audio_tower в bfloat16 (NF4 квантизация) ->
        # RuntimeError: Input type (float) and bias type (BFloat16) should be the same.
        # Кастим float32 тензоры в bfloat16 при переносе на GPU.
        FLOAT_KEYS = {'input_features', 'pixel_values_videos'}
        model_inputs = {
            k: (v.to(self.device, dtype=torch.bfloat16)
                if isinstance(v, torch.Tensor) and k in FLOAT_KEYS
                else v.to(self.device) if isinstance(v, torch.Tensor)
                else v)
            for k, v in model_inputs.items()
        }

        model_inputs["labels_text"] = [item["label"] for item in batch]
        model_inputs["views_a"] = [item["views_a"] for item in batch]
        model_inputs["views_b"] = [item["views_b"] for item in batch]
        model_inputs["video_a"] = [item["video_a"] for item in batch]
        model_inputs["video_b"] = [item["video_b"] for item in batch]
        model_inputs["author_context"] = [item.get("author_context", "") for item in batch]

        return model_inputs


# ══════════════════════════════════════════════════════════════════════════════
# ТРЕНЕР
# ══════════════════════════════════════════════════════════════════════════════

class OmniGRPOTrainer:
    """
    GRPO тренер для Qwen3-Omni Thinker.

    Почему не GRPOTrainer из TRL:
    TRL.GRPOTrainer токенизирует промпты через tokenizer(text) — это ломается
    на мультимодальных входах где нужен processor(text, videos, audio).

    Используем чистый GRPO без ref модели:
    - loss = -E[advantage * log π_policy(c|p)]
    - стабильность обеспечивается нормализацией advantages + clip_grad_norm
    - экономия ~15GB vs варианта с ref моделью
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor,
        reward_fn: Callable,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_generations: int = 4,
        max_completion_length: int = 600,
        learning_rate: float = 5e-6,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 2,
        warmup_ratio: float = 0.05,
        use_audio_in_video: bool = True,
        output_dir: str = "./checkpoints",
        logging_steps: int = 5,
        save_steps: int = 100,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        use_wandb: bool = False,
        temperature: float = 0.9,
        # vLLM-Omni интеграция — опциональная
        # Если передан vllm_server — генерация идёт через vLLM-Omni (быстро)
        # Если None — генерация через HuggingFace generate() (медленно)
        vllm_server=None,
        lora_sync=None,
    ):
        self.model = model
        self.processor = processor
        self.reward_fn = reward_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.G = num_generations
        self.max_completion_length = max_completion_length
        self.use_audio_in_video = use_audio_in_video
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.use_wandb = use_wandb
        self.temperature = temperature
        self.global_step = 0
        self._accum_step = 0

        # vLLM-Omni интеграция
        self.vllm_server = vllm_server
        self.lora_sync = lora_sync
        if vllm_server is not None:
            logger.info("vLLM-Omni mode: generation via vLLM-Omni server")
        else:
            logger.info("HuggingFace mode: generation via model.generate()")

        os.makedirs(output_dir, exist_ok=True)

        # Отключаем dropout раз и навсегда — приём из TRL.
        # Гарантирует что generation и logprob computation используют
        # одинаковое распределение без переключения eval/train на каждом шаге.
        disable_dropout(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = max(
            1, (len(train_dataloader) * num_train_epochs + gradient_accumulation_steps - 1)
            // gradient_accumulation_steps
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_completions(
        self,
        model_inputs: Dict[str, torch.Tensor],
        batch_raw: Optional[Dict] = None,
    ) -> Tuple[List[str], torch.Tensor, Dict[str, Any]]:
        """
        Генерирует G completion'ов через vLLM-Omni.

        Требует self.vllm_server — без него упадёт явно.
        HF fallback удалён намеренно: 350 секунд на батч делает
        обучение нереальным, молчаливый fallback маскирует проблему.
        """
        if self.vllm_server is None:
            raise RuntimeError(
                "vLLM-Omni сервер не инициализирован. "
                "Запусти train.py с флагом --use_vllm и укажи --vllm_model_path."
            )
        if batch_raw is None:
            raise RuntimeError(
                "batch_raw не передан в _generate_completions. "
                "Это баг в _train_step — video_a/video_b должны приходить из коллатора."
            )
        return self._generate_via_vllm(model_inputs, batch_raw)

    def _generate_via_vllm(
        self,
        model_inputs: Dict[str, torch.Tensor],
        batch_raw: Dict,
    ) -> Tuple[List[str], torch.Tensor, Dict[str, Any]]:
        """
        Генерация через vLLM-Omni сервер.

        КРИТИЧНО: промпт строится через build_conversation + apply_chat_template —
        тот же путь что и в коллаторе. Это гарантирует что токены промпта
        в vLLM и в _compute_logprobs идентичны.
        """
        from dataset import build_conversation

        video_a = batch_raw["video_a"][0]  # batch_size=1
        video_b = batch_raw["video_b"][0]
        author_context = batch_raw.get("author_context", [""])[0]

        # Строим conversation тем же способом что коллатор
        item = {
            "video_a": video_a,
            "video_b": video_b,
            "author_context": author_context,
        }
        conv = build_conversation(item, video_fps=1)

        # Путь к текущему LoRA адаптеру для vLLM-Omni
        lora_path = str(self.lora_sync.sync_path) if self.lora_sync else None

        logger.info(f"vLLM-Omni generate: G={self.G}, video_a={video_a}, video_b={video_b}")
        t0 = __import__("time").time()

        completions = self.vllm_server.generate(
            video_a_path=video_a,
            video_b_path=video_b,
            conversation=conv,
            G=self.G,
            max_new_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=0.95,
            lora_adapter_path=lora_path,
        )

        logger.info(f"vLLM-Omni generate done in {__import__('time').time() - t0:.1f}s")

        # Токенизируем completions для _compute_logprobs
        completion_ids = self._tokenize_completions(completions, model_inputs)

        # Расширяем входы до B*G для _compute_logprobs
        expanded = _expand_inputs(model_inputs, self.G)

        return completions, completion_ids, expanded

    def _tokenize_completions(
        self,
        completions: List[str],
        model_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Токенизирует текстовые completions от vLLM-Omni в тензор ids.
        Нужно для _compute_logprobs который ожидает completion_ids тензор.
        """
        device = model_inputs["input_ids"].device
        encoded = self.processor.tokenizer(
            completions,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        return encoded["input_ids"].to(device)  # [G, max_len]

    @torch.no_grad()
    def _generate_via_hf(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[str], torch.Tensor, Dict[str, Any]]:
        """
        Fallback генерация через HuggingFace model.generate().
        Медленно (~350 секунд на пару видео) но не требует vLLM-Omni.
        """
        prompt_len = model_inputs["input_ids"].shape[1]

        self.model.eval()
        self.model.gradient_checkpointing_disable()

        logger.info(f"HF generate() seq_len={model_inputs['input_ids'].shape[1]}, G={self.G}")

        all_completion_ids = []
        all_completions = []

        try:
            for g in range(self.G):
                logger.info(f"HF generate() call {g+1}/{self.G}...")
                output_ids = self.model.generate(
                    **model_inputs,
                    use_cache=True,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_audio_in_video=self.use_audio_in_video,
                )
                c_ids = output_ids[:, prompt_len:].contiguous()
                all_completion_ids.append(c_ids)
                texts = self.processor.tokenizer.batch_decode(
                    c_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                all_completions.extend(texts)
                logger.info(f"HF generate() call {g+1}/{self.G} done")
        finally:
            self.model.train()
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        max_len = max(c.shape[1] for c in all_completion_ids)
        padded = []
        for c in all_completion_ids:
            if c.shape[1] < max_len:
                pad = torch.full(
                    (c.shape[0], max_len - c.shape[1]),
                    self.processor.tokenizer.pad_token_id,
                    device=c.device, dtype=c.dtype,
                )
                c = torch.cat([c, pad], dim=1)
            padded.append(c)
        completion_ids = torch.cat(padded, dim=0)

        expanded = _expand_inputs(model_inputs, self.G)
        return all_completions, completion_ids, expanded

    # ──────────────────────────────────────────────────────────────────────────
    # LOG-PROBABILITIES
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_logprobs(
        self,
        expanded_inputs: Dict[str, torch.Tensor],
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Суммарный нормализованный log P(completion | prompt) для policy модели.

        Используется только для policy (ref модели нет).
        Нормализация по длине устраняет bias в advantages в сторону коротких генераций.

        Returns: [B*G]
        """
        prompt_len = expanded_inputs["input_ids"].shape[1]
        comp_len = completion_ids.shape[1]
        bg = completion_ids.shape[0]

        # Проверяем что все тензоры в expanded_inputs имеют правильную batch dim.
        # Нетензорные поля (списки, numpy) не расширяются repeat_interleave →
        # несоответствие batch dimension ломает forward() с cryptic error внутри attention.
        assert expanded_inputs["input_ids"].shape[0] == bg, (
            f"Batch dim mismatch: input_ids has {expanded_inputs['input_ids'].shape[0]}, "
            f"completion_ids has {bg} (expected B*G={bg}). "
            f"Возможно нетензорное поле не расширилось в _train_step."
        )

        full_input_ids = torch.cat(
            [expanded_inputs["input_ids"], completion_ids], dim=1
        )
        comp_mask = torch.ones(
            bg, comp_len,
            dtype=expanded_inputs["attention_mask"].dtype,
            device=completion_ids.device,
        )
        full_attention_mask = torch.cat(
            [expanded_inputs["attention_mask"], comp_mask], dim=1
        )

        full_inputs = {
            k: v for k, v in expanded_inputs.items()
            if k not in ("input_ids", "attention_mask",
                         # Позиционные тензоры processor вычислил только для промпта.
                         # При передаче в full-sequence forward (промпт + completion)
                         # длина не совпадает → RuntimeError или неверные RoPE позиции
                         # для completion токенов. Исключаем — модель пересчитает сама.
                         "position_ids", "rope_deltas", "token_type_ids")
        }
        full_inputs["input_ids"] = full_input_ids
        full_inputs["attention_mask"] = full_attention_mask

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**full_inputs, use_audio_in_video=self.use_audio_in_video)

        # КРИТИЧНО: outputs.logits = [B*G, full_seq_len, vocab] ~ 41GB bf16.
        # Шаги:
        # 1. Вырезаем нужный срез как VIEW (новая память не аллоцируется).
        # 2. .contiguous() копирует только срез (~1GB bf16) в отдельный тензор.
        # 3. del outputs освобождает 41GB ДО конвертации в float32.
        # 4. Конвертируем маленький срез (~1GB) в float32 (~2GB).
        # Без .contiguous() + del шаги 3-4 держали бы в памяти 41GB + 2GB одновременно.
        comp_logits_bf16 = outputs.logits[
            :, prompt_len - 1: prompt_len - 1 + comp_len, :
        ].contiguous()
        del outputs  # free ~41GB BEFORE float conversion
        comp_logits = comp_logits_bf16.float()
        del comp_logits_bf16

        # КРИТИЧНО: делим на temperature перед log_softmax.
        # Генерация использует softmax(logits / T), поэтому logprob computation
        # должен использовать те же логиты. Без деления получаем logprob под T=1.0
        # при completions сэмплированных из T=0.9 → mismatch распределений →
        # policy gradient толкает в неверном направлении.
        # TRL делает то же самое (см. grpo_trainer.py: "Divide logits by sampling temperature")
        if self.temperature != 1.0:
            comp_logits = comp_logits / self.temperature

        log_probs_all = F.log_softmax(comp_logits, dim=-1)
        del comp_logits  # free ~2GB float32

        token_log_probs = log_probs_all.gather(
            dim=2,
            index=completion_ids.unsqueeze(-1).long(),
        ).squeeze(-1)  # [B*G, comp_len]
        del log_probs_all  # free ~1.4GB — после gather больше не нужен

        # Маскируем только trailing pad ПОСЛЕ EOS, не сам EOS токен.
        # Если pad_id == eos_id (наш случай), наивная маска обнулила бы EOS
        # и модель не получала бы градиент для обучения завершению генерации.
        eos_id = self.processor.tokenizer.eos_token_id
        is_eos = (completion_ids == eos_id).long()
        # cumsum - is_eos: 0 до EOS включительно, 1+ после
        trailing_mask = is_eos.cumsum(dim=1) - is_eos
        active_mask = (trailing_mask == 0).float()  # [B*G, comp_len]

        # Нормализуем по числу активных токенов.
        # Без нормализации абсолютный logprob пропорционален длине completion:
        # короткий <think> получает более высокий (менее отрицательный) logprob
        # → систематический bias в advantages в сторону коротких генераций.
        active_len = active_mask.sum(dim=-1).clamp(min=1.0)
        return (token_log_probs * active_mask).sum(dim=-1) / active_len  # [B*G]

    # ──────────────────────────────────────────────────────────────────────────
    # GRPO LOSS
    # ──────────────────────────────────────────────────────────────────────────

    def _grpo_loss(
        self,
        policy_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Чистый GRPO objective: loss = -E[advantage * log π_policy(c|p)]

        Нет ref модели, нет ratio, нет KL penalty.
        Стабильность обеспечивается нормализацией advantages внутри группы
        и clip_grad_norm_ в train loop.

        Если все G completion'ов получили одинаковый reward → std=0 →
        advantages=0 → loss=0. Правильно: нечему учиться.
        """
        r = rewards.view(batch_size, self.G)

        # Нормализация внутри группы — ключевая идея GRPO.
        # advantages всегда в диапазоне ~[-2, +2] независимо от масштаба наград.
        adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)
        # detach: advantages — константы-веса, не часть оптимизируемого пути.
        # Без detach при дифференцируемых rewards (напр. reward model) градиент
        # потёк бы через advantages, что математически неверно для policy gradient.
        adv = adv.detach().view(-1)  # [B*G]

        # Чистый policy gradient: поощряем completion'ы с adv > 0,
        # штрафуем с adv < 0, пропорционально их вероятности.
        policy_loss = -(adv * policy_logprobs).mean()

        metrics = {
            "loss/policy": policy_loss.item(),
            "train/adv_std": adv.std().item(),
            # reward_range = разброс наград внутри группы G.
            # Падение к 0 → все G completion'ов одинаковые → нечему учиться.
            # adv_mean не логируем — он всегда 0 по построению нормализации.
            "train/reward_range": (
                r.max(dim=1).values - r.min(dim=1).values
            ).mean().item(),
        }

        return policy_loss, metrics

    # ──────────────────────────────────────────────────────────────────────────
    # TRAINING STEP
    # ──────────────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Dict) -> Dict:
        # Не мутируем dict из DataLoader — делаем копию без метаданных
        labels_text = batch["labels_text"]
        # batch_raw нужен для vLLM-Omni режима — содержит пути к видео
        batch_raw = {
            "video_a": batch.get("video_a"),
            "video_b": batch.get("video_b"),
            "author_context": batch.get("author_context"),
        }
        model_inputs = {
            k: v for k, v in batch.items()
            if k not in ("labels_text", "views_a", "views_b", "video_a", "video_b", "author_context")
        }
        batch_size = model_inputs["input_ids"].shape[0]

        # 1. Генерация
        completions, completion_ids, expanded_inputs = self._generate_completions(
            model_inputs, batch_raw=batch_raw
        )
        torch.cuda.empty_cache()

        # 2. Reward
        labels_expanded = [lbl for lbl in labels_text for _ in range(self.G)]
        raw_rewards = self.reward_fn(
            completions=completions,
            labels_text=labels_expanded,
        )
        rewards = torch.tensor(raw_rewards, dtype=torch.float32,
                               device=completion_ids.device)

        # 3. Log probs — используем expanded_inputs из шага 1, не expand снова
        policy_logprobs = self._compute_logprobs(expanded_inputs, completion_ids)

        # 4. GRPO Loss + backward
        loss, metrics = self._grpo_loss(policy_logprobs, rewards, batch_size)
        (loss / self.gradient_accumulation_steps).backward()

        metrics["reward/mean"] = rewards.mean().item()
        metrics["reward/std"] = rewards.std().item()
        metrics["reward/pos_frac"] = (rewards > 0).float().mean().item()

        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        logger.info("=" * 60)
        logger.info("Starting GRPO training")
        logger.info(f"  Epochs: {self.num_train_epochs}")
        logger.info(f"  Steps/epoch: {len(self.train_dataloader)}")
        logger.info(f"  G (completions per prompt): {self.G}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 60)

        self.model.train()
        self.optimizer.zero_grad()
        accumulated_metrics: Dict[str, float] = {}

        for epoch in range(self.num_train_epochs):
            logger.info(f"\n-- Epoch {epoch + 1}/{self.num_train_epochs} --")

            for step, batch in enumerate(self.train_dataloader):
                try:
                    metrics = self._train_step(batch)
                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        "OOM на шаге %d. Уменьши длину видео или G=%d.",
                        self.global_step, self.G,
                    )
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    self._accum_step = 0
                    accumulated_metrics = {}  # сбрасываем частичную сумму
                    continue

                self._accum_step += 1

                # Накапливаем метрики — усредняем по всем батчам в окне аккумуляции
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v / self.gradient_accumulation_steps

                if self._accum_step % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Синхронизируем LoRA веса в vLLM-Omni после каждого шага
                    if self.lora_sync is not None:
                        self.lora_sync.sync(self.model)

                    if self.global_step % self.logging_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        parts = [f"step={self.global_step}", f"lr={lr:.2e}"]
                        parts += [f"{k}={v:.4f}" for k, v in accumulated_metrics.items()]
                        logger.info(" | ".join(parts))

                        if self.use_wandb:
                            try:
                                import wandb
                                wandb.log({
                                    "train/loss":            accumulated_metrics.get("loss/policy", 0),
                                    "train/reward_mean":     accumulated_metrics.get("reward/mean", 0),
                                    # reward/std → главная метрика policy collapse:
                                    # падение к 0 означает что все G ответов одинаковые
                                    "train/reward_std":      accumulated_metrics.get("reward/std", 0),
                                    "train/reward_pos_frac": accumulated_metrics.get("reward/pos_frac", 0),
                                    "train/adv_std":         accumulated_metrics.get("train/adv_std", 0),
                                    "train/reward_range":    accumulated_metrics.get("train/reward_range", 0),
                                    "train/lr":              lr,
                                }, step=self.global_step)
                            except Exception as e:
                                logger.warning(f"wandb.log failed: {e}")

                    accumulated_metrics = {}

                    if self.global_step % self.save_steps == 0:
                        self.evaluate()
                        self._save_checkpoint()

            # Флашим остаток аккумуляции в конце эпохи.
            # Без этого последние (N % grad_accum) батчей каждой эпохи
            # накапливают градиент, который молча выбрасывается на следующем zero_grad().
            if self._accum_step % self.gradient_accumulation_steps != 0:
                remainder = self._accum_step % self.gradient_accumulation_steps
                logger.info(
                    f"Epoch-end grad flush: {remainder} батчей → "
                    f"optimizer.step() at global_step={self.global_step + 1}"
                )
                # Нормализуем accumulated_metrics под фактическое число батчей
                # (не gradient_accumulation_steps — их было меньше)
                if accumulated_metrics and remainder > 0:
                    scale = self.gradient_accumulation_steps / remainder
                    accumulated_metrics = {k: v * scale for k, v in accumulated_metrics.items()}

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self._accum_step = 0

                # Логируем остаток — эти батчи иначе исчезают из метрик
                lr = self.scheduler.get_last_lr()[0]
                parts = [f"step={self.global_step}(epoch-end)", f"lr={lr:.2e}"]
                parts += [f"{k}={v:.4f}" for k, v in accumulated_metrics.items()]
                logger.info(" | ".join(parts))
                if self.use_wandb and accumulated_metrics:
                    try:
                        import wandb
                        wandb.log({
                            "train/loss":            accumulated_metrics.get("loss/policy", 0),
                            "train/reward_mean":     accumulated_metrics.get("reward/mean", 0),
                            "train/reward_std":      accumulated_metrics.get("reward/std", 0),
                            "train/reward_pos_frac": accumulated_metrics.get("reward/pos_frac", 0),
                            "train/adv_std":         accumulated_metrics.get("train/adv_std", 0),
                            "train/reward_range":    accumulated_metrics.get("train/reward_range", 0),
                            "train/lr":              lr,
                        }, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"wandb.log failed: {e}")
                accumulated_metrics = {}

        logger.info("Training complete.")
        self._save_checkpoint(final=True)

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Eval loop: accuracy на eval датасете.

        Используем 1 greedy completion на промпт (не G) — для метрики
        точности нам нужна детерминированная оценка, не стохастическая.
        try/finally гарантирует возврат в train mode даже при исключении.
        """
        if self.eval_dataloader is None:
            return {}
        self.model.eval()
        all_rewards = []
        try:
            for batch in self.eval_dataloader:
                labels_text = batch["labels_text"]
                model_inputs = {
                    k: v for k, v in batch.items()
                    if k not in ("labels_text", "views_a", "views_b")
                }
                # 1 greedy completion per prompt для eval
                prompt_len = model_inputs["input_ids"].shape[1]
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_completion_length,
                    do_sample=False,  # greedy — детерминировано
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_audio_in_video=self.use_audio_in_video,
                    # return_audio не нужен — Thinker класс не умеет в аудио output
                )
                completion_ids = output_ids[:, prompt_len:]
                completions = self.processor.tokenizer.batch_decode(
                    completion_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                rewards = self.reward_fn(
                    completions=completions,
                    labels_text=labels_text,
                )
                all_rewards.extend(rewards)
        finally:
            self.model.train()  # гарантированно выходим из eval mode

        accuracy = sum(r > 0 for r in all_rewards) / max(len(all_rewards), 1)
        mean_reward = sum(all_rewards) / max(len(all_rewards), 1)
        logger.info(f"Eval | accuracy={accuracy:.3f} | mean_reward={mean_reward:.3f}")

        if self.use_wandb:
            try:
                import wandb
                wandb.log({
                    "eval/accuracy": accuracy,
                    "eval/reward_mean": mean_reward,
                }, step=self.global_step)
            except Exception as e:
                logger.warning(f"wandb.log failed: {e}")

        return {"eval/accuracy": accuracy, "eval/reward_mean": mean_reward}

    def _save_checkpoint(self, final: bool = False) -> None:
        name = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.output_dir, f"checkpoint-{name}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        # Processor статичен (не обучается) — сохраняем только при финальном чекпоинте
        if final:
            self.processor.save_pretrained(path)
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "accum_step": self._accum_step,
        }, os.path.join(path, "trainer_state.pt"))
        # Обновляем symlink checkpoint-latest → текущий чекпоинт для --resume
        latest_link = os.path.join(self.output_dir, "checkpoint-latest")
        try:
            if os.path.islink(latest_link):
                os.remove(latest_link)
            os.symlink(os.path.abspath(path), latest_link)
        except Exception as e:
            logger.warning(f"Не удалось обновить checkpoint-latest symlink: {e}")
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        # Сначала веса — потом optimizer state.
        # Порядок критичен: optimizer моменты (Adam m1/m2) должны соответствовать
        # весам из чекпоинта. Если грузить только optimizer без весов — моменты
        # относятся к старым весам, что ломает адаптивный lr с первого же шага.
        logger.info(f"Loading LoRA adapter weights from {checkpoint_path}")
        self.model.load_adapter(checkpoint_path, adapter_name="default")

        state_file = os.path.join(checkpoint_path, "trainer_state.pt")
        if not os.path.exists(state_file):
            logger.warning(f"No trainer_state.pt in {checkpoint_path}, skipping optimizer restore")
            return
        state = torch.load(state_file, map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self._accum_step = state["accum_step"]
        logger.info(f"Resumed from step {self.global_step}")