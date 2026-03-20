"""
dataset.py — датасет и препроцессор для пар видео.

Ожидаемый формат JSONL (один пример = одна строка):
{
    "video_a": "/path/to/viral_video.mp4",
    "video_b": "/path/to/average_video.mp4",
    "views_a": 1500000,
    "views_b": 42000,
    "author_context": "фитнес блогер, продаёт курсы по питанию, постит 3 раза в неделю"
}

Агент-персонализация: когда будет готов агент, его вывод добавляется в author_context.
"""
import json
import logging
import random as _random
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

try:
    import decord as _decord
except ImportError:
    _decord = None  # duration check will be skipped if decord not installed

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Системный промпт
# Цель: дать модели чёткий фрейм задачи БЕЗ излишних инструкций
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert at predicting social media virality on Instagram. "
    "You will see two short videos from the same creator and must decide which one "
    "received significantly more views. Analyze visual quality, hook strength, "
    "editing pace, audio, content clarity, and emotional resonance. "
    "Be concise in your reasoning — focus only on the factors that truly drive views. "
    "Always respond in the format:\n"
    "<think>\n[your analysis]\n</think>\n"
    "<answer>A</answer>  or  <answer>B</answer>"
)


def build_conversation(
    item: Dict[str, Any],
    video_fps: int = 1,
    agent_context: Optional[str] = None,  # будущий хук для агента
) -> List[Dict]:
    """
    Строим разговор в формате Qwen3-Omni ChatML с видео.

    Порядок важен: видео идут ПЕРЕД текстом (так требует модель).
    use_audio_in_video=True обрабатывается на уровне process_mm_info,
    здесь только указываем пути.
    """
    author_ctx = item.get("author_context", "")
    if agent_context:
        # Агент добавляет персонализированный контекст
        author_ctx = f"{author_ctx}\nAgent insights: {agent_context}"

    user_content = [
        # Video A
        {
            "type": "video",
            "video": item["video_a"],
            "fps": video_fps,
        },
        # Video B
        {
            "type": "video",
            "video": item["video_b"],
            "fps": video_fps,
        },
        # Текстовый запрос ПОСЛЕ видео
        {
            "type": "text",
            "text": (
                f"Creator profile: {author_ctx}\n\n"
                "Video A (first video above) vs Video B (second video above) — "
                "both from the same creator. Which received significantly more views?\n\n"
                "Think through the key factors, then give your answer."
            ),
        },
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class ViralityDataset(Dataset):
    """
    Датасет пар видео для GRPO обучения.

    Каждый элемент возвращает:
    - prompt: список сообщений (для apply_chat_template)
    - label: "A" или "B" (ground truth)
    - video_a, video_b: пути к файлам (нужны коллатору)
    - metadata: для дебага и логирования
    """

    def __init__(
        self,
        data_path: str,
        video_fps: int = 1,
        max_video_duration: float = 90.0,  # секунды — фильтрует слишком длинные видео
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.video_fps = video_fps
        self.max_video_duration = max_video_duration
        self.samples: List[Dict] = []

        self._load(max_samples)
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def _load(self, max_samples: Optional[int]) -> None:
        skipped = 0
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Датасет приходит уже отфильтрованным после prepare_dataset.py.
                    # Единственная проверка здесь — длительность видео,
                    # чтобы не улететь за лимит контекста модели.
                    too_long = False
                    for vid_key in ("video_a", "video_b"):
                        dur = self._get_duration(item[vid_key])
                        if dur is not None and dur > self.max_video_duration:
                            logger.warning(
                                f"Video too long ({dur:.1f}s > {self.max_video_duration}s): {item[vid_key]}"
                            )
                            too_long = True
                            break
                    if too_long:
                        skipped += 1
                    else:
                        item["label"] = "A" if item["views_a"] > item["views_b"] else "B"
                        self.samples.append(item)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed line: {e}")
                    skipped += 1

                if max_samples and len(self.samples) >= max_samples:
                    break

        if skipped:
            logger.warning(f"Skipped {skipped} invalid samples")

    def _get_duration(self, path: str) -> Optional[float]:
        """Быстро читает длительность видео через decord (только header)."""
        if _decord is None:
            return None
        try:
            vr = _decord.VideoReader(path)
            fps = vr.get_avg_fps()
            return len(vr) / fps if fps > 0 else None
        except Exception:
            return None  # если не можем прочитать — пропускаем проверку

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.samples[idx])  # копируем — не мутируем self.samples

        # Детерминированный A/B флип по idx.
        #
        # Зачем: если сборщик данных систематически кладёт вирусное видео
        # в video_a (что естественно при парсинге), модель выучит позиционный
        # байас "A почти всегда правильный ответ" вместо признаков виральности.
        #
        # Почему random.Random(idx), а не глобальный random:
        # - Детерминировано: один и тот же idx всегда даёт одинаковый флип.
        # - Безопасно при resume: при продолжении с чекпоинта пара idx=42
        #   перевернётся так же, как при первом проходе — датасет не меняется.
        # - Не зависит от порядка обхода: shuffle в DataLoader не влияет.
        # - ~50% пар будут A=вирусное, ~50% пар будут B=вирусное.
        rng = _random.Random(idx)
        if rng.random() < 0.5:
            item["video_a"], item["video_b"] = item["video_b"], item["video_a"]
            item["views_a"], item["views_b"] = item["views_b"], item["views_a"]

        # Пересчитываем label ПОСЛЕ флипа
        item["label"] = "A" if item["views_a"] > item["views_b"] else "B"

        conversation = build_conversation(item, self.video_fps)

        return {
            # Основные поля для GRPO
            "prompt": conversation,       # List[Dict] — messages
            "label": item["label"],       # "A" или "B"
            # Пути нужны коллатору для multimodal preprocessing
            "video_a": item["video_a"],
            "video_b": item["video_b"],
            # Метаданные (не участвуют в обучении, но нужны для логов)
            "views_a": item["views_a"],
            "views_b": item["views_b"],
            "author_context": item.get("author_context", ""),
        }