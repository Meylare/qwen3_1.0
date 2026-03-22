"""
config.py — все гиперпараметры в одном месте.
Меняй здесь, не трогай остальной код.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    model_path: str = "/home/ubuntu/models/qwen3-omni-30b-thinking-awq-4bit"

    # QLoRA: base model в 4-bit, train только LoRA адаптеры
    # На 96GB это даёт ~15GB для base model, остальное под KV cache + activations
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Только attention layers — MoE FFN слишком дорого
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

@dataclass
class DataConfig:
    # Пути к данным передаются через --train_data / --eval_data аргументы train.py
    # 1 fps как договорились
    video_fps: int = 1
    use_audio_in_video: bool = True

    # Максимальная длина видео (секунды). Instagram Reel ≤ 90s → 90 фреймов @ 1fps
    max_video_duration: float = 91.0

@dataclass
class RewardConfig:
    # Основная логика
    correct_reward: float = 1.0       # угадал какое вирусное
    wrong_reward: float = -1.0        # ошибся
    no_answer_penalty: float = -1.0   # не дал ответ в правильном формате

    # Длина thinking: мягкий штраф после max_tokens через tanh
    thinking_max_tokens: int = 400     # мягкий порог — после него штраф нарастает
    length_penalty_max: float = 0.25   # максимальный штраф за длину

    # Бонус за правильный формат <think>...</think><answer>X</answer>
    format_bonus: float = 0.1

@dataclass
class GRPOHyperParams:
    # G — сколько completion'ов генерируем на один промпт
    # Больше G = лучше сигнал, но дороже. 2 — компромисс для вписывания в 120ч
    num_generations: int = 2

    # Максимальная длина контекста (промпт + completion не должны превышать 66048)
    # Два видео по 45s @ 375 т/кадр = ~33k, + текст ~400 = ~34k промпт
    max_context_length: int = 65000   # передаётся в ViralityCollator
    max_completion_length: int = 600  # thinking + answer

    # Оптимизация
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1  # 1 пара видео за раз
    gradient_accumulation_steps: int = 8  # эффективный batch = 8 пар
    num_train_epochs: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Температура сэмплирования при генерации.
    # ВАЖНО: это же значение используется при вычислении logprob (деление logits на T)
    # чтобы generation и logprob computation работали под одним распределением π_θ(·|p,T).
    temperature: float = 0.9
    # Тренер реализует чистый GRPO без ref модели (стабилизация через
    # нормализацию advantages + clip_grad_norm, без KL-регуляризации).
    # Оставлены для документирования архитектурного решения.
    # kl_coeff: float = 0.05   # UNUSED — ref model удалена
    # cliprange: float = 0.2   # UNUSED — ratio clipping не используется

    # Сохранение
    output_dir: str = "./checkpoints/virality_grpo"
    logging_steps: int = 5
    save_steps: int = 100

    # Seed для воспроизводимости
    seed: int = 42

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOHyperParams = field(default_factory=GRPOHyperParams)
    """
config.py — все гиперпараметры в одном месте.
Меняй здесь, не трогай остальной код.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    model_path: str = "/home/ubuntu/models/qwen3-omni-30b-thinking-awq-4bit"

    # QLoRA: base model в 4-bit, train только LoRA адаптеры
    # На 96GB это даёт ~15GB для base model, остальное под KV cache + activations
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Только attention layers — MoE FFN слишком дорого
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

@dataclass
class DataConfig:
    # Пути к данным передаются через --train_data / --eval_data аргументы train.py
    # 1 fps как договорились
    video_fps: int = 1
    use_audio_in_video: bool = True

    # Максимальная длина видео (секунды). Instagram Reel ≤ 90s → 90 фреймов @ 1fps
    max_video_duration: float = 91.0

@dataclass
class RewardConfig:
    # Основная логика
    correct_reward: float = 1.0       # угадал какое вирусное
    wrong_reward: float = -1.0        # ошибся
    no_answer_penalty: float = -1.0   # не дал ответ в правильном формате

    # Длина thinking: мягкий штраф после max_tokens через tanh
    thinking_max_tokens: int = 400     # мягкий порог — после него штраф нарастает
    length_penalty_max: float = 0.25   # максимальный штраф за длину

    # Бонус за правильный формат <think>...</think><answer>X</answer>
    format_bonus: float = 0.1

@dataclass
class GRPOHyperParams:
    # G — сколько completion'ов генерируем на один промпт
    # Больше G = лучше сигнал, но дороже. 2 — компромисс для вписывания в 120ч
    num_generations: int = 2

    # Максимальная длина контекста (промпт + completion не должны превышать 66048)
    # Два видео по 45s @ 375 т/кадр = ~33k, + текст ~400 = ~34k промпт
    max_context_length: int = 65000   # передаётся в ViralityCollator
    max_completion_length: int = 600  # thinking + answer

    # Оптимизация
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1  # 1 пара видео за раз
    gradient_accumulation_steps: int = 8  # эффективный batch = 8 пар
    num_train_epochs: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Температура сэмплирования при генерации.
    # ВАЖНО: это же значение используется при вычислении logprob (деление logits на T)
    # чтобы generation и logprob computation работали под одним распределением π_θ(·|p,T).
    temperature: float = 0.9
    # Тренер реализует чистый GRPO без ref модели (стабилизация через
    # нормализацию advantages + clip_grad_norm, без KL-регуляризации).
    # Оставлены для документирования архитектурного решения.
    # kl_coeff: float = 0.05   # UNUSED — ref model удалена
    # cliprange: float = 0.2   # UNUSED — ratio clipping не используется

    # Сохранение
    output_dir: str = "./checkpoints/virality_grpo"
    logging_steps: int = 5
    save_steps: int = 100

    # Seed для воспроизводимости
    seed: int = 42

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOHyperParams = field(default_factory=GRPOHyperParams)
    """
config.py — все гиперпараметры в одном месте.
Меняй здесь, не трогай остальной код.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    model_path: str = "/home/ubuntu/models/qwen3-omni-30b-thinking-awq-4bit"

    # QLoRA: base model в 4-bit, train только LoRA адаптеры
    # На 96GB это даёт ~15GB для base model, остальное под KV cache + activations
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Только attention layers — MoE FFN слишком дорого
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

@dataclass
class DataConfig:
    # Пути к данным передаются через --train_data / --eval_data аргументы train.py
    # 1 fps как договорились
    video_fps: int = 1
    use_audio_in_video: bool = True

    # Максимальная длина видео (секунды). Instagram Reel ≤ 90s → 90 фреймов @ 1fps
    max_video_duration: float = 91.0

@dataclass
class RewardConfig:
    # Основная логика
    correct_reward: float = 1.0       # угадал какое вирусное
    wrong_reward: float = -1.0        # ошибся
    no_answer_penalty: float = -1.0   # не дал ответ в правильном формате

    # Длина thinking: мягкий штраф после max_tokens через tanh
    thinking_max_tokens: int = 400     # мягкий порог — после него штраф нарастает
    length_penalty_max: float = 0.25   # максимальный штраф за длину

    # Бонус за правильный формат <think>...</think><answer>X</answer>
    format_bonus: float = 0.1

@dataclass
class GRPOHyperParams:
    # G — сколько completion'ов генерируем на один промпт
    # Больше G = лучше сигнал, но дороже. 2 — компромисс для вписывания в 120ч
    num_generations: int = 2

    # Максимальная длина контекста (промпт + completion не должны превышать 66048)
    # Два видео по 45s @ 375 т/кадр = ~33k, + текст ~400 = ~34k промпт
    max_context_length: int = 65000   # передаётся в ViralityCollator
    max_completion_length: int = 600  # thinking + answer

    # Оптимизация
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1  # 1 пара видео за раз
    gradient_accumulation_steps: int = 8  # эффективный batch = 8 пар
    num_train_epochs: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Температура сэмплирования при генерации.
    # ВАЖНО: это же значение используется при вычислении logprob (деление logits на T)
    # чтобы generation и logprob computation работали под одним распределением π_θ(·|p,T).
    temperature: float = 0.9
    # Тренер реализует чистый GRPO без ref модели (стабилизация через
    # нормализацию advantages + clip_grad_norm, без KL-регуляризации).
    # Оставлены для документирования архитектурного решения.
    # kl_coeff: float = 0.05   # UNUSED — ref model удалена
    # cliprange: float = 0.2   # UNUSED — ratio clipping не используется

    # Сохранение
    output_dir: str = "./checkpoints/virality_grpo"
    logging_steps: int = 5
    save_steps: int = 100

    # Seed для воспроизводимости
    seed: int = 42

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOHyperParams = field(default_factory=GRPOHyperParams)