import torch
import torch.nn as nn
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Tuple, Optional, Union


class ViralPredictorModel(nn.Module):
    def __init__(self, base_model_id, bnb_config):
        super().__init__()

        # 1. Загрузка Backbone (Qwen2.5-Omni-7B)
        self.backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.backbone.gradient_checkpointing_enable()

        # 2. Настройка LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            # Таргетим все проекции для лучшего мультимодального обучения
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        self.backbone.print_trainable_parameters()

        # 3. MLP Head (Блок B)
        # Получаем hidden_size из конфига (может быть в config.hidden_size или config.text_config.hidden_size)
        if hasattr(self.backbone.config, 'hidden_size'):
            hidden_size = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, 'text_config') and hasattr(self.backbone.config.text_config, 'hidden_size'):
            hidden_size = self.backbone.config.text_config.hidden_size
        else:
            # Fallback: используем стандартный размер для Qwen2.5-7B
            hidden_size = 3584
            print(f"⚠️ Warning: Could not determine hidden_size from config, using default: {hidden_size}")
        
        self.viral_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        ).to(self.backbone.device) # Убеждаемся, что голова на GPU

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        pixel_values=None, 
        pixel_values_videos=None, 
        image_grid_thw=None, 
        video_grid_thw=None,
        return_full_sequence: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input_ids: Токены текста [Batch, Seq]
            attention_mask: Маска внимания [Batch, Seq]
            pixel_values: Изображения (опционально)
            pixel_values_videos: Видео-кадры (опционально)
            image_grid_thw: Размеры сетки изображений
            video_grid_thw: Размеры сетки видео [Batch, 3] -> (T, H, W)
            return_full_sequence: Если True, возвращает (score, video_hidden_states)
            
        Returns:
            score: Предсказанный скор виральности [Batch, 1]
            video_hidden_states (опционально): Скрытые состояния видео-токенов [Batch, NumVideoTokens, Hidden]
        """
        # Прогон через Qwen2.5-Omni-7B
        # Поддержка как изображений (pixel_values), так и видео (pixel_values_videos)
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True
        }
        
        # Для видео используем pixel_values_videos и video_grid_thw
        if pixel_values_videos is not None:
            forward_kwargs["pixel_values_videos"] = pixel_values_videos
            if video_grid_thw is not None:
                forward_kwargs["video_grid_thw"] = video_grid_thw
        # Для изображений используем pixel_values и image_grid_thw
        elif pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                forward_kwargs["image_grid_thw"] = image_grid_thw
        
        outputs = self.backbone(**forward_kwargs)

        # Извлечение последнего скрытого состояния (H_final)
        last_hidden_state = outputs.hidden_states[-1]

        # Находим индексы последних токенов (исключая паддинг)
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1

        # [Batch, Seq, Hidden] -> [Batch, Hidden]
        h_final = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

        # Кастуем во float32 для стабильности MLP
        h_final = h_final.float()

        # Прогон через регрессионную голову
        score = self.viral_head(h_final)

        if return_full_sequence:
            # Извлекаем только видео-токены
            video_hidden_states = self._extract_video_tokens(
                last_hidden_state=last_hidden_state,
                input_ids=input_ids,
                video_grid_thw=video_grid_thw,
                image_grid_thw=image_grid_thw
            )
            return score, video_hidden_states

        return score
    
    def _extract_video_tokens(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Извлекает hidden states только для видео/изображения токенов.
        
        Qwen2.5-Omni использует специальные токены-маркеры для обозначения
        визуального контента. Видео-токены находятся между <|vision_start|> и <|vision_end|>.
        
        Args:
            last_hidden_state: [Batch, Seq, Hidden]
            input_ids: [Batch, Seq]
            video_grid_thw: [Batch, 3] - (T, H, W) для видео
            image_grid_thw: [Batch, 3] - (T, H, W) для изображений
            
        Returns:
            video_hidden_states: [Batch, NumVideoTokens, Hidden]
        """
        batch_size = last_hidden_state.shape[0]
        hidden_size = last_hidden_state.shape[-1]
        device = last_hidden_state.device
        
        # Определяем количество визуальных токенов из grid_thw
        # grid_thw содержит (T, H, W) - количество временных шагов и пространственные размеры
        if video_grid_thw is not None:
            # Для видео: T * H * W токенов
            # video_grid_thw: [batch, 3] -> каждая строка это (T, H, W)
            num_visual_tokens = (video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]).tolist()
        elif image_grid_thw is not None:
            # Для изображений: 1 * H * W токенов
            num_visual_tokens = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        else:
            # Fallback: используем эвристику на основе специальных токенов
            return self._extract_video_tokens_by_markers(last_hidden_state, input_ids)
        
        # Собираем видео-токены для каждого элемента батча
        # Визуальные токены в Qwen2.5 обычно идут после vision_start токена
        max_tokens = max(num_visual_tokens) if num_visual_tokens else 1
        video_hidden_states = torch.zeros(batch_size, max_tokens, hidden_size, device=device)
        
        for b in range(batch_size):
            n_tokens = int(num_visual_tokens[b])
            # Находим позицию начала визуальных токенов
            # В Qwen2.5-Omni визуальные токены обычно начинаются после системного промпта
            # Используем эвристику: визуальные токены = токены без текстового embedding
            # Более надежный способ - искать специальные маркеры
            
            video_start_idx = self._find_vision_start_position(input_ids[b])
            video_end_idx = video_start_idx + n_tokens
            
            # Извлекаем токены
            extracted = last_hidden_state[b, video_start_idx:video_end_idx, :]
            video_hidden_states[b, :extracted.shape[0], :] = extracted.float()
        
        return video_hidden_states
    
    def _find_vision_start_position(self, input_ids: torch.Tensor) -> int:
        """
        Находит позицию начала визуальных токенов в последовательности.
        
        Qwen2.5-Omni использует специальные токены:
        - <|vision_start|> (ID: 151652) - начало визуального контента
        - <|vision_end|> (ID: 151653) - конец визуального контента
        - <|image_pad|> (ID: 151655) - padding для изображений
        - <|video_pad|> (ID: 151656) - padding для видео
        
        Returns:
            Индекс первого визуального токена
        """
        # Специальные токены Qwen2.5-Omni
        VISION_START_TOKEN = 151652
        IMAGE_PAD_TOKEN = 151655
        VIDEO_PAD_TOKEN = 151656
        
        # Ищем vision_start токен
        vision_start_mask = (input_ids == VISION_START_TOKEN)
        if vision_start_mask.any():
            # Визуальные токены начинаются сразу после <|vision_start|>
            return int(vision_start_mask.nonzero()[0].item()) + 1
        
        # Альтернативный поиск: ищем первый image_pad или video_pad
        pad_mask = (input_ids == IMAGE_PAD_TOKEN) | (input_ids == VIDEO_PAD_TOKEN)
        if pad_mask.any():
            return int(pad_mask.nonzero()[0].item())
        
        # Fallback: предполагаем, что визуальные токены начинаются с позиции 1
        # (после BOS токена)
        return 1
    
    def _extract_video_tokens_by_markers(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Fallback метод: извлекает видео-токены по специальным маркерам.
        Используется когда grid_thw недоступен.
        """
        batch_size = last_hidden_state.shape[0]
        hidden_size = last_hidden_state.shape[-1]
        device = last_hidden_state.device
        
        VISION_START_TOKEN = 151652
        VISION_END_TOKEN = 151653
        
        # Находим границы визуальных токенов для каждого элемента батча
        video_tokens_list = []
        max_len = 0
        
        for b in range(batch_size):
            ids = input_ids[b]
            start_mask = (ids == VISION_START_TOKEN)
            end_mask = (ids == VISION_END_TOKEN)
            
            if start_mask.any() and end_mask.any():
                start_idx = int(start_mask.nonzero()[0].item()) + 1
                end_idx = int(end_mask.nonzero()[0].item())
                tokens = last_hidden_state[b, start_idx:end_idx, :].float()
            else:
                # Fallback: берем среднее всех токенов
                tokens = last_hidden_state[b, 1:-1, :].float()
            
            video_tokens_list.append(tokens)
            max_len = max(max_len, tokens.shape[0])
        
        # Паддинг до одинаковой длины
        video_hidden_states = torch.zeros(batch_size, max_len, hidden_size, device=device)
        for b, tokens in enumerate(video_tokens_list):
            video_hidden_states[b, :tokens.shape[0], :] = tokens
        
        return video_hidden_states
