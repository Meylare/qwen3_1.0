import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType


class ViralPredictorModel(nn.Module):
    def __init__(self, base_model_id, bnb_config):
        super().__init__()

        # 1. Загрузка Backbone (Qwen)
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        hidden_size = self.backbone.config.hidden_size
        self.viral_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        ).to(self.backbone.device) # Убеждаемся, что голова на GPU

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw=None):
        # Прогон через Qwen
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw, # Важно передать сюда!
            output_hidden_states=True,
            return_dict=True
        )

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

        return score
