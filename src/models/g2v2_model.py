# src/models/g2v2_model.py
"""
G2V2 Model - основная модель для предсказания вирусности видео.
Объединяет Qwen3-VL, AudioProjector и ViralPredictor с поддержкой Fusion.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from .components import Projector, PredictorHead, RMSNorm


class G2V2Model(nn.Module):
    """
    Полная модель G2V2 для предсказания вирусности.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        clap_dim: int = 512,
        hidden_dim: int = 4096,
        dropout_rate: float = 0.1,
        audio_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        projector_weights_path: Optional[str] = None,
        # --- АРГУМЕНТ ДЛЯ ТЕСТОВ ---
        _test_config: Optional[Any] = None, 
    ):
        super().__init__()
        
        # 1. Загрузка Qwen модели
        if _test_config is not None:
            # === РЕЖИМ ТЕСТА (TINY) ===
            print("⚠️ TEST MODE: Creating random weights from config...")
            # Используем правильный класс для VL модели
            self.model = Qwen2_5_VLForConditionalGeneration(_test_config)
            
            # Принудительно float32 для стабильности на CPU при тестах
            self.model.to(torch.float32)
            if hasattr(_test_config, "hidden_size"):
                hidden_dim = _test_config.hidden_size
        else:
            # === РЕЖИМ ПРОДАКШЕНА (REAL) ===
            print(f"[INFO] Загрузка модели {model_name}...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Автоматическое определение hidden_dim
            if hasattr(self.model.config, "hidden_size"):
                real_hidden_dim = self.model.config.hidden_size
                if real_hidden_dim != hidden_dim:
                    print(f"[WARN] Config hidden_dim ({real_hidden_dim}) != Arg ({hidden_dim}). Using Config.")
                    hidden_dim = real_hidden_dim
        
        self.hidden_dim = hidden_dim

        # Загрузка токенизатора
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            self.tokenizer = None
            print("⚠️ Токенизатор не загружен (не критично для теста)")
        
        # 2. Добавление спец-токенов
        if self.tokenizer:
            special_tokens = ['<|audio|>', '<|video|>']
            vocab = self.tokenizer.get_vocab()
            new_tokens = [t for t in special_tokens if t not in vocab]
            
            if new_tokens:
                num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
                if num_added > 0:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    print(f"[OK] Добавлено {num_added} спец-токенов: {new_tokens}")
        
        # 3. Заморозка Vision Encoder
        vision_module = None
        if hasattr(self.model, 'visual'):
            vision_module = self.model.visual
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            vision_module = self.model.model.visual
            
        if vision_module:
            for param in vision_module.parameters():
                param.requires_grad = False
            print("[OK] Vision Encoder заморожен")
        
        # 4. Инициализация Компонентов
        self.audio_projector = Projector(input_dim=clap_dim, hidden_dim=hidden_dim)
        self.viral_predictor = PredictorHead(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        
        if projector_weights_path:
            try:
                state_dict = torch.load(projector_weights_path, map_location='cpu')
                self.audio_projector.load_state_dict(state_dict)
                print(f"[OK] Проектор загружен: {projector_weights_path}")
            except Exception as e:
                print(f"[WARN] Ошибка загрузки проектора: {e}")
        
        # 5. ID токенов
        if self.tokenizer:
            self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio|>")
            self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video|>")
        else:
            self.audio_token_id = None
            self.video_token_id = None
        
        # 6. Применение LoRA
        self.use_lora = use_lora
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            print("[OK] LoRA применен к модели")

    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Безопасное получение эмбеддингов текста."""
        # Приоритет 1: Официальный метод (работает через LoRA)
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()(input_ids)
        
        # Приоритет 2: Прямой доступ (с распаковкой PEFT)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens(input_ids)
        if hasattr(base, "embed_tokens"):
            return base.embed_tokens(input_ids)
            
        raise AttributeError("Не удалось найти слой embed_tokens")
    
    def _get_video_embeddings(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Получает видео эмбеддинги (из кэша или через энкодер).
        """
        if video_features is not None:
            return video_features
        
        if pixel_values is not None:
            # Ищем модуль visual
            visual_module = None
            if hasattr(self.model, 'visual'):
                visual_module = self.model.visual
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
                visual_module = self.model.model.visual
            
            if visual_module:
                with torch.no_grad():
                    return visual_module(hidden_states=pixel_values, grid_thw=image_grid_thw)
        
        return None
    
    def _get_audio_embeddings(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Проецирует CLAP векторы в пространство Qwen."""
        if len(audio_features.shape) == 2:
            audio_features = audio_features.unsqueeze(1) # [B, 1, 512]

        batch_size, seq_len, _ = audio_features.shape
        flat = audio_features.view(-1, audio_features.shape[-1])
        projected = self.audio_projector(flat)
        return projected.view(batch_size, seq_len, -1)

    def get_audio_embeddings(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Публичный метод для получения аудио эмбеддингов (используется в Stage 1)."""
        return self._get_audio_embeddings(audio_features)

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Публичный метод для получения текстовых эмбеддингов (используется в Stage 1)."""
        return self._get_text_embeddings(input_ids)

    def _fuse_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        audio_embeddings: Optional[torch.Tensor] = None,
        video_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # Добавляем аргументы:
        audio_attention_mask: Optional[torch.Tensor] = None, 
        video_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        
        parts_embeds = []
        parts_masks = []
        
        # 1. AUDIO
        if audio_embeddings is not None:
            if len(audio_embeddings.shape) == 2: 
                audio_embeddings = audio_embeddings.unsqueeze(1)
            parts_embeds.append(audio_embeddings)
            
            # ИСПОЛЬЗУЕМ ПЕРЕДАННУЮ МАСКУ
            if audio_attention_mask is not None:
                parts_masks.append(audio_attention_mask.to(device=device, dtype=torch.long))
            else:
                parts_masks.append(torch.ones((batch_size, audio_embeddings.shape[1]), dtype=torch.long, device=device))
            
        # 2. VIDEO
        if video_embeddings is not None:
            parts_embeds.append(video_embeddings)
            
            # ИСПОЛЬЗУЕМ ПЕРЕДАННУЮ МАСКУ
            if video_attention_mask is not None:
                parts_masks.append(video_attention_mask.to(device=device, dtype=torch.long))
            else:
                parts_masks.append(torch.ones((batch_size, video_embeddings.shape[1]), dtype=torch.long, device=device))
            
        # 3. TEXT
        parts_embeds.append(inputs_embeds)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, inputs_embeds.shape[1]), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
        parts_masks.append(attention_mask)
        
        # Склейка
        fused_embeds = torch.cat(parts_embeds, dim=1)
        fused_mask = torch.cat(parts_masks, dim=1)
        
        return fused_embeds, fused_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        # Добавляем маски в аргументы
        audio_attention_mask: Optional[torch.Tensor] = None,
        video_attention_mask: Optional[torch.Tensor] = None,
        task_type: str = "score",
        **kwargs
    ) -> Union[torch.Tensor, Any]:
        """
        Forward pass.
        """
        # 1. Текст
        text_embeds = self._get_text_embeddings(input_ids)
        
        # 2. Видео
        video_embeds = self._get_video_embeddings(pixel_values, video_features, image_grid_thw)
        
        # 3. Аудио
        audio_embeds = None
        if audio_features is not None:
            audio_embeds = self._get_audio_embeddings(audio_features)
            
        # 4. Fusion (Передаем маски)
        inputs_embeds, final_mask = self._fuse_embeddings(
            inputs_embeds=text_embeds,
            audio_embeddings=audio_embeds,
            video_embeddings=video_embeds,
            attention_mask=attention_mask,
            audio_attention_mask=audio_attention_mask,
            video_attention_mask=video_attention_mask
        )
        
        # 5. Backbone Pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=final_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # 6. Pooling (Последний токен)
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        else:
            last_hidden = outputs[0]
        
        # --- FIX: ROBUST LAST TOKEN POOLING ---
        # Вместо суммирования маски (что ломается при паддинге внутри),
        # используем argmax по индексам, умноженным на маску.
        
        # Создаем тензор индексов [0, 1, ..., seq_len-1]
        indices = torch.arange(last_hidden.shape[1], device=last_hidden.device)
        
        # Умножаем индексы на маску. Там где padding (0), будет 0.
        # Broadcasting: [SeqLen] * [Batch, SeqLen] -> [Batch, SeqLen]
        masked_indices = indices * final_mask
        
        # Находим индекс максимального элемента в каждой строке.
        # Так как индексы монотонно возрастают, максимум будет у последнего ненулевого элемента маски.
        last_token_indices = masked_indices.argmax(dim=1)
        
        batch_indices = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        pooled_output = last_hidden[batch_indices, last_token_indices]
        # -------------------------------------
        
        # 7. Выход
        if task_type == "score":
            return self.viral_predictor(pooled_output)
        elif task_type == "text":
            if hasattr(self.model, "lm_head"):
                return self.model.lm_head(pooled_output)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'lm_head'):
                return self.model.model.lm_head(pooled_output)
            else:
                # Fallback для tiny тестов
                if hasattr(self.model, 'get_input_embeddings'):
                     return self.model.get_input_embeddings().weight.matmul(pooled_output.T).T
                else:
                     raise AttributeError("No lm_head found")
        else:
            raise ValueError("Unknown task_type")