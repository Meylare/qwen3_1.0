"""
G2V2 Model - основная модель для предсказания вирусности видео.
Объединяет Qwen3-VL, AudioProjector и ViralPredictor с поддержкой Fusion.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

from .components import Projector, PredictorHead, RMSNorm


class G2V2Model(nn.Module):
    """
    Полная модель G2V2 для предсказания вирусности.
    
    Архитектура:
        1. Qwen3-VL-8B-Instruct (замороженный Vision Encoder)
        2. AudioProjector: CLAP embeddings (512) -> Qwen embeddings (4096)
        3. ViralPredictor: Qwen embeddings (4096) -> Viral Index (1)
        4. Fusion: объединение Text, Video и Audio эмбеддингов
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct", # Или Qwen/Qwen2.5-VL-7B-Instruct
        clap_dim: int = 512,
        hidden_dim: int = 4096, # Внимание: для 7B это может быть 3584, для 8B/72B - 4096. Лучше брать из config.
        dropout_rate: float = 0.1,
        audio_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        use_lora: bool = True,
        # --- ОБНОВЛЕННЫЕ ПАРАМЕТРЫ LORA ---
        lora_r: int = 64,          # Увеличили емкость для понимания видео
        lora_alpha: int = 128,     # alpha = 2 * rank
        lora_dropout: float = 0.05, # Чуть меньше дропаут для стабильности
        projector_weights_path: Optional[str] = None,
    ):
        super().__init__()
        
        # 1. Загрузка Qwen модели
        print(f"🏗️ Загрузка модели {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Автоматическое определение hidden_dim из конфига модели (безопаснее, чем хардкод)
        if hasattr(self.model.config, "hidden_size"):
            real_hidden_dim = self.model.config.hidden_size
            if real_hidden_dim != hidden_dim:
                print(f"⚠️ Внимание: Конфиг модели ({real_hidden_dim}) отличается от аргумента ({hidden_dim}). Используем {real_hidden_dim}.")
                hidden_dim = real_hidden_dim
        
        self.hidden_dim = hidden_dim

        # Загрузка токенизатора
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            self.tokenizer = None
            print("⚠️ Токенизатор не загружен (не критично, если токены подаются извне)")
        
        # 2. Добавление спец-токенов
        # Это критично, чтобы модель не путала медиа с концом текста
        if self.tokenizer:
            special_tokens = ['<|audio|>', '<|video|>']
            # Проверяем, нет ли их уже
            new_tokens = [t for t in special_tokens if t not in self.tokenizer.get_vocab()]
            
            if new_tokens:
                num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
                if num_added > 0:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    print(f"✅ Добавлено {num_added} спец-токенов: {new_tokens}")
        
        # 3. Заморозка Vision Encoder
        # Универсальный способ найти визуальный модуль
        vision_module = None
        if hasattr(self.model, 'visual'):
            vision_module = self.model.visual
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            vision_module = self.model.model.visual
            
        if vision_module:
            for param in vision_module.parameters():
                param.requires_grad = False
            print("✅ Vision Encoder найден и заморожен")
        
        # 4. Инициализация Компонентов
        self.audio_projector = Projector(input_dim=clap_dim, hidden_dim=hidden_dim)
        self.viral_predictor = PredictorHead(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        
        # Загрузка весов проектора (Stage 1)
        if projector_weights_path:
            print(f"📥 Загрузка проектора из {projector_weights_path}...")
            state_dict = torch.load(projector_weights_path, map_location='cpu')
            self.audio_projector.load_state_dict(state_dict)
        
        # 5. ID токенов (для замены, если понадобится, хотя мы используем Prepend)
        if self.tokenizer:
            self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio|>")
            self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video|>")
        
        # 6. Применение LoRA
        if use_lora:
            print(f"💉 Инъекция LoRA (Rank={lora_r}, Alpha={lora_alpha})...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Безопасное получение эмбеддингов текста."""
        return self.model.model.embed_tokens(input_ids)
    
    def _get_audio_embeddings(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Проецирует CLAP векторы в пространство Qwen."""
        # [B, 512] -> [B, 1, 512]
        if len(audio_features.shape) == 2:
            audio_features = audio_features.unsqueeze(1)
            
        batch_size, seq_len, _ = audio_features.shape
        # Flatten для прохода через Linear
        flat = audio_features.view(-1, audio_features.shape[-1])
        projected = self.audio_projector(flat)
        # Unflatten обратно
        return projected.view(batch_size, seq_len, -1)

    def _fuse_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        audio_embeddings: Optional[torch.Tensor] = None,
        video_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Стратегия слияния: [AUDIO] -> [VIDEO] -> [TEXT].
        Это оптимально для Causal LM: сначала факты, потом выводы.
        """
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        
        parts_embeds = []
        parts_masks = []
        
        # 1. AUDIO
        if audio_embeddings is not None:
            if len(audio_embeddings.shape) == 2:
                audio_embeddings = audio_embeddings.unsqueeze(1)
            parts_embeds.append(audio_embeddings)
            # Маска для аудио (все 1)
            parts_masks.append(torch.ones((batch_size, audio_embeddings.shape[1]), dtype=torch.long, device=device))
            
        # 2. VIDEO
        if video_embeddings is not None:
            parts_embeds.append(video_embeddings)
            # Маска для видео
            parts_masks.append(torch.ones((batch_size, video_embeddings.shape[1]), dtype=torch.long, device=device))
            
        # 3. TEXT
        parts_embeds.append(inputs_embeds)
        if attention_mask is None:
            # Дефолтная маска, если не передана
            attention_mask = torch.ones((batch_size, inputs_embeds.shape[1]), dtype=torch.long, device=device)
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
        video_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        task_type: str = "score",
        **kwargs
    ) -> Union[torch.Tensor, Any]:
        """
        Forward pass.
        Поддерживает режимы:
        - Кэшированные векторы (video_features)
        - Сырые пиксели (pixel_values) - прогонит через Vision Encoder
        """
        # 1. Текст
        text_embeds = self._get_text_embeddings(input_ids)
        
        # 2. Видео (Кэш или Encoder)
        video_embeds = None
        if video_features is not None:
            video_embeds = video_features # Уже [B, N, D]
        elif pixel_values is not None:
            # Если подали сырые пиксели, используем Vision Encoder
            # (Осторожно, это ест память!)
            with torch.no_grad(): # Обычно мы не учим Vision Encoder
                video_embeds = self.model.visual(pixel_values)
        
        # 3. Аудио
        audio_embeds = None
        if audio_features is not None:
            audio_embeds = self._get_audio_embeddings(audio_features)
            
        # 4. Fusion
        inputs_embeds, final_mask = self._fuse_embeddings(
            inputs_embeds=text_embeds,
            audio_embeddings=audio_embeds,
            video_embeddings=video_embeds,
            attention_mask=attention_mask
        )
        
        # 5. Backbone Pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=final_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # 6. Pooling (Последний токен)
        last_hidden = outputs.hidden_states[-1]
        
        # Находим индекс последнего реального токена (не паддинга)
        # final_mask имеет форму [B, Seq_Len]. Сумма по строке - 1 дает индекс последнего токена.
        seq_lengths = final_mask.sum(dim=1) - 1
        pooled_output = last_hidden[torch.arange(last_hidden.shape[0]), seq_lengths]
        
        # 7. Выход в зависимости от задачи
        if task_type == "score":
            return self.viral_predictor(pooled_output)
        elif task_type == "text":
            # Используем LM Head для генерации текста
            # В HuggingFace моделях это обычно model.lm_head
            if hasattr(self.model, "lm_head"):
                return self.model.lm_head(pooled_output)
            else:
                return self.model.embed_tokens.weight.matmul(pooled_output.T).T # Fallback (редко нужен)
        else:
            raise ValueError("Unknown task_type")