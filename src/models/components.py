# src/models/components.py
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Стандарт нормализации для LLaMA/Qwen.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Обучаемый параметр масштаба (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Projector(nn.Module):
    """
    MLP-адаптер (projector) для преобразования эмбеддингов CLAP (512) → Qwen (4096).
    Архитектура:
        Linear(512 → hidden_dim)
        GELU
        Linear(hidden_dim → hidden_dim)
    По умолчанию hidden_dim = 4096 (как у Qwen-3).
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 4096):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        # Инициализация весов (как в трансформерных моделях)
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов с использованием xavier_uniform для лучшей совместимости с GELU."""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[float] of shape [Batch, 512]
        Returns:
            Tensor[float] of shape [Batch, 4096] (или hidden_dim)
        """
        return self.projector(x)


class PredictorHead(nn.Module):
    """
    Regression Head ("Судья").
    Принимает эмбеддинги от LLM и выдает одно число (Viral Index).
    """
    def __init__(self, hidden_dim: int = 4096, dropout_rate: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            # 1. Нормализация входа (критично для стабильности)
            RMSNorm(hidden_dim),
            
            # 2. Проекция (Hidden -> 1024)
            nn.Linear(hidden_dim, 1024),
            
            # 3. Активация
            nn.GELU(),
            
            # 4. Регуляризация
            nn.Dropout(dropout_rate),
            
            # 5. Финальное предсказание (1024 -> 1)
            nn.Linear(1024, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Hidden_Dim] -> [Batch, 1]
        return self.net(x)


# Алиасы для обратной совместимости
ViralPredictor = PredictorHead
MLPAdapter = Projector


# ==========================================
# Тесты компонентов
# ==========================================
if __name__ == "__main__":
    print("\n--- Testing Components ---\n")
    
    # Тест 1: RMSNorm
    print("Test 1: RMSNorm")
    try:
        rms_norm = RMSNorm(dim=128)
        x = torch.randn(2, 128)
        out = rms_norm(x)
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"
        print("✅ RMSNorm passed")
    except Exception as e:
        print(f"❌ RMSNorm failed: {e}")
        exit(1)
    
    # Тест 2: Projector
    print("\nTest 2: Projector")
    try:
        projector = Projector(input_dim=512, hidden_dim=4096)
        x = torch.randn(4, 512, requires_grad=True)
        out = projector(x)
        assert out.shape == (4, 4096), f"Expected (4, 4096), got {out.shape}"
        
        # Проверка градиентов (обучаемость)
        loss = out.mean()
        loss.backward()
        first_layer = projector.projector[0]
        assert first_layer.weight.grad is not None, "Градиенты не вычисляются для Projector"
        assert not torch.all(first_layer.weight.grad == 0), "Градиенты равны нулю"
        print("✅ Projector passed (shape + gradients)")
    except Exception as e:
        print(f"❌ Projector failed: {e}")
        exit(1)
    
    # Тест 3: PredictorHead
    print("\nTest 3: PredictorHead")
    try:
        predictor = PredictorHead(hidden_dim=4096, dropout_rate=0.1)
        x = torch.randn(4, 4096, requires_grad=True)
        out = predictor(x)
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"
        
        # Проверка градиентов (обучаемость)
        loss = out.mean()
        loss.backward()
        first_linear = predictor.net[1]  # Первый Linear слой после RMSNorm
        assert first_linear.weight.grad is not None, "Градиенты не вычисляются для PredictorHead"
        assert not torch.all(first_linear.weight.grad == 0), "Градиенты равны нулю"
        print("✅ PredictorHead passed (shape + gradients)")
    except Exception as e:
        print(f"❌ PredictorHead failed: {e}")
        exit(1)
    
    # Тест 4: Алиасы (обратная совместимость)
    print("\nTest 4: Backward compatibility aliases")
    try:
        # Проверяем, что алиасы работают
        mlp_adapter = MLPAdapter(input_dim=512, hidden_dim=4096)
        viral_predictor = ViralPredictor(hidden_dim=4096)
        
        x_proj = torch.randn(2, 512)
        x_pred = torch.randn(2, 4096)
        
        out_proj = mlp_adapter(x_proj)
        out_pred = viral_predictor(x_pred)
        
        assert out_proj.shape == (2, 4096), f"MLPAdapter shape mismatch: {out_proj.shape}"
        assert out_pred.shape == (2, 1), f"ViralPredictor shape mismatch: {out_pred.shape}"
        print("✅ Backward compatibility aliases passed")
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")
        exit(1)
    
    print("\n--- All tests passed! ---\n")