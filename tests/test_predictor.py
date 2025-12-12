import torch
import sys
import os

# Добавляем корневую папку в путь, чтобы питон видел src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.baseline.models.predictor import ViralPredictor

def test_viral_predictor():
    print("🚀 Запуск теста ViralPredictor (The Judge)...")

    # 1. Настройка окружения
    BATCH_SIZE = 8
    HIDDEN_DIM = 4096 # Размерность Qwen 3 (или другой большой LLM)
    model = ViralPredictor(hidden_dim=HIDDEN_DIM)
    
    print(f"   ⚙️ Config: Batch={BATCH_SIZE}, Hidden={HIDDEN_DIM}")

    # 2. Генерация входных данных (Mock Input)
    # Имитируем "last_hidden_state" от LLM
    input_tensor = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    print("   ✅ Input tensor created.")

    # 3. Проверка Forward Pass (Геометрия)
    try:
        output = model(input_tensor)
        print(f"   📊 Output shape: {output.shape}")
        
        expected_shape = (BATCH_SIZE, 1)
        assert output.shape == expected_shape, f"❌ Ошибка размерности! Ожидалось {expected_shape}, получили {output.shape}"
        print("   ✅ Forward pass shape check passed.")
    except Exception as e:
        print(f"   ❌ Forward pass crashed: {e}")
        return

    # 4. Проверка значений (Sanity Check)
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("   ❌ Ошибка: Выход содержит NaN или Inf!")
        return
    else:
        print("   ✅ Sanity check passed (No NaNs/Infs).")

    # 5. Проверка Backward Pass (Обучаемость)
    try:
        loss = output.mean() # Фиктивный лосс
        loss.backward()      # Обратное распространение ошибки
        
        # Проверяем градиенты на первом Linear слое
        # В нашей структуре: [0]=RMSNorm, [1]=Linear
        first_linear_layer = model.net[1]
        grads = first_linear_layer.weight.grad
        
        if grads is None:
            print("   ❌ Ошибка: Градиенты None (цепь разорвана).")
        elif torch.all(grads == 0):
            print("   ❌ Ошибка: Градиенты равны нулю (мертвые нейроны).")
        else:
            grad_norm = grads.norm().item()
            print(f"   ✅ Backward pass passed. Grad norm: {grad_norm:.4f}")
            
    except Exception as e:
        print(f"   ❌ Backward pass crashed: {e}")
        return

    print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! 'Голова' работает корректно.")

if __name__ == "__main__":
    test_viral_predictor()