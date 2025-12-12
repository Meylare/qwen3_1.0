# tests/test_projector.py
import os
import sys
import unittest
import torch
from src.models.components import MLPAdapter


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestMLPAdapter(unittest.TestCase):
    """
    Тесты для класса MLPAdapter, проверяющие архитектуру,
    геометрию и обучаемость.
    """

    def setUp(self):
        """Настройка параметров перед каждым тестом."""
        self.batch_size = 4
        self.input_dim = 512  # Размерность CLAP
        self.hidden_dim = 4096  # Размерность Qwen
        self.adapter = MLPAdapter(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        # Создаем "фейковые" входные данные
        self.dummy_input = torch.randn(self.batch_size, self.input_dim)

    def test_initialization(self):
        """
        1. Проверка Архитектуры (Initialization)
        Цель: Убедиться, что модель создается без ошибок и имеет правильные слои.
        """
        self.assertIsInstance(self.adapter, MLPAdapter, "Объект не является экземпляром MLPAdapter.")
        
        # Проверяем, что слои созданы в правильном порядке и правильного типа
        projector_layers = list(self.adapter.projector.children())
        self.assertEqual(len(projector_layers), 3, "Проектор должен содержать 3 слоя.")
        self.assertIsInstance(projector_layers[0], torch.nn.Linear, "Первый слой должен быть Linear.")
        self.assertIsInstance(projector_layers[1], torch.nn.GELU, "Второй слой должен быть GELU.")
        self.assertIsInstance(projector_layers[2], torch.nn.Linear, "Третий слой должен быть Linear.")
        print("\n[PASS] 1. Проверка Архитектуры: Модель успешно инициализирована.")

    def test_forward_pass_shape(self):
        """
        2. Проверка Геометрии (Forward Pass)
        Цель: Убедиться, что выходной тензор имеет ожидаемую размерность.
        """
        # Прогоняем данные через модель
        output = self.adapter(self.dummy_input)
        
        # Ожидаемая размерность
        expected_shape = (self.batch_size, self.hidden_dim)
        
        self.assertEqual(output.shape, expected_shape,
                         f"Ошибка геометрии: ожидался размер {expected_shape}, но получен {output.shape}.")
        print(f"[PASS] 2. Проверка Геометрии: Размер выходного тензора корректен: {output.shape}.")

    def test_backward_pass_gradients(self):
        """
        3. Проверка Обучаемости (Backward Pass)
        Цель: Убедиться, что градиенты вычисляются и не являются None.
        """
        # Получаем вывод модели
        output = self.adapter(self.dummy_input)
        
        # Вычисляем "выдуманную" ошибку
        loss = output.mean()
        
        # Выполняем обратное распространение ошибки
        loss.backward()
        
        # Проверяем градиент у весов первого слоя
        first_layer_weights = self.adapter.projector[0].weight
        self.assertIsNotNone(first_layer_weights.grad,
                             "Катастрофа: градиенты не дошли до первого слоя (grad is None).")
        
        # Дополнительная проверка, что градиенты не нулевые
        self.assertNotEqual(torch.sum(first_layer_weights.grad**2), 0,
                            "Градиенты равны нулю, возможно, есть проблема в вычислениях.")
                            
        print("[PASS] 3. Проверка Обучаемости: Градиенты успешно вычислены и прошли по сети.")

if __name__ == '__main__':
    unittest.main()