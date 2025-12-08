import pandas as pd
import json

FILE_NAME = "owners.jsonl"  # Убедись, что имя файла совпадает с твоим!

print(f"🧐 Проверяем файл: {FILE_NAME}")

# 1. Проверяем первые символы
try:
    with open(FILE_NAME, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        first_line = f.readline().strip()
        
    print(f"Первый символ файла: '{first_char}'")
    if first_char == '[':
        print("⚠️ Файл начинается с '['. Это ОБЫЧНЫЙ JSON, а не JSONL.")
        print("💡 Решение: В ingestion.py убери параметр lines=True")
    elif first_char == '{':
        print("✅ Файл начинается с '{'. Похоже на JSONL.")
    else:
        print(f"❓ Странное начало файла. Возможно, мусор или BOM.")

    # 2. Пробуем прочитать первую строку как JSON
    try:
        json.loads(first_line)
        print("✅ Первая строка валидна.")
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка в первой строке: {e}")
        print(f"Текст строки: {first_line[:100]}...")

    # 3. Смотрим точную ошибку Pandas
    print("\nПопытка чтения через Pandas...")
    pd.read_json(FILE_NAME, lines=True)
    print("🎉 Pandas прочитал файл без ошибок!")

except FileNotFoundError:
    print(f"❌ Файл {FILE_NAME} не найден!")
except ValueError as e:
    print(f"\n❌ PANDAS ERROR: {e}")
except Exception as e:
    print(f"\n❌ UNKNOWN ERROR: {e}")