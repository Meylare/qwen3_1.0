import pandas as pd
import numpy as np

# --- 🔥 ИЗМЕНЕНИЕ: Указываем Parquet файл ---
INPUT_FILE = "dataset_v1_raw.parquet"
FIELDS_TO_CHECK = [
    'likes', 
    'comments', 
    'views'
]

def find_negatives_in_parquet():
    print(f"🕵️‍♂️ Ищем отрицательные значения в файле {INPUT_FILE}...")
    
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Файл {INPUT_FILE} не найден!")
        return
        
    print(f"   ✅ Файл успешно загружен ({len(df)} строк).")
    
    anomalies_found = False
    
    for field in FIELDS_TO_CHECK:
        if field in df.columns:
            # Ищем, есть ли ХОТЯ БЫ ОДНО значение < 0
            if (df[field] < 0).any():
                anomalies_found = True
                negative_count = (df[field] < 0).sum()
                print("-" * 50)
                print(f"❌ НАЙДЕНЫ АНОМАЛИИ в поле '{field}'!")
                print(f"   Количество отрицательных значений: {negative_count}")
                # Показываем примеры
                print("   Примеры аномальных строк:")
                print(df[df[field] < 0][['video_id', field]].head())
                print("-" * 50)

    if not anomalies_found:
        print("\n✅ Проверка завершена. Отрицательных значений в ключевых полях НЕ НАЙДЕНО.")

if __name__ == "__main__":
    find_negatives_in_parquet()