import pandas as pd
import numpy as np

FILE_NAME = "dataset_v0_complete.parquet"

def audit_real_zeros():
    print(f"🕵️‍♂️ Анализ природы нулей в {FILE_NAME}...\n")
    
    try:
        df = pd.read_parquet(FILE_NAME)
    except FileNotFoundError:
        print("❌ Файл не найден.")
        return

    # --- ОПРЕДЕЛЕНИЕ ЗАГЛУШЕК ---
    # В нашей логике заглушки стоят ТОЛЬКО там, где истории мало (< 2 видео)
    # У "старичков" мы такие строки удалили, так что это только "новички".
    mask_artificial = df['hist_window_size'] < 2
    
    print(f"Всего строк: {len(df)}")
    print(f"Из них строк с заглушками (Новички, видео 1-2): {mask_artificial.sum()}")
    print("-" * 60)

    # Колонки для проверки
    cols_to_check = ['hist_median_likes', 'hist_median_views']

    for col in cols_to_check:
        # 1. Всего нулей
        total_zeros = (df[col] == 0).sum()
        
        # 2. Искусственные нули (Заглушки)
        # Это нули в строках, где mask_artificial == True
        artificial_zeros = ((df[col] == 0) & mask_artificial).sum()
        
        # 3. Реальные нули (Честная история)
        # Это нули там, где история уже есть (mask_artificial == False)
        real_zeros = ((df[col] == 0) & (~mask_artificial)).sum()

        print(f"📊 Колонка '{col}':")
        print(f"   Total Zeros:      {total_zeros}")
        print(f"   ├── 🤖 Искусственные (Заглушки): {artificial_zeros}")
        print(f"   └── 👤 Реальные (Честный 0):     {real_zeros}")
        
        if col == 'hist_median_likes':
            if real_zeros > 0:
                print(f"      -> У {real_zeros} видео в истории реально медиана лайков = 0.")
            else:
                print("      -> Ого! У всех, у кого есть история, есть хоть какие-то лайки.")
                
        if col == 'hist_median_views':
            # Тут 0 быть не должно вообще, так как заглушка = 300 (log ~5.7)
            if artificial_zeros > 0:
                print("      ❌ ОШИБКА! Заглушка для просмотров должна быть 300, а не 0!")
            if real_zeros > 0:
                print("      ⚠️ Внимание: Есть видео с РЕАЛЬНЫМИ 0 просмотров в истории.")

        print("-" * 60)

if __name__ == "__main__":
    audit_real_zeros()