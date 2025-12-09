import pandas as pd
import numpy as np
from tqdm import tqdm
# Импорт по твоему пути
from statistic.features_lib import extract_static

# --- КОНФИГУРАЦИЯ ---
INPUT_HISTORY = "full_history_features.parquet"
OUTPUT_COMPLETE = "dataset_v0_complete.parquet"

def run_merge():
    print("🚀 Задача: Сборка единого Датасета (Merge)...")

    # 1. Загрузка
    try:
        df = pd.read_parquet(INPUT_HISTORY)
        print(f"📥 Читаем {INPUT_HISTORY} ({len(df)} строк)...")
    except FileNotFoundError:
        print("❌ Файл не найден!")
        return

    # Сохраняем Time Travel
    if 'st_hist_total_posts' in df.columns:
        my_correct_time_travel = df['st_hist_total_posts'].copy()
    else:
        my_correct_time_travel = None

    # --- 🔥 ВАЖНО: УДАЛЕНИЕ БИТЫХ СТРОК 🔥 ---
    # В твоем коде этого не было. Это нужно, чтобы убрать 225 битых профилей.
    print("🧹 Очистка битых данных (Drop NaN)...")
    initial_len = len(df)
    
    # Удаляем строки, где нет подписчиков
    df = df.dropna(subset=['raw_followers'])
    
    dropped_count = initial_len - len(df)
    if dropped_count > 0:
        print(f"   ✂️ Удалено {dropped_count} строк с битыми профилями.")
    
    # Сбрасываем индекс
    df = df.reset_index(drop=True)

    # 2. АДАПТЕР (extract_static)
    static_results = []
    print("🛠 Применяем extract_static...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        profile_dict = {
            'followers_count': row.get('raw_followers', 0),
            'following_count': row.get('raw_following', 0),
            'media_count': row.get('profile_current_posts', 0),
            'is_verified': row.get('st_is_verified', False),
            'is_business_account': row.get('st_is_business', False),
            'biography': str(row.get('raw_bio', ''))
        }
        
        try:
            feats = extract_static(profile_dict)
            static_results.append(feats)
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            static_results.append({})

    # 3. Объединение
    df_static = pd.DataFrame(static_results)
    
    # Удаляем конфликтную колонку времени из статики
    if 'st_hist_total_posts' in df_static.columns:
        df_static = df_static.drop(columns=['st_hist_total_posts'])
    
    # Удаляем дубликаты колонок из основного df перед склейкой
    cols_to_drop = [c for c in df_static.columns if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    df_full = pd.concat([df, df_static], axis=1)

    # 4. Проверка таргетов
    # Если ты их уже посчитал в feature_engineering, они здесь есть.
    # Если нет — считаем только views как основной.
    if 'target_views_log' not in df_full.columns and 'views' in df_full.columns:
        print("🎯 Генерация target_views_log...")
        df_full['target_views_log'] = np.log1p(df_full['views'])

    # 5. Сохранение
    print(f"💾 Сохраняем полный датасет: {OUTPUT_COMPLETE}")
    df_full.to_parquet(OUTPUT_COMPLETE, index=False)
    
    print(f"\n✅ MERGE ЗАВЕРШЕН! Итого строк: {len(df_full)}")

if __name__ == "__main__":
    run_merge()