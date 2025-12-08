import pandas as pd
import numpy as np
from tqdm import tqdm
# Импортируем библиотеку друга
from features_lib import extract_static

# --- КОНФИГУРАЦИЯ ---
INPUT_HISTORY = "full_history_features.parquet"
OUTPUT_COMPLETE = "dataset_v0_complete.parquet"

def run_merge():
    print("🚀 Задача: Сборка единого Датасета (Merge via Adapter)...")

    # 1. Загрузка
    try:
        df = pd.read_parquet(INPUT_HISTORY)
        print(f"📥 Читаем {INPUT_HISTORY} ({len(df)} строк)...")
    except FileNotFoundError:
        print("❌ Файл не найден! Сначала выполни прошлую задачу.")
        return

    # Сохраняем твой Time Travel (правильные посты в прошлом), 
    # чтобы скрипт друга их случайно не перезаписал текущими данными.
    if 'st_hist_total_posts' in df.columns:
        my_correct_time_travel = df['st_hist_total_posts'].copy()
    else:
        my_correct_time_travel = None

    # 2. АДАПТЕР: Применяем функцию друга ко всем строкам
    # Функция друга ждет словарь с ключами: followers_count, following_count и т.д.
    # А у нас: raw_followers, raw_following...
    
    static_results = []
    
    print("🛠 Применяем extract_static ко всем строкам...")
    # Используем tqdm для прогресс бара
    for _, row in tqdm(df.iterrows(), total=len(df)):
        
        # 2.1 Подготовка словаря (Маппинг твоих колонок в ключи друга)
        profile_dict = {
            'followers_count': row.get('raw_followers', 0),
            'following_count': row.get('raw_following', 0),
            'media_count': row.get('profile_current_posts', 0), # Нужно для STRICT_MODE
            'is_verified': row.get('st_is_verified', False),
            'is_business_account': row.get('st_is_business', False),
            'biography': row.get('raw_bio', '')
        }
        
        # 2.2 Вызов функции друга
        try:
            # Получаем словарь {'st_log_followers': ..., 'st_ratio_ff': ...}
            feats = extract_static(profile_dict)
            static_results.append(feats)
        except Exception as e:
            # Если что-то упало, заполняем нулями, чтобы не крашить весь процесс
            print(f"⚠️ Ошибка в extract_static: {e}")
            static_results.append({})

    # 3. Создаем DataFrame из результатов статики
    df_static = pd.DataFrame(static_results)
    
    # 4. Объединяем твою историю с полученной статикой
    # Сначала удалим статику из основного df, если она там есть (чтобы не дублировать)
    cols_to_drop = [c for c in df_static.columns if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    # Склеиваем по индексу
    df_full = pd.concat([df.reset_index(drop=True), df_static.reset_index(drop=True)], axis=1)

    # 5. ВОССТАНОВЛЕНИЕ СПРАВЕДЛИВОСТИ (Time Travel)
    # Скрипт друга вернул st_hist_total_posts как "media_count" (текущее).
    # Мы возвращаем твой расчет "Time Travel".
    if my_correct_time_travel is not None:
        df_full['st_hist_total_posts'] = my_correct_time_travel.reset_index(drop=True)
        print("✅ Time Travel (исторические посты) сохранен (приоритет над статикой).")

    # 6. Добавляем Таргет
    if 'views' in df_full.columns:
        df_full['target_views_log'] = np.log1p(df_full['views'])

    # 7. Чистка мусора (raw_ колонки больше не нужны по контракту, но можно оставить для дебага)
    # Оставим пока всё.

    # 8. Сохранение
    print(f"💾 Сохраняем полный датасет: {OUTPUT_COMPLETE}")
    df_full.to_parquet(OUTPUT_COMPLETE, index=False)
    
    print("\n✅ MERGE ЗАВЕРШЕН!")
    print("Пример полученных данных:")
    print(df_full[['account_id', 'st_log_followers', 'st_ratio_ff', 'st_hist_total_posts']].head())

if __name__ == "__main__":
    run_merge()