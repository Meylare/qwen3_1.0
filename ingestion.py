import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
PROFILES_PATH = "owners.jsonl"
VIDEOS_PATH = "100k.jsonl"
OUTPUT_FILE = "dataset_v1_raw.parquet"

def load_jsonl_robust(file_path):
    """
    Безопасная загрузка JSONL.
    """
    data = []
    errors = 0
    print(f"📖 Читаем {file_path} в безопасном режиме...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f)):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                errors += 1
    
    if errors > 0:
        print(f"⚠️ Пропущено битых строк: {errors}")
    return pd.DataFrame(data)

def run_etl():
    print("🚀 Начинаем ETL процесс (FIX: video_id = shortCode)...")

    # 1. ПРОФИЛИ
    if not os.path.exists(PROFILES_PATH):
        print(f"❌ Файл не найден: {PROFILES_PATH}")
        return

    df_profiles = load_jsonl_robust(PROFILES_PATH)
    
    df_profiles = df_profiles.rename(columns={
        'username': 'account_id',
        'followersCount': 'raw_followers',
        'followsCount': 'raw_following',
        'postsCount': 'profile_current_posts',
        'biography': 'raw_bio',
        'verified': 'st_is_verified',
        'isBusinessAccount': 'st_is_business'
    })
    
    cols_to_keep_prof = ['account_id', 'raw_followers', 'raw_following', 
                         'profile_current_posts', 'raw_bio', 'st_is_verified', 'st_is_business']
    cols_exist = [c for c in cols_to_keep_prof if c in df_profiles.columns]
    df_profiles = df_profiles[cols_exist].drop_duplicates(subset=['account_id'])
    
    print(f"✅ Профилей: {len(df_profiles)}")

    # 2. ВИДЕО
    if not os.path.exists(VIDEOS_PATH):
        print(f"❌ Файл не найден: {VIDEOS_PATH}")
        return

    df_videos = load_jsonl_robust(VIDEOS_PATH)

    # Метрика просмотров
    if 'videoPlayCount' in df_videos.columns:
        df_videos['views'] = df_videos['videoPlayCount'].fillna(0)
    elif 'videoViewCount' in df_videos.columns:
        df_videos['views'] = df_videos['videoViewCount'].fillna(0)
    else:
        df_videos['views'] = 0

    # --- 🔥 ГЛАВНОЕ ИЗМЕНЕНИЕ ЗДЕСЬ 🔥 ---
    # Мы берем shortCode как основной идентификатор видео
    df_videos = df_videos.rename(columns={
        'shortCode': 'video_id',           # <--- ТЕПЕРЬ БЕРЕМ shortCode
        'ownerUsername': 'account_id_join',
        'caption': 'video_text',
        'timestamp': 'upload_date',
        'likesCount': 'likes',
        'commentsCount': 'comments'
    })

    df_videos['upload_date'] = pd.to_datetime(df_videos['upload_date'], errors='coerce')
    df_videos = df_videos.dropna(subset=['upload_date'])
    
    # Если shortCode вдруг пустой, удаляем такие строки, иначе потом файлы не найдем
    if 'video_id' in df_videos.columns:
        df_videos = df_videos.dropna(subset=['video_id'])

    print(f"✅ Видео (валидных): {len(df_videos)}")

    # 3. MERGE
    print("🔗 Объединяем...")
    full_df = df_videos.merge(
        df_profiles, 
        left_on='account_id_join', 
        right_on='account_id', 
        how='inner'
    )

    # 4. СОРТИРОВКА
    full_df = full_df.sort_values(by=['account_id', 'upload_date'], ascending=[True, True])
    full_df = full_df.drop_duplicates(subset=['video_id'], keep='first')

    # Финальные колонки
    final_columns = [
        'account_id', 'video_id', 'upload_date', 'video_text',
        'views', 'likes', 'comments',
        'raw_followers', 'raw_following', 'profile_current_posts', 
        'st_is_verified', 'st_is_business', 'raw_bio'
    ]
    cols_to_save = [c for c in final_columns if c in full_df.columns]
    full_df = full_df[cols_to_save]

    # Типы
    pd.set_option('future.no_silent_downcasting', True)
    if 'st_is_verified' in full_df.columns:
        full_df['st_is_verified'] = full_df['st_is_verified'].fillna(False).astype(int)
    if 'st_is_business' in full_df.columns:
        full_df['st_is_business'] = full_df['st_is_business'].fillna(False).astype(int)
    full_df['views'] = full_df['views'].astype(float)

    # Строки для Parquet
    full_df['video_id'] = full_df['video_id'].astype(str)
    full_df['account_id'] = full_df['account_id'].astype(str)

    # 5. СОХРАНЕНИЕ
    print(f"💾 Сохраняем в {OUTPUT_FILE}...")
    full_df.to_parquet(OUTPUT_FILE, index=False)
    
    print("\n✅ ETL ЗАВЕРШЕН! (ID теперь ShortCode)")
    print(f"Итого строк: {len(full_df)}")

if __name__ == "__main__":
    run_etl()