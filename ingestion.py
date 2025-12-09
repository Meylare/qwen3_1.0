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
    data, errors = [], 0
    print(f"📖 Читаем {file_path} в безопасном режиме...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Чтение {os.path.basename(file_path)}"):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                errors += 1
    if errors > 0: print(f"⚠️ Пропущено битых строк: {errors}")
    return pd.DataFrame(data)

def run_etl():
    print("🚀 Начинаем ETL процесс (ФИНАЛЬНАЯ ВЕРСИЯ)...")

    # 1. ЗАГРУЗКА
    df_profiles = load_jsonl_robust(PROFILES_PATH)
    df_videos = load_jsonl_robust(VIDEOS_PATH)

    # 2. ПЕРЕИМЕНОВАНИЕ И ОЧИСТКА
    df_profiles = df_profiles.rename(columns={'username': 'account_id', 'followersCount': 'raw_followers', 'followsCount': 'raw_following', 'postsCount': 'profile_current_posts', 'biography': 'raw_bio', 'verified': 'st_is_verified', 'isBusinessAccount': 'st_is_business'})
    cols_to_keep_prof = ['account_id', 'raw_followers', 'raw_following', 'profile_current_posts', 'raw_bio', 'st_is_verified', 'st_is_business']
    cols_exist = [c for c in cols_to_keep_prof if c in df_profiles.columns]
    df_profiles = df_profiles[cols_exist].drop_duplicates(subset=['account_id'])
    
    # Используем shortCode
    df_videos = df_videos.rename(columns={'shortCode': 'video_id', 'ownerUsername': 'account_id_join', 'caption': 'video_text', 'timestamp': 'upload_date', 'likesCount': 'likes', 'commentsCount': 'comments'})
    
    if 'videoPlayCount' in df_videos.columns: df_videos['views'] = df_videos['videoPlayCount'].fillna(0)
    elif 'videoViewCount' in df_videos.columns: df_videos['views'] = df_videos['videoViewCount'].fillna(0)
    else: df_videos['views'] = 0
    
    df_videos['upload_date'] = pd.to_datetime(df_videos['upload_date'], errors='coerce')
    df_videos = df_videos.dropna(subset=['upload_date', 'video_id'])

    # --- 🔥 САНИТАЙЗЕР ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ 🔥 ---
    print("🧹 Санитарная обработка: убираем отрицательные значения...")
    cols_to_sanitize = ['views', 'likes', 'comments']
    for col in cols_to_sanitize:
        if col in df_videos.columns:
            # clip(lower=0) заменяет все отрицательные числа на 0
            df_videos[col] = df_videos[col].clip(lower=0)

    # 3. MERGE И ФИНАЛИЗАЦИЯ
    print("🔗 Объединяем...")
    full_df = df_videos.merge(df_profiles, left_on='account_id_join', right_on='account_id', how='inner')
    full_df = full_df.sort_values(by=['account_id', 'upload_date'], ascending=[True, True])
    full_df = full_df.drop_duplicates(subset=['video_id'], keep='first')
    
    final_columns = ['account_id', 'video_id', 'upload_date', 'video_text', 'views', 'likes', 'comments', 'raw_followers', 'raw_following', 'profile_current_posts', 'st_is_verified', 'st_is_business', 'raw_bio']
    cols_to_save = [c for c in final_columns if c in full_df.columns]
    full_df = full_df[cols_to_save]
    
    # Приведение типов
    if 'st_is_verified' in full_df.columns: full_df['st_is_verified'] = full_df['st_is_verified'].fillna(False).astype(int)
    if 'st_is_business' in full_df.columns: full_df['st_is_business'] = full_df['st_is_business'].fillna(False).astype(int)
    full_df['video_id'] = full_df['video_id'].astype(str)
    full_df['account_id'] = full_df['account_id'].astype(str)
    
    # 4. СОХРАНЕНИЕ
    print(f"💾 Сохраняем в {OUTPUT_FILE}...")
    full_df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n✅ ETL ЗАВЕРШЕН! (ФИНАЛЬНАЯ ВЕРСИЯ)")

if __name__ == "__main__":
    run_etl()