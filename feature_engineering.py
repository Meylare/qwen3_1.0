import pandas as pd
import numpy as np
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = "dataset_v1_raw.parquet"
OUTPUT_CSV = "dataset_v1_labeled.csv"          # Для Sprint Goal
OUTPUT_PARQUET = "full_history_features.parquet" # Для Task DoD

MIN_WINDOW = 3
MAX_WINDOW = 12

# Значения для импутации (заглушки)
DEFAULT_VIEWS = 300.0
DEFAULT_TREND = 1.0
DEFAULT_ER = 0.0
DEFAULT_DAYS = -1.0

def run_feature_engineering():
    print("🚀 Начинаем Feature Engineering (Parquet + CSV)...")

    # 1. Загрузка
    print(f"📥 Читаем {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    df = df.sort_values(by=['account_id', 'upload_date'], ascending=[True, True])

    # ==========================================
    # 🧩 ЧАСТЬ 1: STATIC & TIME TRAVEL
    # ==========================================
    print("⏳ Расчет Time Travel...")
    video_counts = df.groupby('account_id')['video_id'].transform('count')
    video_rank = df.groupby('account_id').cumcount()
    videos_after = video_counts - 1 - video_rank
    df['st_hist_total_posts'] = (df['profile_current_posts'] - videos_after).clip(lower=0)

    # Static
    df['st_log_followers'] = np.log1p(df['raw_followers'])
    df['st_ratio_ff'] = df['raw_followers'] / (df['raw_following'] + 1)
    df['st_bio_length'] = df['raw_bio'].astype(str).apply(len)
    
    # ==========================================
    # 🎯 ЧАСТЬ 2: TARGET
    # ==========================================
    df['target_views_log'] = np.log1p(df['views'])

    # ==========================================
    # 🪟 ЧАСТЬ 3: SLIDING WINDOW
    # ==========================================
    print("🪟 Запуск Rolling Window...")
    grouped = df.groupby('account_id')
    
    lag_views = grouped['views'].shift(1)
    lag_likes = grouped['likes'].shift(1)
    lag_comments = grouped['comments'].shift(1)
    lag_date = grouped['upload_date'].shift(1)
    
    df['hist_days_since_last'] = (df['upload_date'] - lag_date).dt.total_seconds() / 86400
    
    roller_12 = lag_views.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW)
    df['hist_median_views'] = roller_12.median()
    df['hist_mean_views'] = roller_12.mean()
    df['hist_std_views'] = roller_12.std()
    
    roller_3 = lag_views.rolling(window=3, min_periods=MIN_WINDOW)
    df['hist_trend_views'] = roller_3.median() / (df['hist_median_views'] + 1e-5)

    df['hist_median_likes'] = lag_likes.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW).median()
    prev_er = (lag_likes + lag_comments) / (lag_views + 1)
    df['hist_avg_er'] = prev_er.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW).mean()
    
    df['hist_window_size'] = lag_views.rolling(window=MAX_WINDOW, min_periods=0).count()

    # ==========================================
    # 🩹 ЧАСТЬ 4: IMPUTATION
    # ==========================================
    print("🩹 Заполнение пропусков (Imputation)...")
    df['is_new_account'] = (df['hist_window_size'] < 3).astype(int)
    
    values_map = {
        'hist_median_views': DEFAULT_VIEWS,
        'hist_mean_views': DEFAULT_VIEWS,
        'hist_std_views': 0.0,
        'hist_trend_views': DEFAULT_TREND,
        'hist_median_likes': 0.0,
        'hist_avg_er': DEFAULT_ER,
        'hist_days_since_last': DEFAULT_DAYS,
        'hist_post_freq': 0.0
    }
    df = df.fillna(value=values_map)

    # ==========================================
    # 🧹 ЧАСТЬ 5: SAVE
    # ==========================================
    contract_columns = [
        'account_id', 'video_id', 'upload_date', 'video_text',
        'target_views_log',
        'st_log_followers', 'st_ratio_ff', 'st_is_verified', 'st_is_business', 
        'st_bio_length', 'st_hist_total_posts',
        'hist_window_size', 'is_new_account',
        'hist_median_views', 'hist_mean_views', 'hist_std_views', 
        'hist_trend_views', 'hist_median_likes', 'hist_avg_er', 
        'hist_days_since_last'
    ]
    
    final_cols = [c for c in contract_columns if c in df.columns]
    final_df = df[final_cols].copy()
    
    # 1. Сохраняем Parquet (Task DoD)
    # Приводим ID к строке для безопасности Parquet
    final_df['video_id'] = final_df['video_id'].astype(str)
    final_df['account_id'] = final_df['account_id'].astype(str)
    
    print(f"💾 Сохраняем Parquet: {OUTPUT_PARQUET}")
    final_df.to_parquet(OUTPUT_PARQUET, index=False)
    
    # 2. Сохраняем CSV (Sprint Goal)
    print(f"💾 Сохраняем CSV: {OUTPUT_CSV}")
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n✅ FEATURE ENGINEERING ЗАВЕРШЕН!")
    print(f"Итого строк: {len(final_df)}")
    print(f"Без пропусков в медиане: {final_df['hist_median_views'].count()}")

if __name__ == "__main__":
    run_feature_engineering()