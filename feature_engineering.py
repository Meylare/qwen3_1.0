import pandas as pd
import numpy as np
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = "dataset_v1_raw.parquet"
OUTPUT_PARQUET = "full_history_features.parquet"

MIN_WINDOW = 3
MAX_WINDOW = 12

# Значения для импутации
DEFAULT_VIEWS = 300.0
DEFAULT_TREND = 1.0
DEFAULT_ER = 0.0
DEFAULT_DAYS = -1.0

def run_sliding_window_task():
    print("🚀 Запуск задачи: Sliding Window & Time Travel (FIXED)...")

    # 1. Загрузка
    print(f"📥 Читаем {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    df = df.sort_values(by=['account_id', 'upload_date'], ascending=[True, True])

    # ==========================================
    # ⏳ ЧАСТЬ 1: TIME TRAVEL LOGIC
    # ==========================================
    print("⏳ Выполнение Time Travel...")
    video_counts = df.groupby('account_id')['video_id'].transform('count')
    video_rank = df.groupby('account_id').cumcount()
    videos_after = video_counts - 1 - video_rank
    df['st_hist_total_posts'] = (df['profile_current_posts'] - videos_after).clip(lower=0)

    # ==========================================
    # 🪟 ЧАСТЬ 2: SLIDING WINDOW ENGINE
    # ==========================================
    print("🪟 Расчет скользящего окна (Sliding Window)...")
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
    
    # --- 🔥 ВОТ ИСПРАВЛЕНИЕ 🔥 ---
    # Вместо сложного rolling().count() используем простой и надежный cumcount().
    # cumcount() для каждой группы создает счетчик [0, 1, 2, 3, ...], что нам и нужно.
    # Мы ограничиваем его сверху значением MAX_WINDOW, так как история дальше нам не важна.
    print("   ... (FIX) Расчет hist_window_size через cumcount...")
    df['hist_window_size'] = grouped.cumcount().clip(upper=MAX_WINDOW)

    # ==========================================
    # 🩹 ЧАСТЬ 3: IMPUTATION (Обработка пустот)
    # ==========================================
    print("🩹 Заполнение пропусков (Imputation)...")
    df['is_new_account'] = (df['hist_window_size'] < 3).astype(int)
    
    values_map = {
        'hist_median_views': DEFAULT_VIEWS, 'hist_mean_views': DEFAULT_VIEWS,
        'hist_std_views': 0.0, 'hist_trend_views': DEFAULT_TREND,
        'hist_median_likes': 0.0, 'hist_avg_er': DEFAULT_ER,
        'hist_days_since_last': DEFAULT_DAYS
    }
    df = df.fillna(value=values_map)

    # ==========================================
    # 💾 СОХРАНЕНИЕ
    # ==========================================
    df['video_id'] = df['video_id'].astype(str)
    df['account_id'] = df['account_id'].astype(str)
    
    print(f"💾 Сохраняем результат задачи: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    
    print("\n✅ TASK 3 (SLIDING WINDOW) DONE! (FIXED)")

if __name__ == "__main__":
    run_sliding_window_task()