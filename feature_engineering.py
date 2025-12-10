import pandas as pd
import numpy as np
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = "dataset_v1_raw.parquet"
OUTPUT_PARQUET = "full_history_features.parquet"

WINDOW_LONG = 15
WINDOW_SHORT = 4

# ЗАГЛУШКИ
DEFAULT_VIEWS = 300.0
DEFAULT_LIKES = 0.0
DEFAULT_ER = 0.0
DEFAULT_DAYS = -1.0
DEFAULT_FREQ = 0.0
DEFAULT_TREND = 1.0

def run_feature_engineering_v5():
    print("🚀 Запуск Feature Engineering (V5.3: Filter Real Zeros)...")

    # 1. Загрузка
    print(f"📥 Читаем {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    df = df.sort_values(by=['account_id', 'upload_date'], ascending=[True, True])

    # 2. Time Travel
    print("⏳ Time Travel...")
    video_counts = df.groupby('account_id')['video_id'].transform('count')
    df['total_videos_count'] = video_counts 
    
    video_rank = df.groupby('account_id').cumcount()
    videos_after = video_counts - 1 - video_rank
    df['st_hist_total_posts'] = (df['profile_current_posts'] - videos_after).clip(lower=0)

    # ==========================================
    # 🪟 SLIDING WINDOW
    # ==========================================
    print("🪟 Расчет истории...")
    grouped = df.groupby('account_id')

    lag_views = grouped['views'].shift(1)
    lag_likes = grouped['likes'].shift(1)
    lag_comments = grouped['comments'].shift(1)
    lag_date = grouped['upload_date'].shift(1)

    # --- Days Since Last ---
    df['hist_days_since_last'] = (df['upload_date'] - lag_date).dt.total_seconds() / 86400

    # --- Post Frequency ---
    date_diffs = grouped['upload_date'].diff()
    date_diffs_days = date_diffs.dt.total_seconds() / 86400
    lag_date_diffs = date_diffs_days.shift(1)
    df['hist_post_freq'] = lag_date_diffs.rolling(window=WINDOW_LONG, min_periods=1).mean()

    # --- Views ---
    roller_15 = lag_views.rolling(window=WINDOW_LONG, min_periods=1)
    df['hist_median_views'] = roller_15.median()
    df['hist_mean_views'] = roller_15.mean()
    df['hist_std_views'] = roller_15.std()

    roller_4 = lag_views.rolling(window=WINDOW_SHORT, min_periods=1)
    median_4 = roller_4.median()
    df['hist_trend_views'] = median_4 / (df['hist_median_views'] + 1e-5)

    # --- Likes & ER ---
    df['hist_median_likes'] = lag_likes.rolling(window=WINDOW_LONG, min_periods=1).median()
    prev_er = (lag_likes + lag_comments) / (lag_views + 1)
    df['hist_avg_er'] = prev_er.rolling(window=WINDOW_LONG, min_periods=1).mean()

    # --- Размер истории ---
    df['hist_window_size'] = grouped.cumcount()

    # ==========================================
    # ✂️ ФИЛЬТРАЦИЯ И ЗАГЛУШКИ
    # ==========================================
    print("⚖️ Фильтрация и Импутация...")

    # 1. Логика "Старичков"
    mask_keep = (df['total_videos_count'] < 15) | (df['hist_window_size'] >= 4)
    print(f"   Было строк: {len(df)}")
    df = df[mask_keep].copy()
    print(f"   Стало строк (после обрезки начала): {len(df)}")

    # 2. Логика "Новичков" (заглушки)
    mask_impute = df['hist_window_size'] < 2
    
    values_map = {
        'hist_median_views': DEFAULT_VIEWS, 'hist_mean_views': DEFAULT_VIEWS,
        'hist_std_views': 0.0, 'hist_trend_views': DEFAULT_TREND,
        'hist_median_likes': DEFAULT_LIKES, 'hist_avg_er': DEFAULT_ER,
        'hist_post_freq': DEFAULT_FREQ
    }
    
    for col, val in values_map.items():
        if col in df.columns:
            df.loc[mask_impute, col] = val

    # Отдельная обработка для дней
    df['hist_days_since_last'] = df['hist_days_since_last'].fillna(DEFAULT_DAYS)

    # --- 🔥 НОВЫЙ БЛОК: УДАЛЕНИЕ МЕРТВЫХ ДУШ 🔥 ---
    print("🧹 Очистка: Удаляем видео с РЕАЛЬНЫМ нулем в медиане лайков...")
    initial_len = len(df)
    
    # Удаляем, если (Лайков в истории 0) И (Это НЕ заглушка, т.е. history >= 2)
    # Заглушки для новичков (history < 2) не трогаем, им положено быть с нулем.
    mask_dead_accounts = (df['hist_median_likes'] == 0) & (df['hist_window_size'] >= 2)
    
    df = df[~mask_dead_accounts].copy()
    
    dropped = initial_len - len(df)
    print(f"   ✂️ Удалено {dropped} строк (мертвые аккаунты без лайков).")

    # ==========================================
    # 🚩 IS_NEW_ACCOUNT
    # ==========================================
    df['is_new_account'] = (df['total_videos_count'] < 15).astype(int)

    # ==========================================
    # 📐 ЛОГАРИФМЫ И СОХРАНЕНИЕ
    # ==========================================
    print("📐 Финализация...")
    
    log_cols = ['hist_median_views', 'hist_mean_views', 'hist_std_views', 'hist_median_likes']
    for col in log_cols:
        df[col] = np.log1p(df[col].clip(lower=0))

    df['target_views_log'] = np.log1p(df['views'])
    
    # Доп. таргеты
    df['target_likes_log'] = np.log1p(df['likes'].clip(lower=0))
    df['target_comments_log'] = np.log1p(df['comments'].clip(lower=0))

    df['video_id'] = df['video_id'].astype(str)
    df['account_id'] = df['account_id'].astype(str)

    cols_to_save = [
        'account_id', 'video_id', 'upload_date', 'video_text',
        'target_views_log', 'target_likes_log', 'target_comments_log',
        'st_hist_total_posts', 'is_new_account',
        'hist_window_size', 'hist_days_since_last', 'hist_post_freq',
        'hist_median_views', 'hist_mean_views', 'hist_std_views',
        'hist_trend_views', 'hist_median_likes', 'hist_avg_er',
        'raw_followers', 'raw_following', 'raw_bio', 'st_is_verified', 'st_is_business'
    ]
    
    final_cols = [c for c in cols_to_save if c in df.columns]
    df_final = df[final_cols]

    print(f"💾 Сохраняем: {OUTPUT_PARQUET}")
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print("✅ ГОТОВО! V5.3 (Clean Zeros).")

if __name__ == "__main__":
    run_feature_engineering_v5()