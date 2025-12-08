import pandas as pd
import numpy as np
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = "dataset_v1_raw.parquet"
OUTPUT_PARQUET = "full_history_features.parquet" # Результат ТОЛЬКО этой задачи

MIN_WINDOW = 3
MAX_WINDOW = 12

# Значения для импутации
DEFAULT_VIEWS = 300.0
DEFAULT_TREND = 1.0
DEFAULT_ER = 0.0
DEFAULT_DAYS = -1.0

def run_sliding_window_task():
    print("🚀 Запуск задачи: Sliding Window & Time Travel (Task 3)...")

    # 1. Загрузка
    print(f"📥 Читаем {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    df = df.sort_values(by=['account_id', 'upload_date'], ascending=[True, True])

    # ==========================================
    # ⏳ ЧАСТЬ 1: TIME TRAVEL LOGIC
    # ==========================================
    print("⏳ Выполнение Time Travel...")
    # Нам нужно узнать кол-во постов в момент публикации.
    # Это часть динамики, поэтому оставляем здесь.
    video_counts = df.groupby('account_id')['video_id'].transform('count')
    video_rank = df.groupby('account_id').cumcount()
    videos_after = video_counts - 1 - video_rank
    
    # Расчет исторического кол-ва постов
    df['st_hist_total_posts'] = (df['profile_current_posts'] - videos_after).clip(lower=0)

    # ==========================================
    # 🪟 ЧАСТЬ 2: SLIDING WINDOW ENGINE
    # ==========================================
    print("🪟 Расчет скользящего окна (Sliding Window)...")
    grouped = df.groupby('account_id')
    
    # Создаем лаги (сдвиг назад), чтобы не видеть будущее
    lag_views = grouped['views'].shift(1)
    lag_likes = grouped['likes'].shift(1)
    lag_comments = grouped['comments'].shift(1)
    lag_date = grouped['upload_date'].shift(1)
    
    # Дней с прошлого видео
    df['hist_days_since_last'] = (df['upload_date'] - lag_date).dt.total_seconds() / 86400
    
    # Агрегаты по окну
    roller_12 = lag_views.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW)
    df['hist_median_views'] = roller_12.median()
    df['hist_mean_views'] = roller_12.mean()
    df['hist_std_views'] = roller_12.std()
    
    # Тренд
    roller_3 = lag_views.rolling(window=3, min_periods=MIN_WINDOW)
    df['hist_trend_views'] = roller_3.median() / (df['hist_median_views'] + 1e-5)

    # ER и Лайки
    df['hist_median_likes'] = lag_likes.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW).median()
    prev_er = (lag_likes + lag_comments) / (lag_views + 1)
    df['hist_avg_er'] = prev_er.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW).mean()
    
    # Размер окна
    df['hist_window_size'] = lag_views.rolling(window=MAX_WINDOW, min_periods=0).count()

    # ==========================================
    # 🩹 ЧАСТЬ 3: IMPUTATION (Обработка пустот)
    # ==========================================
    print("🩹 Заполнение пропусков (Imputation)...")
    # Флаг нового аккаунта
    df['is_new_account'] = (df['hist_window_size'] < 3).astype(int)
    
    # Заполнение
    values_map = {
        'hist_median_views': DEFAULT_VIEWS,
        'hist_mean_views': DEFAULT_VIEWS,
        'hist_std_views': 0.0,
        'hist_trend_views': DEFAULT_TREND,
        'hist_median_likes': 0.0,
        'hist_avg_er': DEFAULT_ER,
        'hist_days_since_last': DEFAULT_DAYS
    }
    df = df.fillna(value=values_map)

    # ==========================================
    # 💾 СОХРАНЕНИЕ (Только результат этой задачи)
    # ==========================================
    # Мы пока НЕ сохраняем финальный датасет и не считаем статику (bio, followers log).
    # Мы сохраняем промежуточный файл с историей, как просит карточка.
    
    # Приводим ID к строке
    df['video_id'] = df['video_id'].astype(str)
    df['account_id'] = df['account_id'].astype(str)
    
    print(f"💾 Сохраняем результат задачи: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    
    print("\n✅ TASK 3 (SLIDING WINDOW) DONE!")

if __name__ == "__main__":
    run_sliding_window_task()