import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = "dataset_v1_raw.parquet"
OUTPUT_PARQUET = "full_history_features.parquet"

# ВАЖНО: Ставим 1, чтобы включить "Накопительную медиану"
# Теперь мы не ждем 3 видео, а используем то, что есть.
MIN_WINDOW = 1 
MAX_WINDOW = 12
TREND_EPS = 1e-5

# Значения для импутации (Только для самого ПЕРВОГО видео, где истории нет вообще)
DEFAULT_VIEWS = 300.0
DEFAULT_TREND = 1.0
DEFAULT_ER = 0.0
DEFAULT_DAYS = -1.0 # Маркер отсутствия предыдущего видео

def run_sliding_window_task():
    print("🚀 Запуск задачи: Sliding Window & Time Travel (Task 3 & 4)...")

    # 1. Загрузка
    print(f"📥 Читаем {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # КРИТИЧНО: Сортировка для правильной работы shift()
    df = df.sort_values(by=['account_id', 'upload_date'], ascending=[True, True])

    # ==========================================
    # ⏳ ЧАСТЬ 1: TIME TRAVEL LOGIC (Posts Count)
    # ==========================================
    print("⏳ Выполнение Time Travel (корректировка счетчика постов)...")
    
    # Считаем, сколько видео у автора ВПЕРЕДИ (относительно текущей строки)
    # cumcount дает 0, 1, 2...
    # transform('count') дает общее число
    # future_videos = total - 1 - current_rank
    
    video_counts = df.groupby('account_id')['video_id'].transform('count')
    video_rank = df.groupby('account_id').cumcount()
    videos_after = video_counts - 1 - video_rank
    
    # Вычитаем будущие видео из текущего счетчика в профиле
    df['st_hist_total_posts'] = (df['profile_current_posts'] - videos_after).clip(lower=0)

    # ==========================================
    # 🪟 ЧАСТЬ 2: SLIDING WINDOW ENGINE
    # ==========================================
    print("🪟 Расчет скользящего окна (Cumulative Logic)...")
    grouped = df.groupby('account_id')
    
    # 1. Создаем ЛАГИ (сдвиг назад на 1 шаг)
    # Мы не имеем права видеть текущие просмотры при расчете истории!
    lag_views = grouped['views'].shift(1)
    lag_likes = grouped['likes'].shift(1)
    lag_comments = grouped['comments'].shift(1)
    lag_date = grouped['upload_date'].shift(1)
    
    # 2. Дней с прошлого видео
    # (Текущая дата - Дата предыдущего видео)
    df['hist_days_since_last'] = (df['upload_date'] - lag_date).dt.total_seconds() / 86400
    
    # 3. Основные Агрегаты (Views)
    # min_periods=1 включает накопительную логику
    roller_12 = lag_views.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW)
    
    df['hist_median_views'] = roller_12.median()
    df['hist_mean_views'] = roller_12.mean()
    df['hist_std_views'] = roller_12.std() # Будет NaN, если в истории 1 элемент (это норм)
    
    # 4. Тренд (Короткое окно / Длинное окно)
    # min_periods=1 позволяет считать тренд уже на 2-м видео (будет 1.0)
    roller_3 = lag_views.rolling(window=3, min_periods=MIN_WINDOW)
    # Добавляем эпсилон, чтобы не делить на ноль и сохранить масштаб сырого отношения
    df['hist_trend_views'] = roller_3.median() / (df['hist_median_views'] + TREND_EPS)

    # 5. ER и Лайки
    df['hist_median_likes'] = lag_likes.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW).median()
    
    # ER считаем построчно, потом усредняем
    # +1 защита от деления на ноль
    prev_er_series = (lag_likes + lag_comments) / (lag_views + 1)
    df['hist_avg_er'] = prev_er_series.rolling(window=MAX_WINDOW, min_periods=MIN_WINDOW).mean()
    
    # 6. Размер окна (Честный счетчик)
    # Показывает, сколько реально видео взято в расчет (0, 1, 2 ... 12)
    df['hist_window_size'] = lag_views.rolling(window=MAX_WINDOW, min_periods=0).count()

    # ==========================================
    # 🩹 ЧАСТЬ 3: IMPUTATION (Заполнение пустот)
    # ==========================================
    print("🩹 Заполнение пропусков (Imputation)...")
    
    # is_new_account теперь ставим, если истории ВООБЩЕ нет (window_size == 0)
    # Или можно оставить < 3 как логический флаг, но данные уже будут заполнены
    df['is_new_account'] = (df['hist_window_size'] == 0).astype(int)
    
    # Словарь заполнения (Сработает только для самых первых видео аккаунтов)
    values_map = {
        'hist_median_views': DEFAULT_VIEWS,
        'hist_mean_views': DEFAULT_VIEWS,
        'hist_std_views': 0.0,      # Если 1 видео, отклонение 0
        'hist_trend_views': DEFAULT_TREND,
        'hist_median_likes': 0.0,
        'hist_avg_er': DEFAULT_ER,
        'hist_days_since_last': DEFAULT_DAYS
    }
    df = df.fillna(value=values_map)

    # ==========================================
    # 📐 ЧАСТЬ 4: LOG1P ДЛЯ АБСОЛЮТНЫХ АГРЕГАТОВ
    # ==========================================
    # Логарифмируем только абсолютные величины после расчётов и импутации,
    # оставляя тренд/ER в исходной шкале.
    log_cols = [
        'hist_median_views',
        'hist_mean_views',
        'hist_std_views',
        'hist_median_likes'
    ]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # ==========================================
    # 💾 СОХРАНЕНИЕ
    # ==========================================
    # Приводим ID к строке для безопасности
    df['video_id'] = df['video_id'].astype(str)
    df['account_id'] = df['account_id'].astype(str)
    
    print(f"💾 Сохраняем промежуточный файл: {OUTPUT_PARQUET}")
    print(f"📊 Размер датасета: {df.shape}")
    
    # Проверка на дурака (вывод пары строк)
    print("\n🔍 Пример данных (Window Size = 1):")
    print(df[df['hist_window_size'] == 1][['account_id', 'views', 'hist_median_views', 'hist_trend_views']].head(2))
    
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print("\n✅ TASK 3 DONE! Файл готов для ML.")

if __name__ == "__main__":
    run_sliding_window_task()