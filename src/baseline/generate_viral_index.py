import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import os

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = "dataset_v0_complete.parquet"
# Убедись, что папка совпадает с той, что в train.py
ARTIFACTS_DIR = "XGBoost/artifacts_real_clean_data" 

OUTPUT_TRAIN = "dataset_v1_train.csv"
OUTPUT_VAL = "dataset_v1_val.csv"

def create_context_string(row):
    """
    Формирует контекст для нейросети строго по ТЗ.
    Важно: Переводим логарифмы обратно в реальные числа для удобства модели.
    """
    # 1. Достаем медиану (она у нас в логарифме)
    log_median = row.get('hist_median_views', 0)
    # 2. Переводим в реальные просмотры (exp(x) - 1)
    real_median = int(np.expm1(log_median))
    
    context = {
        # Сила аккаунта (Звезда или Массфоловер?)
        "ratio_ff": round(row.get('st_ratio_ff', 0), 2),
        
        # Ожидаемый уровень (Реальное число просмотров, не логарифм!)
        "expected_views": real_median,
        
        # Статус Новичка (Может ли статистика врать?)
        "is_new": int(row.get('is_new_account', 0))
    }
    return json.dumps(context)

def run_generation():
    print("🚀 Финальная задача: Генерация Viral Index и Context (Real Numbers)...")

    # 1. Проверка путей
    model_path = os.path.join(ARTIFACTS_DIR, "xgboost_baseline.json")
    stats_path = os.path.join(ARTIFACTS_DIR, "fill_stats.joblib")
    feats_path = os.path.join(ARTIFACTS_DIR, "feature_cols.joblib")

    if not os.path.exists(model_path):
        print(f"❌ Ошибка: Не найдена модель в {ARTIFACTS_DIR}.")
        return

    # 2. Загрузка
    print("📥 Загружаем данные и артефакты...")
    df = pd.read_parquet(INPUT_FILE)
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    fill_stats = joblib.load(stats_path)
    feature_cols = joblib.load(feats_path)
    
    print(f"   Данные: {len(df)} строк")

    # 3. Подготовка фичей (Imputation)
    print("⚙️ Подготовка фичей для предикта...")
    # Убеждаемся, что порядок колонок такой же, как при обучении
    X = df[feature_cols].copy()
    X = X.fillna(value=fill_stats)
    
    # 4. Предсказание (Baseline Prediction)
    print("🧠 Модель предсказывает 'Норму' (baseline)...")
    # Получаем предсказание в логарифмах
    df['baseline_log_pred'] = model.predict(X)

    # 5. Расчет Viral Index
    # Формула: Log(Real) - Log(Predicted)
    print("📉 Вычисляем Viral Index...")
    df['viral_index'] = df['target_views_log'] - df['baseline_log_pred']

    # 6. Генерация Context String
    print("📝 Генерируем context_str (JSON с реальными числами)...")
    tqdm.pandas() 
    df['context_str'] = df.progress_apply(create_context_string, axis=1)

    # 7. Сборка финального датафрейма
    final_cols = [
        'video_id',       # Ключ
        'video_text',     # Вход для LLM
        'context_str',    # Вход для LLM (метаданные)
        'viral_index',    # ТАРГЕТ для LLM
        'account_id'      # Для сплита
    ]
    
    export_df = df[final_cols].copy()

    # 8. Сплит на Train/Val (GroupShuffleSplit)
    print("✂️ Разделение на Train / Val (по авторам)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(export_df, groups=export_df['account_id']))
    
    train_df = export_df.iloc[train_idx].drop(columns=['account_id'])
    val_df = export_df.iloc[val_idx].drop(columns=['account_id'])

    # 9. Сохранение
    print(f"💾 Сохраняем {OUTPUT_TRAIN} ({len(train_df)} строк)...")
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    
    print(f"💾 Сохраняем {OUTPUT_VAL} ({len(val_df)} строк)...")
    val_df.to_csv(OUTPUT_VAL, index=False)
    
    print("\n🎉🎉🎉 СПРИНТ ПОЛНОСТЬЮ ЗАВЕРШЕН! 🎉🎉🎉")
    print("Пример строки (обрати внимание на context_str):")
    print(train_df.iloc[0])

if __name__ == "__main__":
    run_generation()