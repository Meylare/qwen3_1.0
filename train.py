import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import logging
import joblib
from pathlib import Path

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = "dataset_v0_complete.parquet"
MODEL_SAVE_DIR = "XGBoost/artifacts_real_clean_data"
TARGET_COL = 'target_views_log'
GROUP_COL = 'account_id'

# --- 🔥 НАСТРОЙКА ФИЧЕЙ (Черный список) 🔥 ---
# Сюда пиши названия колонок, которые ты хочешь ИСКЛЮЧИТЬ из обучения.
# Например, если считаешь, что они шумят.
IGNORE_COLS = [
    'hist_days_since_last',
    'st_is_verified',
    'st_log_followers'
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PARAMS = {
    'objective': 'reg:squarederror', 'n_estimators': 1000,
    'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1,
    'early_stopping_rounds': 50
}

def preprocess_data(df, fill_values=None):
    df_clean = df.copy()
    
    if fill_values is None:
        fill_values = {}
        
        # 1. Логика для Views (Основная заглушка)
        # Если hist_median_views есть, считаем медиану, иначе дефолт 300
        if 'hist_median_views' in df.columns:
            global_median_views = df['hist_median_views'].median()
            if pd.isna(global_median_views): global_median_views = 300.0
            fill_values['hist_median_views'] = global_median_views
        else:
            global_median_views = 300.0 # На случай, если и эту колонку удалили
            
        if 'hist_mean_views' in df.columns:
            fill_values['hist_mean_views'] = global_median_views

        # 2. Динамический расчет медиан для остальных колонок
        # Мы проверяем: "А есть ли эта колонка вообще?", прежде чем считать
        potential_cols = [
            'hist_std_views', 'hist_median_likes', 
            'hist_avg_er', 'hist_days_since_last'
        ]
        
        for col in potential_cols:
            if col in df.columns:
                fill_values[col] = df[col].median()

        # 3. Хардкод (Тренд)
        if 'hist_trend_views' in df.columns:
            fill_values['hist_trend_views'] = 1.0

        # Защита от NaN в самих медианах
        for k, v in fill_values.items():
            if pd.isna(v): fill_values[k] = 0.0
            
        return_stats = True
    else:
        return_stats = False
        
    df_clean = df_clean.fillna(value=fill_values)
    
    if return_stats:
        return df_clean, fill_values
    return df_clean

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

def train_model(df, save_path=None):
    logger.info("Настройка обучения модели...")
    
    # 1. Автоматический отбор всех потенциальных фичей
    feature_cols = [c for c in df.columns if c.startswith('st_') or c.startswith('hist_') or c == 'is_new_account']
    
    # 2. Фильтрация через Черный список (IGNORE_COLS)
    feature_cols = [c for c in feature_cols if c not in IGNORE_COLS]
    
    if IGNORE_COLS:
        logger.info(f"🚫 ИГНОРИРУЕМ колонки: {IGNORE_COLS}")
        
    X = df[feature_cols]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    
    logger.info(f"✅ Используем фичей ({len(feature_cols)}): {feature_cols}")
    
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    
    rmse_scores, r2_scores, smape_scores = [], [], []
    
    logger.info(f"Запуск Cross-Validation ({n_splits} folds)...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        X_train_clean, fill_stats = preprocess_data(X_train_raw)
        X_val_clean = preprocess_data(X_val_raw, fill_values=fill_stats)
        
        model = xgb.XGBRegressor(**MODEL_PARAMS)
        model.fit(X_train_clean, y_train, eval_set=[(X_val_clean, y_val)], verbose=False)
        
        preds = model.predict(X_val_clean)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        
        real_views = np.expm1(y_val)
        pred_views = np.expm1(preds)
        s_mape = smape(real_views, pred_views)

        rmse_scores.append(rmse)
        r2_scores.append(r2)
        smape_scores.append(s_mape)
        
        logger.info(f"Fold {fold}: RMSE={rmse:.4f}, R2={r2:.4f}, SMAPE={s_mape:.2f}%")
        
    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_smape = np.mean(smape_scores)
    
    print("-" * 40)
    logger.info(f"CV Results: Mean RMSE = {avg_rmse:.4f}")
    logger.info(f"CV Results: Mean R² = {avg_r2:.4f}")
    logger.info(f"CV Results: Mean SMAPE = {avg_smape:.2f}%")
    print("-" * 40)
    
    logger.info("Обучение финальной модели на всех данных...")
    X_full_clean, full_fill_stats = preprocess_data(X)
    
    final_params = MODEL_PARAMS.copy()
    final_params.pop('early_stopping_rounds')
    final_params['n_estimators'] = 150

    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_full_clean, y)
    
    importance = pd.DataFrame({'feature': feature_cols, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
    
    logger.info(f"Top-5 Features:\n{importance.head().to_string(index=False)}")
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        final_model.save_model(save_path / 'xgboost_baseline.json')
        joblib.dump(full_fill_stats, save_path / 'fill_stats.joblib')
        joblib.dump(feature_cols, save_path / 'feature_cols.joblib')
        logger.info(f"Модель сохранена в {save_path}")
    
    return final_model

if __name__ == "__main__":
    try:
        logger.info(f"Загрузка реальных данных из {INPUT_FILE}...")
        real_df = pd.read_parquet(INPUT_FILE)
        train_model(real_df, save_path=MODEL_SAVE_DIR)
        logger.info("Скрипт обучения на реальных данных завершен успешно.")
    except FileNotFoundError:
        logger.error(f"Файл не найден: {INPUT_FILE}.")
    except Exception as e:
        logger.error(f"Ошибка в пайплайне: {e}", exc_info=True)