import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import logging
import joblib
from pathlib import Path

# ... (все конфиги и функция preprocess_data без изменений) ...
# --- КОНФИГУРАЦІЯ ---
INPUT_FILE = "dataset_v0_complete.parquet"
MODEL_SAVE_DIR = "XGBoost/artifacts_real"
TARGET_COL = 'target_views_log'
GROUP_COL = 'account_id'

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
        global_median_views = df['hist_median_views'].median()
        if pd.isna(global_median_views): global_median_views = 300.0
        fill_values = {
            'hist_median_views': global_median_views, 'hist_mean_views': global_median_views,
            'hist_std_views': df['hist_std_views'].median(), 'hist_median_likes': df['hist_median_likes'].median(),
            'hist_avg_er': df['hist_avg_er'].median(), 'hist_trend_views': 1.0,
            'hist_days_since_last': df['hist_days_since_last'].median()
        }
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
    """
    Расчет SMAPE (Symmetric Mean Absolute Percentage Error).
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 # Делим на 2 для классической формулы
    # Используем where, чтобы избежать деления на ноль, если оба значения 0
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

def train_model(df, save_path=None):
    logger.info("Настройка обучения модели...")
    feature_cols = [c for c in df.columns if c.startswith('st_') or c.startswith('hist_') or c == 'is_new_account']
    X = df[feature_cols]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    
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
        
        # --- Используем SMAPE ---
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
    
    # ... (остальная часть без изменений)
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