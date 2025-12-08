import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import logging
import datetime
import joblib
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Параметры модели (единое место для CV и финальной модели)
MODEL_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 500,  # увеличено для early stopping
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50
}

def generate_mock_data(n_samples=1000):
    """
    Генерирует синтетический датасет согласно Data Contract.
    """
    logger.info(f"Генерация Mock Data ({n_samples} строк)...")
    
    np.random.seed(42)
    
    # 1. Идентификаторы (Meta)
    # Генерируем ~100 уникальных авторов
    n_accounts = 100
    account_ids = [f"acc_{i:03d}" for i in range(n_accounts)]
    
    data = {
        'account_id': np.random.choice(account_ids, n_samples),
        'video_id': [f"vid_{i:05d}" for i in range(n_samples)],
        'upload_date': [datetime.date(2023, 1, 1) + datetime.timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)],
        'video_text': [f"Video description {i}" for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # 2. Таргет (Target)
    # log1p от случайных просмотров (экспоненциальное распределение)
    raw_views = np.random.exponential(scale=5000, size=n_samples)
    df['target_views_log'] = np.log1p(raw_views)
    
    # 3. Статика (Static Features)
    df['st_log_followers'] = np.random.uniform(0, 15, n_samples) # log(1) to log(3M+)
    df['st_ratio_ff'] = np.random.exponential(1.0, n_samples)
    df['st_is_verified'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    df['st_is_business'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    df['st_bio_length'] = np.random.randint(0, 150, n_samples)
    df['st_hist_total_posts'] = np.random.randint(0, 1000, n_samples)
    
    # 4. Исторические (History / Rolling Window)
    # Генерируем размер окна
    df['hist_window_size'] = np.random.randint(0, 13, n_samples) # 0..12
    
    # Симуляция исторических значений
    df['hist_median_views'] = np.log1p(np.random.exponential(3000, n_samples))
    df['hist_mean_views'] = df['hist_median_views'] * np.random.uniform(0.8, 1.2, n_samples)
    df['hist_std_views'] = df['hist_mean_views'] * np.random.uniform(0.3, 0.7, n_samples)  # вариация
    df['hist_trend_views'] = np.clip(np.random.normal(1.0, 0.3, n_samples), 0.1, 3.0)  # clip > 0
    df['hist_median_likes'] = df['hist_median_views'] * 0.1
    df['hist_avg_er'] = np.random.uniform(0.01, 0.20, n_samples)
    df['hist_days_since_last'] = np.random.exponential(5, n_samples)
    df['hist_post_freq'] = np.random.uniform(0, 2, n_samples)

    # ВНЕДРЕНИЕ ПРОПУСКОВ (NaN) для новых аккаунтов (hist_window_size < 3)
    # Как указано в Tech Specs: "если hist_window_size < 3... Разработчик 1 может оставлять NaN"
    mask_new_account = df['hist_window_size'] < 3
    
    # Ставим NaN для исторических метрик новых аккаунтов
    cols_to_nan = [
        'hist_median_views', 'hist_mean_views', 'hist_std_views',
        'hist_trend_views', 'hist_median_likes', 'hist_avg_er',
        'hist_days_since_last', 'hist_post_freq'
    ]
    
    df.loc[mask_new_account, cols_to_nan] = np.nan
    
    logger.info(f"Сгенерировано {len(df)} строк. Пропусков в hist_median_views: {df['hist_median_views'].isna().sum()}")
    
    return df

def preprocess_data(df, fill_values=None):
    """
    Заполняет пропуски (Imputation) согласно Tech Specs.
    Если fill_values передан, использует его значения.
    Иначе вычисляет медианы по df и возвращает их вместе с df_clean.
    """
    
    df_clean = df.copy()
    
    if fill_values is None:
        # 1. Views -> Глобальная медиана по текущей выборке, fallback = 300 (контракт)
        global_median_views = df['hist_median_views'].median()
        if np.isnan(global_median_views):
            global_median_views = 300.0
            logger.warning("Imputation: hist_median_views медиана NaN, используем дефолт 300.0")
        
        # 2. Вычисляем медианы для остальных полей (fallback = пропорции из mock data)
        global_median_std = df['hist_std_views'].median()
        if np.isnan(global_median_std):
            global_median_std = global_median_views * 0.5
        
        global_median_likes = df['hist_median_likes'].median()
        if np.isnan(global_median_likes):
            global_median_likes = global_median_views * 0.1
        
        global_median_er = df['hist_avg_er'].median()
        if np.isnan(global_median_er):
            global_median_er = 0.05
        
        global_median_freq = df['hist_post_freq'].median()
        if np.isnan(global_median_freq):
            global_median_freq = 0.5
        
        global_median_days = df['hist_days_since_last'].median()
        if np.isnan(global_median_days):
            global_median_days = 7.0  # fallback: неделя
        
        logger.info(f"Imputation: views={global_median_views:.2f}, std={global_median_std:.2f}, "
                    f"likes={global_median_likes:.2f}, er={global_median_er:.3f}, days={global_median_days:.1f}")
        
        fill_values = {
            'hist_median_views': global_median_views,
            'hist_mean_views': global_median_views,
            'hist_std_views': global_median_std,
            'hist_median_likes': global_median_likes,
            'hist_avg_er': global_median_er,
            'hist_post_freq': global_median_freq,
            'hist_trend_views': 1.0,
            'hist_days_since_last': global_median_days
        }
        return_stats = True
    else:
        return_stats = False
        
    # Применяем fill_values
    for col, val in fill_values.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(val)
            
    if return_stats:
        return df_clean, fill_values
    return df_clean

def train_model(df, save_path=None):
    """
    Обучает XGBoost с GroupKFold валидацией.
    Внутри цикла CV делает imputation, чтобы избежать Data Leakage.
    
    Returns:
        tuple: (final_model, fill_stats, feature_cols)
    """
    logger.info("Настройка обучения модели...")
    
    # Выбор фичей: все st_ и все hist_ включая hist_window_size (важно для cold start)
    feature_cols = [c for c in df.columns if c.startswith('st_') or c.startswith('hist_')]
    target_col = 'target_views_log'
    group_col = 'account_id'
    
    X = df[feature_cols]
    y = df[target_col]
    groups = df[group_col]
    
    logger.info(f"Фичи ({len(feature_cols)}): {feature_cols}")
    
    # GroupKFold Split
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    
    rmse_scores = []
    
    logger.info(f"Запуск Cross-Validation ({n_splits} folds)...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        # Разбиваем на трейн и валидацию (пока с NaN)
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. Imputation на Train (вычисляем статистики)
        X_train_clean, fill_stats = preprocess_data(X_train_raw)
        
        # 2. Imputation на Val (применяем статистики с трейна)
        X_val_clean = preprocess_data(X_val_raw, fill_values=fill_stats)
        
        # Обучение с early stopping
        model = xgb.XGBRegressor(**MODEL_PARAMS)
        model.fit(
            X_train_clean, y_train,
            eval_set=[(X_val_clean, y_val)],
            verbose=False
        )
        
        # Предсказание
        preds = model.predict(X_val_clean)
        
        # Метрика
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
        
        logger.info(f"Fold {fold}: RMSE = {rmse:.4f} "
                    f"(Train: {len(X_train_clean)}, Val: {len(X_val_clean)}, "
                    f"Trees: {model.best_iteration + 1})")
        
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores, ddof=1)  # sample std
    
    logger.info(f"CV Results: Mean RMSE = {avg_rmse:.4f} +/- {std_rmse:.4f}")
    
    # Финальная модель: обучаем на всем датасете
    X_full_clean, full_fill_stats = preprocess_data(X)
    
    # Для финальной модели используем среднее кол-во деревьев из CV (без early stopping)
    final_params = MODEL_PARAMS.copy()
    final_params.pop('early_stopping_rounds', None)
    final_params['n_estimators'] = 100  # фиксированное для финала
    
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_full_clean, y)
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Top-5 Features:\n{importance.head().to_string(index=False)}")
    
    # Сохранение модели и статистик
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(final_model, save_path / 'model.joblib')
        joblib.dump(full_fill_stats, save_path / 'fill_stats.joblib')
        joblib.dump(feature_cols, save_path / 'feature_cols.joblib')
        logger.info(f"Model saved to {save_path}")
    
    return final_model, full_fill_stats, feature_cols

if __name__ == "__main__":
    try:
        # 1. Генерация
        mock_df = generate_mock_data(n_samples=2000)
        
        # 2. Обучение (imputation внутри CV для предотвращения data leakage)
        final_model, fill_stats, feature_cols = train_model(
            mock_df, 
            save_path='XGBoost/artifacts'
        )
        
        logger.info("Прототип обучения завершен успешно.")
        
    except Exception as e:
        logger.error(f"Ошибка в пайплайне: {e}")
        raise