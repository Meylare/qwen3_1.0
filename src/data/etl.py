"""
ETL скрипт подготовки данных для Viral Index.
Извлекает данные из train_stage2.parquet, формирует system_prompt,
создает таргеты с логарифмической формулой и готовит XGBoost матрицу.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import json
import numpy as np
import pandas as pd
import xgboost as xgb


INPUT_PARQUET_PATH = Path(__file__).resolve().parents[2] / "train_stage2.parquet"
OUTPUT_JSONL_PATH = "TODO_OUTPUT_JSONL_PATH.jsonl"
VIDEO_ROOT_DIR = "TODO_SERVER_VIDEO_PATH"


def _get_text_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def build_system_prompt(row: pd.Series) -> str:
    author = _get_text_value(row.get("author"))
    bio = _get_text_value(row.get("bio"))
    description = _get_text_value(row.get("description"))
    music = _get_text_value(row.get("music"))
    upload_date = _get_text_value(row.get("upload_date"))

    parts = [
        f"author: {author}",
        f"bio: {bio}",
        f"description: {description}",
        f"music: {music}",
        f"date: {upload_date}",
    ]
    return " | ".join(parts)


def _parse_context_features(df: pd.DataFrame) -> pd.DataFrame:
    if "context_str" not in df.columns:
        return pd.DataFrame(index=df.index)

    def _safe_load(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, str) or not payload.strip():
            return {}
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return {}

    context_series = df["context_str"].map(_safe_load)
    context_df = pd.json_normalize(context_series)
    context_df.index = df.index
    return context_df


def _add_log_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Missing required target column: {target_col}")

    values = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    # Логарифмическая формула: log1p с защитой от отрицательных значений.
    df[f"{target_col}_log"] = np.log1p(values.clip(lower=0))
    return df


def _build_xgb_matrix(df: pd.DataFrame, target_col: str) -> Tuple[xgb.DMatrix, List[str]]:
    context_df = _parse_context_features(df)
    numeric_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col not in {target_col, f"{target_col}_log"}
    ]
    feature_df = pd.concat([context_df, df[numeric_cols]], axis=1)
    if feature_df.empty:
        feature_df = pd.DataFrame({"bias": np.ones(len(df))}, index=df.index)

    dmatrix = xgb.DMatrix(feature_df, label=df[target_col])
    return dmatrix, feature_df.columns.tolist()


def _build_jsonl_records(
    df: pd.DataFrame,
    video_root_dir: str
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        video_id = _get_text_value(row.get("video_id"))
        record = {
            "video_id": video_id,
            "system_prompt": row.get("system_prompt", ""),
            "viral_index": row.get("viral_index"),
            "viral_index_log": row.get("viral_index_log"),
            "video_path": f"{video_root_dir}/{video_id}",
        }
        records.append(record)
    return records


def run_etl(
    input_parquet: Path = INPUT_PARQUET_PATH,
    output_jsonl: str = OUTPUT_JSONL_PATH,
    video_root_dir: str = VIDEO_ROOT_DIR
) -> None:
    df = pd.read_parquet(input_parquet)
    if "viral_index" not in df.columns:
        raise ValueError("Viral index column not found in input data.")

    df = df.copy()
    df["system_prompt"] = df.apply(build_system_prompt, axis=1)
    df = _add_log_target(df, "viral_index")

    _dmatrix, feature_cols = _build_xgb_matrix(df, "viral_index")
    print(f"[INFO] XGBoost features prepared: {len(feature_cols)} columns.")

    records = _build_jsonl_records(df, video_root_dir)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] JSONL saved to {output_path}")


if __name__ == "__main__":
    run_etl()
