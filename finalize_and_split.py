import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

# --- КОНФИГУРАЦИЯ ---
INPUT_FILE = "dataset_v0_complete.parquet" # Твой результат Merge
TRAIN_OUTPUT = "train.csv"
VAL_OUTPUT = "val.csv"

def run_split():
    print("✂️ Финальный этап: Сплит на Train / Val...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Нет файла {INPUT_FILE}. Сначала сделай Merge.")
        return

    # 1. Читаем
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Всего данных: {len(df)} строк")

    # 2. Group Shuffle Split
    # Это гарантирует, что все видео одного блогера попадут ВМЕСТЕ либо в Train, либо в Val
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.02, random_state=42)
    
    train_inds, val_inds = next(splitter.split(df, groups=df['account_id']))
    
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    # 3. Проверка на утечки (Leakage Check)
    train_authors = set(train_df['account_id'])
    val_authors = set(val_df['account_id'])
    
    intersect = train_authors.intersection(val_authors)
    if intersect:
        print(f"❌ ОШИБКА! {len(intersect)} авторов попали и туда и сюда.")
        return
    else:
        print("✅ Проверка пройдена: Пересечений по авторам нет.")

    # 4. Сохранение
    print(f"💾 Сохраняем train.csv ({len(train_df)} строк)...")
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    
    print(f"💾 Сохраняем val.csv ({len(val_df)} строк)...")
    val_df.to_csv(VAL_OUTPUT, index=False)

    print("\n🎉 СПРИНТ ЗАКРЫТ! Файлы готовы для обучения.")

if __name__ == "__main__":
    run_split()