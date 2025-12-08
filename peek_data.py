import pandas as pd

# Настройки, чтобы Pandas не прятал колонки и текст
pd.set_option('display.max_columns', None)  # Показать все колонки
pd.set_option('display.width', 1000)        # Ширина терминала
pd.set_option('display.max_colwidth', 50)   # Обрезать слишком длинный текст

df = pd.read_parquet("dataset_v1_raw.parquet")

print("🔍 Первые 5 строк твоего датасета:")
print("-" * 80)
print(df.head(5))
print("-" * 80)

# Показать пример одного конкретного видео вертикально (удобно читать)
print("\n📝 Пример одной записи (вертикально):")
print(df.iloc[0])
