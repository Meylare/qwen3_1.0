import pandas as pd
import numpy as np

# Загружаем датасет
print("📥 Загружаем dataset_v0_complete.parquet...")
df = pd.read_parquet('dataset_v0_complete.parquet')

print(f"📊 Общее количество строк в датасете: {len(df):,}")
print()

# Подсчитываем -1 в hist_days_since_last
count_minus_one = (df['hist_days_since_last'] == -1).sum()
percentage = (count_minus_one / len(df)) * 100

print("="*60)
print("🔍 АНАЛИЗ КОЛОНКИ: hist_days_since_last")
print("="*60)
print(f"Количество -1:        {count_minus_one:,}")
print(f"Процент от всех строк: {percentage:.2f}%")
print()

# Дополнительная статистика
print("="*60)
print("📈 ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА")
print("="*60)

# Описательная статистика
print("\nОписание колонки:")
print(df['hist_days_since_last'].describe())

# Уникальные значения (если их немного)
unique_values = df['hist_days_since_last'].unique()
print(f"\nКоличество уникальных значений: {len(unique_values):,}")

# Топ-5 самых частых значений
print("\nТоп-5 самых частых значений:")
top_values = df['hist_days_since_last'].value_counts().head()
for value, count in top_values.items():
    pct = (count / len(df)) * 100
    print(f"  {value:>10.2f}: {count:>8,} раз ({pct:>5.2f}%)")

# Распределение
print("\n" + "="*60)
print("📊 РАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ")
print("="*60)

bins = [-2, -0.5, 1, 7, 30, 90, 365, df['hist_days_since_last'].max()]
labels = ['= -1', '0-1 день', '1-7 дней', '7-30 дней', '30-90 дней', '90-365 дней', '>365 дней']

df['days_category'] = pd.cut(df['hist_days_since_last'], bins=bins, labels=labels)
distribution = df['days_category'].value_counts().sort_index()

print("\nРаспределение по категориям:")
for category, count in distribution.items():
    pct = (count / len(df)) * 100
    print(f"  {str(category):>15}: {count:>8,} ({pct:>5.2f}%)")

# Значение -1 означает первое видео аккаунта
print("\n" + "="*60)
print("💡 ИНТЕРПРЕТАЦИЯ")
print("="*60)
print("""
hist_days_since_last = -1 означает:
  ✓ Это ПЕРВОЕ видео данного аккаунта
  ✓ У него нет предыдущего видео
  ✓ Поэтому нельзя вычислить разницу в днях

Другие значения:
  ✓ >= 0: Количество дней с момента предыдущего видео
""")

# Проверяем гипотезу: -1 должно совпадать с hist_window_size == 0
if 'hist_window_size' in df.columns:
    window_zero = (df['hist_window_size'] == 0).sum()
    match = window_zero == count_minus_one
    
    print(f"\n🔍 ПРОВЕРКА КОНСИСТЕНТНОСТИ:")
    print(f"  hist_days_since_last == -1:  {count_minus_one:,}")
    print(f"  hist_window_size == 0:        {window_zero:,}")
    print(f"  Совпадают? {'✅ ДА' if match else '❌ НЕТ'}")
    
    if not match:
        print(f"  ⚠️ ВНИМАНИЕ: Расхождение в {abs(window_zero - count_minus_one):,} строках")

print("\n✅ Анализ завершен!")
