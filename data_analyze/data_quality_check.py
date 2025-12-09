import pandas as pd
import numpy as np

print("📥 Загружаем dataset_v0_complete.parquet...")
df = pd.read_parquet('dataset_v0_complete.parquet')

print(f"📊 Общее количество строк: {len(df):,}")
print(f"📊 Общее количество колонок: {len(df.columns)}")
print()

# =====================================================
# 1. ОПРЕДЕЛЯЕМ ТИПЫ КОЛОНОК
# =====================================================
print("="*80)
print("🔍 КЛАССИФИКАЦИЯ КОЛОНОК")
print("="*80)

# Флаговые колонки (бинарные индикаторы) - не проверяем на отрицательные значения
flag_columns = [
    'is_new_account', 
    'st_is_verified', 
    'st_is_business'
]

# ID колонки - не проверяем на числовые аномалии
id_columns = [
    'account_id', 
    'video_id'
]

# Текстовые колонки - не проверяем на числовые аномалии
text_columns = [
    'video_text', 
    'raw_bio'
]

# Временные колонки
datetime_columns = [
    'upload_date'
]

# Числовые колонки для проверки (все остальные)
numeric_columns = [col for col in df.columns if col not in 
                   flag_columns + id_columns + text_columns + datetime_columns]

print(f"\n✓ Флаговые колонки ({len(flag_columns)}): {flag_columns}")
print(f"✓ ID колонки ({len(id_columns)}): {id_columns}")
print(f"✓ Текстовые колонки ({len(text_columns)}): {text_columns}")
print(f"✓ Временные колонки ({len(datetime_columns)}): {datetime_columns}")
print(f"✓ Числовые колонки для проверки ({len(numeric_columns)}): {numeric_columns[:5]}... (всего {len(numeric_columns)})")

# =====================================================
# 2. ПРОВЕРКА ДУБЛИКАТОВ video_id
# =====================================================
print("\n" + "="*80)
print("🔄 ПРОВЕРКА ДУБЛИКАТОВ video_id")
print("="*80)

duplicates = df['video_id'].duplicated().sum()
unique_count = df['video_id'].nunique()

print(f"Всего строк:           {len(df):,}")
print(f"Уникальных video_id:   {unique_count:,}")
print(f"Дубликаты:             {duplicates:,}")

if duplicates > 0:
    print(f"\n⚠️  НАЙДЕНЫ ДУБЛИКАТЫ! ({duplicates} строк)")
    # Показываем примеры дубликатов
    duplicate_ids = df[df['video_id'].duplicated(keep=False)]['video_id'].unique()[:5]
    print(f"\nПримеры дублирующихся video_id:")
    for vid in duplicate_ids:
        count = (df['video_id'] == vid).sum()
        print(f"  - {vid}: встречается {count} раз")
else:
    print("✅ Дубликатов не найдено!")

# =====================================================
# 3. ПРОВЕРКА NaN и None ВО ВСЕХ КОЛОНКАХ
# =====================================================
print("\n" + "="*80)
print("❓ ПРОВЕРКА NaN и None (ВСЕ КОЛОНКИ)")
print("="*80)

nan_stats = []
total_nans = 0

for col in df.columns:
    nan_count = df[col].isna().sum()
    none_count = df[col].isnull().sum()  # isnull ловит и NaN и None
    
    if nan_count > 0 or none_count > 0:
        pct = (nan_count / len(df)) * 100
        nan_stats.append({
            'Колонка': col,
            'NaN/None': nan_count,
            'Процент': f"{pct:.2f}%",
            'Тип': df[col].dtype
        })
        total_nans += nan_count

if nan_stats:
    df_nan = pd.DataFrame(nan_stats).sort_values('NaN/None', ascending=False)
    print(f"\n⚠️  НАЙДЕНЫ ПРОПУСКИ В {len(nan_stats)} КОЛОНКАХ:")
    print(df_nan.to_string(index=False))
    print(f"\n📊 Всего пропущенных значений: {total_nans:,}")
else:
    print("✅ Пропусков не найдено!")

# =====================================================
# 4. ПРОВЕРКА ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ (только числовые)
# =====================================================
print("\n" + "="*80)
print("➖ ПРОВЕРКА ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ (числовые колонки)")
print("="*80)

negative_stats = []
total_negatives = 0

# Специальные исключения: колонки, где -1 допустимо
special_exceptions = {
    'hist_days_since_last': -1.0  # -1 означает первое видео
}

for col in numeric_columns:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        if col in special_exceptions:
            # Для hist_days_since_last проверяем значения < -1
            negative_count = (df[col] < special_exceptions[col]).sum()
            exception_note = f"(допустимо {special_exceptions[col]})"
        else:
            # Для остальных проверяем все отрицательные
            negative_count = (df[col] < 0).sum()
            exception_note = ""
        
        if negative_count > 0:
            pct = (negative_count / len(df)) * 100
            min_val = df[col].min()
            negative_stats.append({
                'Колонка': col,
                'Отрицательных': negative_count,
                'Процент': f"{pct:.2f}%",
                'Минимум': f"{min_val:.2f}",
                'Примечание': exception_note
            })
            total_negatives += negative_count

if negative_stats:
    df_neg = pd.DataFrame(negative_stats).sort_values('Отрицательных', ascending=False)
    print(f"\n⚠️  НАЙДЕНЫ ОТРИЦАТЕЛЬНЫЕ ЗНАЧЕНИЯ В {len(negative_stats)} КОЛОНКАХ:")
    print(df_neg.to_string(index=False))
    print(f"\n📊 Всего отрицательных значений: {total_negatives:,}")
else:
    print("✅ Отрицательных значений не найдено!")

# =====================================================
# 5. ПРОВЕРКА СПЕЦИАЛЬНЫХ СЛУЧАЕВ
# =====================================================
print("\n" + "="*80)
print("🔍 ДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ")
print("="*80)

# 5.1 Проверка на inf/-inf
print("\n1. Проверка на бесконечности (inf/-inf):")
inf_found = False
for col in numeric_columns:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"   ⚠️  {col}: {inf_count} бесконечных значений")
            inf_found = True
if not inf_found:
    print("   ✅ Бесконечных значений не найдено")

# 5.2 Проверка пустых строк в текстовых полях
print("\n2. Проверка пустых строк в текстовых полях:")
for col in text_columns:
    if col in df.columns:
        empty_count = (df[col] == '').sum()
        if empty_count > 0:
            pct = (empty_count / len(df)) * 100
            print(f"   📝 {col}: {empty_count:,} пустых строк ({pct:.2f}%)")

# 5.3 Проверка нулевых значений в важных метриках
print("\n3. Проверка нулевых значений в ключевых метриках:")
key_metrics = ['views', 'likes', 'comments']
for col in key_metrics:
    if col in df.columns:
        zero_count = (df[col] == 0).sum()
        pct = (zero_count / len(df)) * 100
        print(f"   📊 {col}: {zero_count:,} нулевых значений ({pct:.2f}%)")

# 5.4 Проверка консистентности hist_days_since_last
print("\n4. Проверка консистентности hist_days_since_last = -1:")
if 'hist_days_since_last' in df.columns and 'hist_window_size' in df.columns:
    minus_one = (df['hist_days_since_last'] == -1).sum()
    window_zero = (df['hist_window_size'] == 0).sum()
    match = minus_one == window_zero
    print(f"   hist_days_since_last == -1: {minus_one:,}")
    print(f"   hist_window_size == 0:      {window_zero:,}")
    print(f"   Совпадают? {'✅ ДА' if match else '❌ НЕТ'}")

# =====================================================
# 6. ИТОГОВЫЙ ОТЧЕТ
# =====================================================
print("\n" + "="*80)
print("📋 ИТОГОВЫЙ ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ")
print("="*80)

issues_found = 0
if duplicates > 0:
    print(f"❌ Дубликаты video_id: {duplicates:,}")
    issues_found += 1
else:
    print(f"✅ Дубликаты video_id: не найдены")

if total_nans > 0:
    print(f"❌ Пропущенные значения (NaN/None): {total_nans:,} в {len(nan_stats)} колонках")
    issues_found += 1
else:
    print(f"✅ Пропущенные значения (NaN/None): не найдены")

if total_negatives > 0:
    print(f"❌ Недопустимые отрицательные значения: {total_negatives:,} в {len(negative_stats)} колонках")
    issues_found += 1
else:
    print(f"✅ Недопустимые отрицательные значения: не найдены")

print(f"\n{'='*80}")
if issues_found == 0:
    print("🎉 ДАТАСЕТ ЧИСТЫЙ! Все проверки пройдены успешно.")
else:
    print(f"⚠️  Обнаружено проблем: {issues_found}")
    print("Рекомендуется очистка данных перед обучением модели.")
print("="*80)
