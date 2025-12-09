#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для анализа Parquet датасета
Автор: Claude
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def print_separator(title=""):
    """Печатает разделитель с заголовком"""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)
    print()

def analyze_dataset(file_path):
    """Основная функция анализа датасета"""
    
    print("🔍 Начинаем анализ датасета...")
    print(f"📁 Файл: {file_path}")
    
    try:
        # Загружаем датасет
        print("\n⏳ Загрузка данных...")
        df = pd.read_parquet(file_path)
        print("✅ Данные успешно загружены!")
        
        # 1. БАЗОВАЯ ИНФОРМАЦИЯ
        print_separator("📊 БАЗОВАЯ ИНФОРМАЦИЯ")
        print(f"Количество строк: {df.shape[0]:,}")
        print(f"Количество колонок: {df.shape[1]:,}")
        print(f"Общее количество ячеек: {df.shape[0] * df.shape[1]:,}")
        
        # Размер в памяти
        memory_usage = df.memory_usage(deep=True).sum()
        memory_mb = memory_usage / (1024 * 1024)
        print(f"Использование памяти: {memory_mb:.2f} MB")
        
        # 2. ИНФОРМАЦИЯ О КОЛОНКАХ
        print_separator("📋 КОЛОНКИ И ТИПЫ ДАННЫХ")
        print(f"Список колонок ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"  {i:2d}. {col:30s} | Тип: {str(dtype):15s} | Пропуски: {null_count:6d} ({null_pct:5.2f}%)")
        
        # 3. ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О ТИПАХ
        print_separator("🔢 ТИПЫ ДАННЫХ")
        print(df.dtypes)
        
        # 4. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ
        print_separator("❓ ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Колонка': missing_data.index,
            'Пропусков': missing_data.values,
            'Процент': missing_pct.values
        })
        missing_df = missing_df[missing_df['Пропусков'] > 0].sort_values('Пропусков', ascending=False)
        
        if len(missing_df) > 0:
            print("Колонки с пропущенными значениями:")
            print(missing_df.to_string(index=False))
        else:
            print("✅ Пропущенных значений не обнаружено!")
        
        # 5. СТАТИСТИКА ДЛЯ ЧИСЛОВЫХ КОЛОНОК
        print_separator("📈 СТАТИСТИКА ЧИСЛОВЫХ КОЛОНОК")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"Найдено {len(numeric_cols)} числовых колонок:")
            print(df[numeric_cols].describe().to_string())
        else:
            print("Числовые колонки не найдены")
        
        # 6. КАТЕГОРИАЛЬНЫЕ КОЛОНКИ
        print_separator("🏷️  КАТЕГОРИАЛЬНЫЕ КОЛОНКИ")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"Найдено {len(categorical_cols)} категориальных колонок:")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                top_value = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else "N/A"
                top_count = df[col].value_counts().values[0] if len(df[col].value_counts()) > 0 else 0
                print(f"\n  {col}:")
                print(f"    Уникальных значений: {unique_count:,}")
                print(f"    Самое частое: '{top_value}' (встречается {top_count:,} раз)")
                
                # Показываем топ-5 значений, если их не слишком много
                if unique_count <= 20:
                    print(f"    Распределение всех значений:")
                    value_counts = df[col].value_counts()
                    for val, count in value_counts.items():
                        pct = (count / len(df)) * 100
                        print(f"      - {val}: {count:,} ({pct:.2f}%)")
                else:
                    print(f"    Топ-5 значений:")
                    value_counts = df[col].value_counts().head(5)
                    for val, count in value_counts.items():
                        pct = (count / len(df)) * 100
                        print(f"      - {val}: {count:,} ({pct:.2f}%)")
        else:
            print("Категориальные колонки не найдены")
        
        # 7. ПЕРВЫЕ И ПОСЛЕДНИЕ СТРОКИ
        print_separator("👀 ПЕРВЫЕ 5 СТРОК")
        print(df.head().to_string())
        
        print_separator("👀 ПОСЛЕДНИЕ 5 СТРОК")
        print(df.tail().to_string())
        
        # 8. ДУБЛИКАТЫ
        print_separator("🔄 ДУБЛИКАТЫ")
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            print(f"⚠️  Найдено {duplicate_count:,} дублирующихся строк ({(duplicate_count/len(df)*100):.2f}%)")
        else:
            print("✅ Дублирующихся строк не найдено")
        
        # 9. КОРРЕЛЯЦИЯ (только для числовых колонок)
        if len(numeric_cols) > 1:
            print_separator("🔗 КОРРЕЛЯЦИЯ ЧИСЛОВЫХ КОЛОНОК")
            correlation = df[numeric_cols].corr()
            print(correlation.to_string())
        
        # 10. ИТОГОВАЯ ИНФОРМАЦИЯ
        print_separator("✅ АНАЛИЗ ЗАВЕРШЕН")
        print(f"Датасет успешно проанализирован!")
        print(f"Строк: {df.shape[0]:,}")
        print(f"Колонок: {df.shape[1]:,}")
        print(f"Память: {memory_mb:.2f} MB")
        
        # Сохраняем подробный отчет в текстовый файл
        report_path = file_path.replace('.parquet', '_analysis_report.txt')
        print(f"\n💾 Полный отчет сохранен в: {report_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"\n❌ ОШИБКА: Файл '{file_path}' не найден!")
        print("Пожалуйста, проверьте путь к файлу.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ОШИБКА при анализе: {str(e)}")
        print(f"Тип ошибки: {type(e).__name__}")
        sys.exit(1)

if __name__ == "__main__":
    # Путь к файлу
    file_path = "dataset_v0_complete.parquet"
    
    # Проверяем существование файла
    if not Path(file_path).exists():
        print(f"❌ Файл '{file_path}' не найден в текущей директории!")
        print(f"📂 Текущая директория: {Path.cwd()}")
        sys.exit(1)
    
    # Запускаем анализ
    df = analyze_dataset(file_path)
    
    print("\n" + "=" * 80)
    print("Готово! Датасет загружен в переменную 'df'")
    print("Вы можете продолжить работу с ним в интерактивном режиме Python")
    print("=" * 80 + "\n")
