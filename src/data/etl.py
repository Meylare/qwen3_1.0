import json
import os
import pandas as pd
from datetime import datetime

# --- КОНФИГУРАЦИЯ ФАЙЛОВ ---
# Укажите корректные пути к вашим файлам

# Исходные данные
VIDEOS_FILE = '100k.jsonl'     # Файл с видео
PROFILES_FILE = 'owners.jsonl'    # Файл с профилями (или jsonl_2.jsonl)

# Файлы с метриками (Viral Index), разделенные на Train и Val
TRAIN_PARQUET = 'train_stage2.parquet' 
VAL_PARQUET = 'val_stage2.parquet'

# Куда сохранять готовые датасеты
OUTPUT_TRAIN = 'dataset_train.jsonl'
OUTPUT_VAL = 'dataset_val.jsonl'

# Дни недели на русском
DAYS_RU = {
    'Monday': 'Понедельник', 'Tuesday': 'Вторник', 'Wednesday': 'Среда',
    'Thursday': 'Четверг', 'Friday': 'Пятница', 'Saturday': 'Суббота',
    'Sunday': 'Воскресенье'
}

def load_profiles_db(filepath):
    """
    Загружает профили в память.
    Returns: dict {username: full_profile_data}
    """
    print(f"📂 Загрузка профилей из {filepath}...")
    profiles = {}
    if not os.path.exists(filepath):
        print("⚠️ Файл профилей не найден! Данные профиля будут пустыми.")
        return profiles
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'username' in data:
                    profiles[data['username']] = data
            except:
                continue
    return profiles

def load_targets_from_parquet(filepath):
    """
    Читает Parquet файл и возвращает словарь метрик.
    Ключ: video_id (он же shortCode)
    Значение: {viral_index, baseline_views}
    """
    if not os.path.exists(filepath):
        print(f"⚠️ Файл {filepath} не найден. Пропускаем.")
        return {}

    print(f"📊 Чтение метрик из {filepath}...")
    # Читаем только нужные колонки для экономии памяти
    df = pd.read_parquet(filepath, columns=['video_id', 'viral_index', 'context_str'])
    
    targets = {}
    for _, row in df.iterrows():
        vid = str(row['video_id'])
        v_index = row['viral_index']
        ctx_str = row['context_str']
        
        # Достаем baseline из context_str
        baseline = 0.0
        try:
            if ctx_str:
                ctx_json = json.loads(ctx_str)
                baseline = float(ctx_json.get('expected_views', 0.0))
        except:
            baseline = 0.0
            
        targets[vid] = {
            'viral_index': v_index,
            'baseline_views': baseline
        }
    
    print(f"   -> Загружено {len(targets)} записей.")
    return targets

def format_timestamp_ru(ts_str):
    """Преобразует ISO дату в формат: Пятница, 18:00"""
    try:
        if not ts_str: return "Неизвестно"
        clean_ts = ts_str.split('.')[0].replace('Z', '')
        dt = datetime.fromisoformat(clean_ts)
        
        day_eng = dt.strftime('%A')
        day_ru = DAYS_RU.get(day_eng, day_eng)
        time_str = dt.strftime('%H:%M')
        
        return f"{day_ru}, {time_str}"
    except:
        return str(ts_str)

def process_etl():
    # 1. Загружаем справочники
    profiles_db = load_profiles_db(PROFILES_FILE)
    train_targets_db = load_targets_from_parquet(TRAIN_PARQUET)
    val_targets_db = load_targets_from_parquet(VAL_PARQUET)
    
    print("-" * 50)
    print("🚀 Начинаем обработку видео и генерацию JSONL...")
    
    count_train = 0
    count_val = 0
    count_skipped = 0
    
    # Множество для защиты от дубликатов
    processed_ids = set()
    
    with open(VIDEOS_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f_train, \
         open(OUTPUT_VAL, 'w', encoding='utf-8') as f_val:
        
        for line in f_in:
            try:
                vid = json.loads(line)
            except:
                continue
                
            short_code = vid.get('shortCode')
            if not short_code:
                continue

            # --- ПРОВЕРКА НА ДУБЛИКАТЫ ---
            if short_code in processed_ids:
                # Мы уже обработали это видео ранее -> пропускаем
                continue
            
            # Добавляем ID в список обработанных
            processed_ids.add(short_code)

            # --- ОПРЕДЕЛЕНИЕ: КУДА ПИСАТЬ? ---
            target_info = None
            writer = None
            
            if short_code in train_targets_db:
                target_info = train_targets_db[short_code]
                writer = f_train
                count_train += 1
            elif short_code in val_targets_db:
                target_info = val_targets_db[short_code]
                writer = f_val
                count_val += 1
            else:
                # Видео нет ни в трейне, ни в валидации
                count_skipped += 1
                continue

            # --- СБОР ДАННЫХ ДЛЯ PROMPT ---
            
            username = vid.get('ownerUsername', '')
            prof = profiles_db.get(username, {})
            
            # 1. Никнейм
            nickname = username
            
            # 2. Био
            bio = prof.get('biography', '').replace('\n', ' ').strip()
            if not bio: bio = "Нет описания"
            
            # 3. Статус
            is_verified = prof.get('verified', False)
            ver_status = "Verified" if is_verified else "Not verified"
            
            # 4. Описание
            caption = vid.get('caption', '').replace('\n', ' ').strip()
            
            # 5. Отметки
            tags = vid.get('hashtags', []) + vid.get('mentions', [])
            tags_str = ", ".join(tags) if tags else "Нет"
            
            # 6. Локация
            loc_data = vid.get('location')
            location = loc_data.get('name', 'Не указана') if loc_data else 'Не указана'
            
            # 7. Музыка
            m_info = vid.get('musicInfo', {})
            music_str = f"{m_info.get('artist_name','')} - {m_info.get('song_name','')}"
            if music_str.strip() == "-": music_str = "Оригинальный звук"
            
            # 8. Таймстамп
            ts_str = format_timestamp_ru(vid.get('timestamp'))
            
            # 9. Длительность (Спец. логика + защита от None)
            raw_duration = vid.get('videoDuration')
            
            if raw_duration is None:
                duration = 10.0
            else:
                try:
                    d_val = float(raw_duration)
                    if d_val >= 10.0:
                        duration = 10.0
                    else:
                        duration = round(d_val, 1)
                except (ValueError, TypeError):
                    duration = 10.0

            # --- ФОРМИРОВАНИЕ SYSTEM PROMPT ---
            system_prompt = (
                f"Никнейм: {nickname}. "
                f"Био: {bio}. "
                f"Статус веревекации: {ver_status}. "
                f"Описание: {caption}. "
                f"Отметки: {tags_str}. "
                f"Локация: {location}. "
                f"Инфо по музыке: {music_str}. "
                f"Таймстамп: {ts_str}. "
                f"Длительность видео: {duration} сек."
            )

            # --- Target Values ---
            # Безопасное получение просмотров
            views_val = vid.get('videoPlayCount')
            real_views = float(views_val) if views_val is not None else 0.0

            # --- СБОРКА ОБЪЕКТА ---
            output_obj = {
                "id": short_code,
                "video_path": f"/mnt/disks/fast_data/videos/{short_code}.mp4",
                
                "system_prompt": system_prompt,
                "aux_caption": caption,
                
                "targets": {
                    "viral_index": target_info['viral_index'],
                    "baseline_views": target_info['baseline_views'],
                    "real_views": real_views
                }
            }
            
            writer.write(json.dumps(output_obj, ensure_ascii=False) + '\n')

    print("-" * 50)
    print("✅ Готово.")
    print(f"Записано в TRAIN: {count_train} -> {OUTPUT_TRAIN}")
    print(f"Записано в VAL:   {count_val} -> {OUTPUT_VAL}")
    print(f"Пропущено (нет в parquet или дубликаты): {count_skipped}")

if __name__ == "__main__":
    process_etl()