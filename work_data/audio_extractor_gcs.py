import pandas as pd
import subprocess
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

# --- КОНФИГУРАЦИЯ ---
INPUT_DATASET = "dataset_v0_complete.parquet"
GCS_KEY_PATH = "gcs_key2.json"
SILENT_LIST_FILE = "silent_videos.txt" # Чтобы не мучать видео без звука по кругу

# Бакет и папки
BUCKET_NAME = "slon_bucket2"
INPUT_PREFIX = "100k_videos/"                      # Папка с видео (если в корне, то "")

OUTPUT_PREFIX = "audio_wav_16kHz/"   # Папка для аудио

TEMP_DIR = Path("temp_processing")
NUM_WORKERS = 11

def get_existing_wav_ids(client, bucket_name, prefix):
    """
    Скачивает список уже готовых WAV файлов из бакета за один раз.
    Возвращает set (множество) с ID видео.
    """
    print("🔎 Сканируем GCS на наличие готовых аудио... (Это может занять пару секунд)")
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    existing_ids = set()
    for blob in blobs:
        if blob.name.endswith(".wav"):
            # blob.name = "audio_dataset_16k/12345.wav" -> "12345"
            vid_id = blob.name.split('/')[-1].replace('.wav', '')
            existing_ids.add(vid_id)
            
    return existing_ids

def process_video_cycle(video_id):
    """
    Скачивает MP4 -> Конвертирует -> Загружает WAV
    """
    blob_in_name = f"{INPUT_PREFIX}{video_id}.mp4"
    blob_out_name = f"{OUTPUT_PREFIX}{video_id}.wav"
    
    local_mp4_path = TEMP_DIR / f"{video_id}.mp4"
    local_wav_path = TEMP_DIR / f"{video_id}.wav"

    status = "unknown"

    try:
        # Инициализация клиента внутри потока
        client = storage.Client.from_service_account_json(GCS_KEY_PATH)
        bucket = client.bucket(BUCKET_NAME)
        
        # Скачиваем видео
        blob_in = bucket.blob(blob_in_name)
        if not blob_in.exists():
            return video_id, "missing_video"
        
        blob_in.download_to_filename(local_mp4_path)

        # Конвертируем
        command = [
            "ffmpeg", "-i", str(local_mp4_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", str(local_wav_path)
        ]
        # capture_output=True глушит логи ffmpeg
        subprocess.run(command, check=True, capture_output=True)

        # Загружаем обратно
        blob_out = bucket.blob(blob_out_name)
        blob_out.upload_from_filename(local_wav_path)
        status = "success"

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='ignore')
        if "Output file does not contain any stream" in err_msg:
            status = "no_audio_stream"
        else:
            status = "ffmpeg_error"
            
    except Exception as e:
        status = "gcs_error"
    
    finally:
        # Чистим диск
        if local_mp4_path.exists(): os.remove(local_mp4_path)
        if local_wav_path.exists(): os.remove(local_wav_path)

    return video_id, status

def run_extraction():
    print("🚀 Запуск GCS Audio Extractor (Smart Skip)...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Загружаем список всех видео из датасета
    if not os.path.exists(INPUT_DATASET):
        print(f"❌ Нет файла {INPUT_DATASET}")
        return
    df = pd.read_parquet(INPUT_DATASET)
    all_video_ids = set(df['video_id'].astype(str)) # Используем set для скорости
    print(f"📄 Всего видео в датасете: {len(all_video_ids)}")

    # 2. Получаем список УЖЕ ГОТОВЫХ аудио из облака
    client = storage.Client.from_service_account_json(GCS_KEY_PATH)
    existing_ids = get_existing_wav_ids(client, BUCKET_NAME, OUTPUT_PREFIX)
    print(f"☁️  Уже есть в облаке: {len(existing_ids)}")

    # 3. Получаем список ИЗВЕСТНЫХ ПУСТЫХ (без звука)
    silent_ids = set()
    if os.path.exists(SILENT_LIST_FILE):
        with open(SILENT_LIST_FILE, 'r') as f:
            silent_ids = set(line.strip() for line in f if line.strip())
        print(f"🔇 Известные 'немые' видео: {len(silent_ids)}")

    # 4. Вычисляем, что осталось сделать
    # Нужно сделать = Все - (Уже готовые + Известные немые)
    ids_to_process = list(all_video_ids - existing_ids - silent_ids)
    
    print("-" * 40)
    print(f"🔥 ОСТАЛОСЬ ОБРАБОТАТЬ: {len(ids_to_process)}")
    print("-" * 40)

    if not ids_to_process:
        print("✅ Все файлы уже обработаны! Отдыхай.")
        return

    # 5. Запуск обработки
    stats = {"success": 0, "missing_video": 0, "no_audio_stream": 0, "ffmpeg_error": 0, "gcs_error": 0}
    new_silent_ids = []

    print(f"🚀 Запускаем {NUM_WORKERS} потоков...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_video_cycle, ids_to_process), total=len(ids_to_process)))

    # 6. Сбор статистики
    for vid, status in results:
        stats[status] += 1
        if status == "no_audio_stream":
            new_silent_ids.append(vid)
    
    try: os.rmdir(TEMP_DIR)
    except: pass

    # Дописываем новые немые видео в файл
    if new_silent_ids:
        with open(SILENT_LIST_FILE, "a") as f:
            for vid in new_silent_ids:
                f.write(f"{vid}\n")
        print(f"📝 Добавлено {len(new_silent_ids)} новых немых видео в {SILENT_LIST_FILE}")

    print("\n✅ СЕССИЯ ЗАВЕРШЕНА!")
    print(f"   🎉 Успешно загружено: {stats['success']}")
    print(f"   🔇 Найдено без звука: {stats['no_audio_stream']}")
    print(f"   ❌ Нет видео исходника: {stats['missing_video']}")
    print(f"   ⚠️ Ошибки: {stats['ffmpeg_error'] + stats['gcs_error']}")

if __name__ == "__main__":
    run_extraction()