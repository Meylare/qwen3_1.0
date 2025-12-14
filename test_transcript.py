#!/usr/bin/env python3
import json

# Тестовый JSON в упрощенном виде (без специальных символов)
test_json = '''[{"start": 0.0, "end": 4.8, "text": "Hello world"}, {"start": 5.92, "end": 10.02, "text": "This is a test"}]'''

print('Test JSON format:')
print(test_json)
print()

# Тестируем обработку как в коде
def process_transcript_text(transcript_text: str) -> str:
    if not isinstance(transcript_text, str) or not transcript_text.strip():
        return ""

    try:
        transcript_data = json.loads(transcript_text)

        if isinstance(transcript_data, list):
            processed_segments = []
            for segment in transcript_data:
                if isinstance(segment, dict) and 'text' in segment:
                    start_time = segment.get('start', 0)
                    text = segment['text'].strip()
                    if text:
                        minutes = int(start_time // 60)
                        seconds = int(start_time % 60)
                        timestamp = f"[{minutes}:{seconds:02d}]"
                        processed_segments.append(f"{timestamp} {text}")

            if processed_segments:
                return " ".join(processed_segments)

    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return transcript_text

result = process_transcript_text(test_json)
print('Processing result:')
print(result)
print()
print('Result length:', len(result))
print('Number of segments:', result.count('['))

# Проверим, что ваш формат работает
user_json = '''[{"start": 0.0, "end": 4.8, "text": "Text 1"}, {"start": 5.92, "end": 10.02, "text": "Text 2"}]'''
user_result = process_transcript_text(user_json)
print()
print('Your format result:')
print(user_result)