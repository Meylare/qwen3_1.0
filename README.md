# Feature Extraction Library

Библиотека для извлечения статических признаков (static features) из профиля пользователя.

## `extract_static(user_profile_dict)`

Функция в `features_lib.py` преобразует сырые данные профиля в числовые признаки для ML.

### Признаки (Output)

| Ключ | Тип | Описание | Формула |
| :--- | :--- | :--- | :--- |
| `st_log_followers` | float | Логарифм подписчиков | `log1p(followers)` |
| `st_ratio_ff` | float | Отношение подписок | `followers / (following + 1)` |
| `st_is_verified` | int | Верификация | `1` или `0` |
| `st_is_business` | int | Бизнес-аккаунт | `1` или `0` |
| `st_bio_length` | int | Длина био | `len(biography)` |
| `st_hist_total_posts` | int | Всего постов | `media_count` |

### Пример

```python
from features_lib import extract_static

data = {"followers_count": 1500, "following_count": 300, "media_count": 42}
print(extract_static(data))
# {'st_log_followers': 7.3, 'st_ratio_ff': 4.98, ..., 'st_hist_total_posts': 42}
```
