"""
Модуль для извлечения статических признаков из профилей Instagram/TikTok.
"""
import numpy as np
import math


def extract_static(profile_dict):
    """
    Извлекает статические признаки из словаря профиля.

    Args:
        profile_dict: словарь с полями профиля

    Returns:
        dict: словарь с извлеченными признаками
    """
    features = {}

    # 1. Логарифм подписчиков (защита от нуля)
    followers = profile_dict.get('followers_count', 0)
    if followers > 0:
        features['st_log_followers'] = math.log1p(followers)
    else:
        features['st_log_followers'] = 0.0

    # 2. Соотношение following/followers (защита от деления на ноль)
    following = profile_dict.get('following_count', 0)
    if followers > 0:
        features['st_ratio_ff'] = following / followers
    else:
        features['st_ratio_ff'] = 0.0

    # 3. Статус верификации
    features['st_is_verified'] = int(profile_dict.get('is_verified', False))

    # 4. Бизнес-аккаунт
    features['st_is_business'] = int(profile_dict.get('is_business_account', False))

    # 5. Длина биографии
    bio = str(profile_dict.get('biography', ''))
    features['st_bio_length'] = len(bio.strip())

    return features