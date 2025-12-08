import numpy as np
import logging

# Configure logger
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIG: Set to False to allow missing keys (they will default to 0/False/Empty)
STRICT_MODE = True

def extract_static(user_profile_dict: dict) -> dict:
    """
    Extracts static features from a user profile dictionary.
    
    If STRICT_MODE is True, raises KeyError for ANY missing field.
    If STRICT_MODE is False, uses defaults for missing fields.
    """
    # Define all keys we try to access. 
    # Note: 'is_business' logic checks multiple keys, handled below.
    required_keys = ['followers_count', 'following_count', 'media_count', 'is_verified', 'biography']
    
    if STRICT_MODE:
        missing_keys = [key for key in required_keys if key not in user_profile_dict]
        
        # Special check for business flag which can be under different names
        if 'is_business_account' not in user_profile_dict and 'is_business' not in user_profile_dict:
             missing_keys.append('is_business_account/is_business')

        if missing_keys:
            error_msg = f"STRICT MODE: Missing required keys in user_profile_dict: {missing_keys}"
            logger.error(error_msg)
            raise KeyError(error_msg)

    # 1. Логарифм подписчиков (st_log_followers)
    followers = user_profile_dict.get('followers_count', 0)
    st_log_followers = np.log1p(followers)

    # 2. Ratio (подписки/подписчики) (st_ratio_ff)
    following = user_profile_dict.get('following_count', 0)
    st_ratio_ff = followers / (following + 1)

    # 3. Flags (st_is_verified, st_is_business)
    st_is_verified = int(user_profile_dict.get('is_verified', False))
    
    is_business = user_profile_dict.get('is_business_account') or user_profile_dict.get('is_business') or False
    st_is_business = int(is_business)

    # 4. Длина Bio (st_bio_length)
    bio = user_profile_dict.get('biography', '')
    if bio is None:
        bio = ""
    st_bio_length = len(str(bio))

    # 5. Исторические посты (st_hist_total_posts)
    st_hist_total_posts = user_profile_dict.get('media_count', 0)

    # 6. One-Hot/Label encoding категории
    # REMOVED as per user request
    # category = user_profile_dict.get('category_name') or user_profile_dict.get('category') or 'Unknown'
    # st_category_label = abs(hash(category)) % 10000

    return {
        'st_log_followers': float(st_log_followers),
        'st_ratio_ff': float(st_ratio_ff),
        'st_is_verified': st_is_verified,
        'st_is_business': st_is_business,
        'st_bio_length': st_bio_length,
        'st_hist_total_posts': st_hist_total_posts
    }

