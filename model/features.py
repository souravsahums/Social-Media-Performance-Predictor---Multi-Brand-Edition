"""
Feature Engineering Module
Extracts features from Instagram post data for engagement prediction.
"""

import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from model.visual_features import (
    extract_visual_features_from_post,
    extract_visual_features_batch,
    VISUAL_FEATURE_NAMES,
    DEFAULT_VISUAL_FEATURES,
)


# ----------------------------------------------
# Text features
# ----------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

_CTA_KEYWORDS = [
    'link in bio', 'swipe', 'tap', 'click', 'comment', 'share', 'tag',
    'follow', 'send this', 'watch', 'check out', 'dm', 'subscribe',
]


def extract_text_features(caption: str) -> Dict[str, float]:
    if not caption:
        return {k: 0 for k in (
            'caption_length', 'word_count', 'hashtag_count', 'mention_count',
            'emoji_count', 'has_cta', 'has_question', 'line_count',
            'exclamation_count', 'avg_word_length',
        )}

    words = caption.split()
    return {
        'caption_length': len(caption),
        'word_count': len(words),
        'hashtag_count': len(re.findall(r'#\w+', caption)),
        'mention_count': len(re.findall(r'@\w+', caption)),
        'emoji_count': len(_EMOJI_RE.findall(caption)),
        'has_cta': int(any(kw in caption.lower() for kw in _CTA_KEYWORDS)),
        'has_question': int('?' in caption),
        'line_count': caption.count('\n') + 1,
        'exclamation_count': caption.count('!'),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
    }


# ----------------------------------------------
# Media / visual features
# ----------------------------------------------

_BRAND_KEYWORDS = ['sprite', 'coca-cola', 'coca cola', 'pepsi', 'red bull',
                   'redbull', 'thums up', 'thumsup', 'coke']


def extract_media_features(data: Dict) -> Dict[str, float]:
    metadata = data.get('metadata_content', {})
    media_list = data.get('media', [])
    media_name = metadata.get('media_name', 'post')

    summaries = [m.get('summary', '') for m in media_list if m.get('summary')]
    total_summary_len = sum(len(s) for s in summaries)
    has_brand_in_visual = int(any(
        kw in s.lower() for s in summaries for kw in _BRAND_KEYWORDS
    ))
    has_person_in_visual = int(any(
        kw in s.lower() for s in summaries
        for kw in ['woman', 'man', 'person', 'people', 'crowd', 'athlete', 'player']
    ))

    return {
        'duration': metadata.get('duration', 0),
        'is_reel': int(media_name == 'reel'),
        'is_post': int(media_name == 'post'),
        'is_album': int(media_name == 'album'),
        'has_video': int(any(m.get('type') == 'video' for m in media_list)),
        'has_thumbnail': int(any(m.get('type') == 'thumbnail' for m in media_list)),
        'media_count': len(media_list),
        'summary_length': total_summary_len,
        'has_brand_in_visual': has_brand_in_visual,
        'has_person_in_visual': has_person_in_visual,
    }


# ----------------------------------------------
# Temporal features
# ----------------------------------------------

def extract_temporal_features(created_at: str) -> Dict[str, float]:
    if not created_at:
        return {'hour': 12, 'day_of_week': 3, 'is_weekend': 0,
                'month': 6, 'is_morning': 0, 'is_evening': 0}
    try:
        dt = datetime.fromisoformat(created_at)
    except Exception:
        dt = datetime(2026, 1, 1, 12, 0, 0)

    h, dow = dt.hour, dt.weekday()
    return {
        'hour': h,
        'day_of_week': dow,
        'is_weekend': int(dow >= 5),
        'month': dt.month,
        'is_morning': int(6 <= h <= 10),
        'is_evening': int(17 <= h <= 21),
    }


# ----------------------------------------------
# Collaboration features
# ----------------------------------------------

def extract_collaboration_features(data: Dict) -> Dict[str, float]:
    mc = data.get('metadata_content', {})
    ps = data.get('profile_stats', {})
    collabs = mc.get('collaborators', [])
    author = ps.get('post_author_username', '')
    brand = ps.get('username', '')

    return {
        'is_collaborated': int(mc.get('is_collaborated_post', False)),
        'collaborator_count': len(collabs),
        'is_ugc': int(author != brand and author != ''),
    }


# ----------------------------------------------
# Brand one-hot
# ----------------------------------------------

BRANDS = ['cocacola_india', 'redbullindia', 'pepsiindia',
          'sprite_india', 'thumsupofficial']


def extract_brand_features(username: str) -> Dict[str, float]:
    feats = {f'brand_{b}': 0.0 for b in BRANDS}
    if username in BRANDS:
        feats[f'brand_{username}'] = 1.0
    return feats


# ----------------------------------------------
# Aggregate
# ----------------------------------------------

def extract_all_features(post: Dict, include_visual: bool = False) -> Dict[str, float]:
    """Extract the full feature vector from a single post dict.
    
    Args:
        post: Post data dict
        include_visual: If True, download thumbnail and extract image features.
                       Set to False for fast predictions without network calls.
    """
    data = post.get('data', post)
    mc = data.get('metadata_content', {})
    ps = data.get('profile_stats', {})
    followers = ps.get('followers', 0)

    features: Dict[str, float] = {}
    features.update(extract_text_features(mc.get('caption', '')))
    features.update(extract_media_features(data))
    features.update(extract_temporal_features(mc.get('created_at', '')))
    features.update(extract_collaboration_features(data))
    features.update(extract_brand_features(ps.get('username', '')))
    features['followers'] = followers
    features['log_followers'] = float(np.log1p(followers))

    # Visual features from thumbnail
    if include_visual:
        features.update(extract_visual_features_from_post(post))
    else:
        # Add defaults so feature vector is consistent size
        features.update(DEFAULT_VISUAL_FEATURES)

    return features


# ----------------------------------------------
# Target computation
# ----------------------------------------------

def compute_target(data: Dict) -> Dict[str, float]:
    eng = data.get('engagements', {})
    likes = eng.get('likes', 0)
    comments = eng.get('comments', 0)
    shares = eng.get('shares', 0)
    er = eng.get('engagement_rate', 0.0)
    return {
        'likes': likes,
        'views': eng.get('views', 0),
        'comments': comments,
        'shares': shares,
        'engagement_rate': er,
        'log_engagement_rate': float(np.log1p(er)),
        'total_interactions': likes + comments + shares,
    }


# ----------------------------------------------
# DataFrame builders
# ----------------------------------------------

FEATURE_COLUMNS = [
    'caption_length', 'word_count', 'hashtag_count', 'mention_count',
    'emoji_count', 'has_cta', 'has_question', 'line_count',
    'exclamation_count', 'avg_word_length',
    'duration', 'is_reel', 'is_post', 'is_album', 'has_video',
    'has_thumbnail', 'media_count', 'summary_length',
    'has_brand_in_visual', 'has_person_in_visual',
    'hour', 'day_of_week', 'is_weekend', 'month', 'is_morning', 'is_evening',
    'is_collaborated', 'collaborator_count', 'is_ugc',
    'brand_cocacola_india', 'brand_redbullindia', 'brand_pepsiindia',
    'brand_sprite_india', 'brand_thumsupofficial',
    'followers', 'log_followers',
] + VISUAL_FEATURE_NAMES


def build_dataframe(dataset: List[Dict],
                    visual_features: List[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature and target DataFrames from dataset.
    
    Args:
        dataset: List of post dicts
        visual_features: Pre-computed visual features (from batch extraction).
                        If None, defaults are used (zeros).
    """
    feats, tgts = [], []
    for i, post in enumerate(dataset):
        data = post.get('data', post)
        f = extract_all_features(post, include_visual=False)
        # Merge pre-computed visual features if provided
        if visual_features is not None and i < len(visual_features):
            f.update(visual_features[i])
        feats.append(f)
        tgts.append(compute_target(data))
    return pd.DataFrame(feats), pd.DataFrame(tgts)


# ----------------------------------------------
# Brand-relative performance labeling
# ----------------------------------------------

def compute_brand_stats(dataset: List[Dict]) -> Dict:
    brand_rates: Dict[str, List[float]] = {}
    for post in dataset:
        data = post.get('data', post)
        brand = data.get('profile_stats', {}).get('username', 'unknown')
        rate = data.get('engagements', {}).get('engagement_rate', 0)
        brand_rates.setdefault(brand, []).append(rate)

    stats = {}
    for brand, rates in brand_rates.items():
        a = np.array(rates)
        stats[brand] = {
            'mean': float(np.mean(a)), 'median': float(np.median(a)),
            'std': float(np.std(a)),
            'p25': float(np.percentile(a, 25)),
            'p75': float(np.percentile(a, 75)),
            'min': float(np.min(a)), 'max': float(np.max(a)),
            'count': len(rates),
        }
    return stats


def categorize_performance(er: float, brand: str, brand_stats: Dict) -> str:
    if brand not in brand_stats:
        return 'high' if er > 5 else ('medium' if er > 1 else 'low')
    s = brand_stats[brand]
    if er >= s['p75']:
        return 'high'
    elif er >= s['p25']:
        return 'medium'
    return 'low'
