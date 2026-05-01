"""
Data loader and preprocessing utilities.
Loads the assignment dataset and performs initial cleaning.
"""

import json
import os
from typing import Dict, List, Optional


DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'assignment-dataset.json')


def load_dataset(path: Optional[str] = None) -> List[Dict]:
    """Load the Instagram dataset from JSON."""
    path = path or DATASET_PATH
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_post(post: Dict) -> Dict:
    """Normalize a single post, filling missing fields with defaults."""
    data = post.get('data', {})

    # Ensure all expected top-level keys exist
    data.setdefault('metadata_content', {})
    data.setdefault('profile_stats', {})
    data.setdefault('engagements', {})
    data.setdefault('media', [])
    data.setdefault('url', '')
    data.setdefault('comments_collected', 0)

    mc = data['metadata_content']
    mc.setdefault('caption', '')
    mc.setdefault('created_at', '')
    mc.setdefault('duration', 0)
    mc.setdefault('media_type', 1)
    mc.setdefault('media_name', 'post')
    mc.setdefault('is_collaborated_post', False)
    mc.setdefault('collaborators', [])

    ps = data['profile_stats']
    ps.setdefault('followers', 0)
    ps.setdefault('username', 'unknown')
    ps.setdefault('post_author_username', ps['username'])

    eng = data['engagements']
    eng.setdefault('likes', 0)
    eng.setdefault('views', 0)
    eng.setdefault('comments', 0)
    eng.setdefault('shares', 0)
    eng.setdefault('engagement_rate', 0.0)

    return post


def load_and_clean(path: Optional[str] = None) -> List[Dict]:
    """Load dataset and clean all posts."""
    dataset = load_dataset(path)
    return [clean_post(p) for p in dataset]


def get_brand_posts(dataset: List[Dict], brand: str) -> List[Dict]:
    """Filter posts by brand username."""
    return [
        p for p in dataset
        if p.get('data', {}).get('profile_stats', {}).get('username', '') == brand
    ]


BRANDS = [
    'cocacola_india',
    'redbullindia',
    'pepsiindia',
    'sprite_india',
    'thumsupofficial',
]
