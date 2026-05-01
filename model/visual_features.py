"""
Visual Feature Extraction from S3-hosted media thumbnails.
Downloads thumbnail images and extracts color/composition features using Pillow.
Handles broken links gracefully with fallback defaults.
"""

import io
import logging
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    from PIL import Image, ImageStat, ImageFilter
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default features returned when image cannot be loaded
DEFAULT_VISUAL_FEATURES = {
    'img_width': 0,
    'img_height': 0,
    'img_aspect_ratio': 0.0,
    'img_brightness': 0.0,
    'img_contrast': 0.0,
    'img_saturation': 0.0,
    'img_red_mean': 0.0,
    'img_green_mean': 0.0,
    'img_blue_mean': 0.0,
    'img_color_variance': 0.0,
    'img_edge_density': 0.0,
    'img_warmth': 0.0,
    'img_is_bright': 0,
    'img_is_high_contrast': 0,
    'img_has_dominant_color': 0,
}

VISUAL_FEATURE_NAMES = list(DEFAULT_VISUAL_FEATURES.keys())

# Timeout for downloading images (seconds)
DOWNLOAD_TIMEOUT = 10


def _download_image(url: str) -> Optional[Image.Image]:
    """Download an image from URL with timeout and error handling."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SMPP/1.0'})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as response:
            data = response.read()
            return Image.open(io.BytesIO(data))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, Exception) as e:
        logger.debug(f"Failed to download image from {url}: {e}")
        return None


def extract_image_features(img: Image.Image) -> Dict[str, float]:
    """Extract visual features from a PIL Image."""
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size
    aspect_ratio = width / max(height, 1)

    # Basic color statistics
    stat = ImageStat.Stat(img)
    r_mean, g_mean, b_mean = stat.mean
    r_std, g_std, b_std = stat.stddev

    # Brightness (perceived luminance)
    brightness = (0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean) / 255.0

    # Contrast (average standard deviation across channels)
    contrast = np.mean([r_std, g_std, b_std]) / 128.0

    # Saturation from HSV
    hsv_img = img.convert('HSV')
    hsv_stat = ImageStat.Stat(hsv_img)
    saturation = hsv_stat.mean[1] / 255.0  # S channel mean

    # Color variance (how diverse the colors are)
    color_variance = float(np.std([r_mean, g_mean, b_mean]) / 128.0)

    # Edge density (proxy for visual complexity)
    gray = img.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_density = edge_stat.mean[0] / 255.0

    # Warmth (red/yellow vs blue dominance)
    warmth = ((r_mean + g_mean * 0.5) - b_mean) / 255.0

    # Binary indicators
    is_bright = int(brightness > 0.55)
    is_high_contrast = int(contrast > 0.35)
    has_dominant_color = int(color_variance > 0.15)

    return {
        'img_width': width,
        'img_height': height,
        'img_aspect_ratio': round(aspect_ratio, 4),
        'img_brightness': round(brightness, 4),
        'img_contrast': round(contrast, 4),
        'img_saturation': round(saturation, 4),
        'img_red_mean': round(r_mean / 255.0, 4),
        'img_green_mean': round(g_mean / 255.0, 4),
        'img_blue_mean': round(b_mean / 255.0, 4),
        'img_color_variance': round(color_variance, 4),
        'img_edge_density': round(edge_density, 4),
        'img_warmth': round(warmth, 4),
        'img_is_bright': is_bright,
        'img_is_high_contrast': is_high_contrast,
        'img_has_dominant_color': has_dominant_color,
    }


def extract_visual_features_from_url(url: str) -> Dict[str, float]:
    """Download image from URL and extract visual features. Returns defaults on failure."""
    if not PILLOW_AVAILABLE:
        logger.warning("Pillow not available, returning default visual features")
        return DEFAULT_VISUAL_FEATURES.copy()

    if not url:
        return DEFAULT_VISUAL_FEATURES.copy()

    img = _download_image(url)
    if img is None:
        return DEFAULT_VISUAL_FEATURES.copy()

    try:
        return extract_image_features(img)
    except Exception as e:
        logger.debug(f"Feature extraction failed for {url}: {e}")
        return DEFAULT_VISUAL_FEATURES.copy()


def extract_visual_features_from_post(post: Dict) -> Dict[str, float]:
    """Extract visual features from a post's thumbnail URL."""
    data = post.get('data', post)
    media_list = data.get('media', [])

    # Find thumbnail URL (prefer thumbnail over video frame)
    thumbnail_url = None
    for m in media_list:
        if m.get('type') == 'thumbnail' and m.get('url'):
            thumbnail_url = m['url']
            break

    # Fallback: use any media URL
    if not thumbnail_url:
        for m in media_list:
            if m.get('url'):
                thumbnail_url = m['url']
                break

    return extract_visual_features_from_url(thumbnail_url or '')


def extract_visual_features_batch(
    posts: List[Dict], max_workers: int = 8,
    progress_callback=None
) -> List[Dict[str, float]]:
    """
    Extract visual features for a batch of posts using parallel downloads.
    Returns list of feature dicts in same order as input posts.
    """
    if not PILLOW_AVAILABLE:
        return [DEFAULT_VISUAL_FEATURES.copy() for _ in posts]

    # Collect all thumbnail URLs
    urls = []
    for post in posts:
        data = post.get('data', post)
        media_list = data.get('media', [])
        thumb_url = None
        for m in media_list:
            if m.get('type') == 'thumbnail' and m.get('url'):
                thumb_url = m['url']
                break
        if not thumb_url:
            for m in media_list:
                if m.get('url'):
                    thumb_url = m['url']
                    break
        urls.append(thumb_url or '')

    results = [None] * len(urls)
    broken_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(extract_visual_features_from_url, url): idx
            for idx, url in enumerate(urls)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = DEFAULT_VISUAL_FEATURES.copy()
                broken_count += 1
            completed += 1
            if progress_callback and completed % 50 == 0:
                progress_callback(completed, len(urls))

    if broken_count > 0:
        logger.info(f"Visual features: {broken_count}/{len(urls)} images failed to load")

    # Replace any None with defaults (shouldn't happen but safety)
    for i in range(len(results)):
        if results[i] is None:
            results[i] = DEFAULT_VISUAL_FEATURES.copy()

    return results
