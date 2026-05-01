"""
SLM Predictor — Structured reasoning predictor integrated into the model pipeline.
Uses domain-knowledge-driven rule scoring as an interpretable fallback/supplement
to the neural network.  Can also be used standalone when the NN model is unavailable.
"""

import json
import os
import numpy as np
from typing import Dict, List

from model.features import extract_all_features, FEATURE_COLUMNS


class SLMPredictor:
    """
    Structured Language Model predictor — a rule-based scoring system that
    reasons about social media post performance using domain knowledge.

    Acts as:
      1. A fallback when NN confidence is low
      2. An interpretable 'second opinion' for the prediction
      3. A standalone predictor when the NN is unavailable
    """

    INV_LABEL = {0: 'low', 1: 'medium', 2: 'high'}

    def __init__(self, brand_stats: Dict = None):
        self.brand_stats = brand_stats or {}
        self.reasoning_log: List[Dict] = []

    def load_brand_stats(self, path: str):
        with open(path) as f:
            self.brand_stats = json.load(f)

    def score_post(self, features: Dict, brand: str = '') -> Dict:
        """
        Score a post and return a structured reasoning trace.
        Returns dict with prediction, confidence, score, and reasoning chain.
        """
        reasons = []
        score = 50.0

        # ----- Content Format -----
        if features.get('is_reel', 0):
            score += 12
            reasons.append(("Reel format", +12, "Reels get 3-5x more reach than static posts"))
        elif features.get('is_album', 0):
            score += 3
            reasons.append(("Album/Carousel", +3, "Carousels encourage swipe engagement"))
        else:
            score -= 5
            reasons.append(("Static post", -5, "Static posts get lowest organic reach"))

        # ----- Duration -----
        dur = features.get('duration', 0)
        if dur > 0:
            if 15 <= dur <= 30:
                score += 8
                reasons.append(("Optimal duration (15-30s)", +8, "Short reels have highest completion rate"))
            elif 30 < dur <= 60:
                score += 3
                reasons.append(("Medium duration (30-60s)", +3, "Acceptable length"))
            elif dur > 60:
                score -= 5
                reasons.append(("Long video (>60s)", -5, "Viewer retention drops significantly"))

        # ----- Collaboration -----
        if features.get('is_collaborated', 0):
            collab_count = features.get('collaborator_count', 1)
            boost = 10 + min(collab_count * 3, 9)
            score += boost
            reasons.append(("Collaboration", +boost, f"Collab with {collab_count} creator(s) expands reach"))
        if features.get('is_ugc', 0):
            score += 5
            reasons.append(("User-generated content", +5, "UGC feels authentic, drives engagement"))

        # ----- Caption Quality -----
        wc = features.get('word_count', 0)
        if 10 <= wc <= 50:
            score += 5
            reasons.append(("Good caption length", +5, f"{wc} words — concise and readable"))
        elif wc > 100:
            score -= 3
            reasons.append(("Very long caption", -3, f"{wc} words — may lose attention"))
        elif wc == 0:
            score -= 2
            reasons.append(("No caption", -2, "Missing caption reduces discoverability"))

        if features.get('has_cta', 0):
            score += 6
            reasons.append(("CTA present", +6, "Call-to-action drives comments/shares"))
        if features.get('has_question', 0):
            score += 4
            reasons.append(("Question in caption", +4, "Questions boost comment engagement"))
        if features.get('emoji_count', 0) >= 1:
            score += 2
            reasons.append(("Emojis used", +2, "Emojis increase visual appeal"))

        hc = features.get('hashtag_count', 0)
        if 3 <= hc <= 10:
            score += 3
            reasons.append(("Good hashtag count", +3, f"{hc} hashtags — optimal discoverability"))
        elif hc > 15:
            score -= 2
            reasons.append(("Excessive hashtags", -2, f"{hc} hashtags — may look spammy"))

        # ----- Visual Signals -----
        if features.get('has_person_in_visual', 0):
            score += 5
            reasons.append(("Person in visual", +5, "Faces/people increase engagement"))
        if features.get('has_brand_in_visual', 0):
            score += 3
            reasons.append(("Brand visible", +3, "Brand visibility reinforces awareness"))

        # ----- Timing -----
        if features.get('is_evening', 0):
            score += 5
            reasons.append(("Evening post (5-9 PM)", +5, "Peak scrolling hours in India"))
        elif features.get('is_morning', 0):
            score += 1
            reasons.append(("Morning post (6-10 AM)", +1, "Early scrollers — moderate reach"))
        if features.get('is_weekend', 0):
            score += 3
            reasons.append(("Weekend post", +3, "Higher casual engagement on weekends"))

        # ----- Follower context -----
        followers = features.get('followers', 0)
        if followers > 500000:
            score -= 3
            reasons.append(("Large audience", -3, "ER naturally lower for large accounts"))
        elif followers < 50000:
            score += 3
            reasons.append(("Smaller audience", +3, "Smaller accounts often have higher ER"))

        # Map score → prediction
        if score >= 65:
            pred_idx = 2
        elif score >= 45:
            pred_idx = 1
        else:
            pred_idx = 0

        # Compute pseudo-confidence from score distance to boundaries
        if pred_idx == 2:
            confidence = min(0.95, 0.5 + (score - 65) / 70)
        elif pred_idx == 1:
            confidence = min(0.85, 0.4 + min(score - 45, 65 - score) / 40)
        else:
            confidence = min(0.95, 0.5 + (45 - score) / 70)

        return {
            'prediction': self.INV_LABEL[pred_idx],
            'prediction_idx': pred_idx,
            'score': round(score, 1),
            'confidence': round(confidence, 3),
            'reasoning': [
                {'factor': r[0], 'points': r[1], 'explanation': r[2]}
                for r in reasons
            ],
        }

    def predict(self, post_data: Dict) -> Dict:
        """Predict from a raw post data dict (same format as NN predictor)."""
        if 'data' not in post_data:
            post_data = {'data': post_data}
        features = extract_all_features(post_data)
        brand = post_data.get('data', post_data).get(
            'profile_stats', {}
        ).get('username', 'unknown')
        result = self.score_post(features, brand)
        result['features_used'] = features
        result['brand_context'] = self.brand_stats.get(brand, {})
        return result

    def predict_batch(self, features_list: List[Dict],
                      brands: List[str]) -> np.ndarray:
        """Predict class indices for a batch of feature dicts."""
        preds = []
        for feats, brand in zip(features_list, brands):
            r = self.score_post(feats, brand)
            preds.append(r['prediction_idx'])
        return np.array(preds)
