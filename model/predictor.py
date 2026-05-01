"""
Predictor – loads trained artifacts and makes predictions with explanations.

Primary model: Random Forest (held-out test accuracy 55.3%, CV 58.7%).
Secondary signal: Neural Network (held-out 54.0%, CV 45.5%) — exposed in the
response for diagnostic comparison but does NOT override the RF prediction.

Why RF is primary:
    On every honest evaluation we ran (stratified 5-fold CV and a held-out
    20% test split that the model never saw during training), RF outperformed
    both the standalone NN and a weighted ensemble. We therefore ship the RF
    as the canonical prediction rather than carrying a weaker ensemble.

Interpretability:
    SHAP TreeExplainer provides per-feature contributions for the RF
    prediction. Rule-based explanations augment SHAP with human-readable
    factors (content type, timing, collaboration, etc.).
"""

import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, List, Optional

from model.features import extract_all_features, FEATURE_COLUMNS
from model.train import EngagementNet, INV_LABEL_MAP
from model.visual_features import extract_visual_features_from_post

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')


class EngagementPredictor:
    """Wraps the Random Forest (primary) and Neural Network (diagnostic) and
    provides predictions, SHAP attributions, and human-readable explanations.
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.rf_model = None
        self._shap_explainer = None  # Lazy-initialised TreeExplainer for RF
        self._load()

    def _load(self):
        with open(os.path.join(self.artifacts_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.artifacts_dir, 'brand_stats.json')) as f:
            self.brand_stats = json.load(f)
        with open(os.path.join(self.artifacts_dir, 'feature_columns.json')) as f:
            self.feature_columns = json.load(f)

        self.model = EngagementNet(input_dim=len(self.feature_columns))
        state = torch.load(
            os.path.join(self.artifacts_dir, 'model.pt'),
            map_location='cpu', weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()

        # Load Random Forest model if available
        rf_path = os.path.join(self.artifacts_dir, 'rf_model.pkl')
        if os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                self.rf_model = pickle.load(f)

        # Load feature means for explanation
        self._feature_means = np.zeros(len(self.feature_columns))
        self._feature_stds = np.ones(len(self.feature_columns))
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            self._feature_means = self.scaler.mean_
            self._feature_stds = self.scaler.scale_

    def predict(self, post_data: Dict, include_visual: bool = False) -> Dict:
        """
        Predict engagement performance using ensemble (NN + RF).

        Args:
            post_data: Dict with keys matching the dataset schema.
            include_visual: If True, download thumbnail and extract visual features.

        Returns:
            Dict with ensemble prediction, individual model outputs, explanation.
        """
        if 'data' not in post_data:
            post_data = {'data': post_data}

        # Extract features (with optional visual features from URL)
        features = extract_all_features(post_data, include_visual=include_visual)

        feature_vec = np.array(
            [features.get(col, 0) for col in self.feature_columns],
            dtype=np.float32,
        ).reshape(1, -1)

        X_scaled = self.scaler.transform(feature_vec)

        # Neural Network prediction
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            nn_probs = torch.softmax(logits, dim=1).numpy()[0]
        nn_pred_idx = int(nn_probs.argmax())
        nn_label = INV_LABEL_MAP[nn_pred_idx]
        nn_confidence = float(nn_probs[nn_pred_idx])

        # Random Forest prediction
        rf_label = None
        rf_probs = None
        rf_confidence = None
        if self.rf_model is not None:
            rf_probs_raw = self.rf_model.predict_proba(X_scaled)[0]
            rf_pred_idx = int(rf_probs_raw.argmax())
            rf_label = INV_LABEL_MAP[rf_pred_idx]
            rf_confidence = float(rf_probs_raw[rf_pred_idx])
            rf_probs = rf_probs_raw

        # PRIMARY prediction: Random Forest (best held-out performance).
        # If RF is unavailable, fall back to NN.
        if rf_probs is not None:
            primary_probs = rf_probs
            primary_label = rf_label
            primary_confidence = rf_confidence
            primary_idx = int(rf_probs.argmax())
            primary_source = 'random_forest'
        else:
            primary_probs = nn_probs
            primary_label = nn_label
            primary_confidence = nn_confidence
            primary_idx = nn_pred_idx
            primary_source = 'neural_network'

        # SHAP attribution for the RF prediction (interpretability layer)
        shap_top_features = self._shap_top_features(
            X_scaled, primary_idx, top_k=8,
        ) if self.rf_model is not None and primary_source == 'random_forest' else []

        explanation = self._explain(features, X_scaled[0], primary_label, primary_probs)
        brand = post_data.get('data', post_data).get(
            'profile_stats', {}
        ).get('username', 'unknown')

        brand_context = self.brand_stats.get(brand, {})

        # Diagnostic: do NN and RF agree?
        models_agree = (rf_label == nn_label) if rf_label else None

        result = {
            'prediction': primary_label,
            'confidence': round(primary_confidence, 4),
            'primary_model': primary_source,
            'probabilities': {
                'low': round(float(primary_probs[0]), 4),
                'medium': round(float(primary_probs[1]), 4),
                'high': round(float(primary_probs[2]), 4),
            },
            'model_predictions': {
                'random_forest': {
                    'prediction': rf_label,
                    'confidence': round(rf_confidence, 4) if rf_confidence else None,
                    'role': 'primary',
                } if rf_label else None,
                'neural_network': {
                    'prediction': nn_label,
                    'confidence': round(nn_confidence, 4),
                    'role': 'secondary_diagnostic',
                },
                'models_agree': models_agree,
            },
            'shap_top_features': shap_top_features,
            'explanation': explanation,
            'brand_context': brand_context,
            'features_used': features,
        }

        return result

    # ------------------------------------------------------------------
    # SHAP interpretability
    # ------------------------------------------------------------------

    def _get_shap_explainer(self):
        """Lazy-initialise SHAP TreeExplainer for the Random Forest."""
        if self._shap_explainer is not None:
            return self._shap_explainer
        if self.rf_model is None:
            return None
        try:
            import shap
            # TreeExplainer is fast (milliseconds) for tree-based models.
            self._shap_explainer = shap.TreeExplainer(self.rf_model)
            return self._shap_explainer
        except Exception as exc:  # pragma: no cover - SHAP optional at runtime
            print(f"[predictor] SHAP unavailable: {exc}")
            return None

    def _shap_top_features(self, X_scaled: np.ndarray, class_idx: int,
                           top_k: int = 8) -> List[Dict]:
        """Return top-k SHAP contributions for the predicted class.

        Each entry: {feature, value, shap_value, direction}
        - shap_value > 0 → push toward the predicted class
        - shap_value < 0 → push away from the predicted class
        """
        explainer = self._get_shap_explainer()
        if explainer is None:
            return []

        try:
            shap_values = explainer.shap_values(X_scaled)
            # sklearn RF returns either a list[n_classes] of (n,F) arrays
            # OR a single (n, F, n_classes) ndarray depending on shap version.
            if isinstance(shap_values, list):
                cls_vals = np.asarray(shap_values[class_idx])[0]
            else:
                arr = np.asarray(shap_values)
                if arr.ndim == 3:
                    cls_vals = arr[0, :, class_idx]
                else:
                    cls_vals = arr[0]
        except Exception as exc:  # pragma: no cover
            print(f"[predictor] SHAP computation failed: {exc}")
            return []

        # Rank by absolute contribution
        order = np.argsort(np.abs(cls_vals))[::-1][:top_k]
        results = []
        for idx in order:
            feat_name = self.feature_columns[int(idx)]
            shap_v = float(cls_vals[int(idx)])
            results.append({
                'feature': feat_name,
                'scaled_value': round(float(X_scaled[0, int(idx)]), 4),
                'shap_value': round(shap_v, 4),
                'direction': 'pushes_toward' if shap_v > 0 else 'pushes_away',
            })
        return results

    def _explain(self, raw_features: Dict, scaled_vec: np.ndarray,
                 prediction: str, probs: np.ndarray) -> List[Dict]:
        """Generate human-readable explanations based on feature importance."""
        explanations = []

        # Content type insight
        if raw_features.get('is_reel'):
            explanations.append({
                'factor': 'Content Type: Reel',
                'impact': 'positive',
                'detail': 'Reels generally get higher reach and engagement than static posts.',
            })
        elif raw_features.get('is_album'):
            explanations.append({
                'factor': 'Content Type: Album/Carousel',
                'impact': 'neutral',
                'detail': 'Albums get moderate engagement; no view counts are expected.',
            })
        else:
            explanations.append({
                'factor': 'Content Type: Static Post',
                'impact': 'negative',
                'detail': 'Static posts typically get lower reach than reels.',
            })

        # Duration
        dur = raw_features.get('duration', 0)
        if dur > 0:
            if 15 <= dur <= 30:
                explanations.append({
                    'factor': f'Video Duration: {dur}s',
                    'impact': 'positive',
                    'detail': 'Short-form content (15-30s) tends to perform best on Instagram.',
                })
            elif dur > 60:
                explanations.append({
                    'factor': f'Video Duration: {dur}s',
                    'impact': 'negative',
                    'detail': 'Longer videos (>60s) tend to lose viewer retention.',
                })

        # Collaboration
        if raw_features.get('is_collaborated'):
            explanations.append({
                'factor': 'Collaborated Post',
                'impact': 'positive',
                'detail': f'Collaboration with {raw_features.get("collaborator_count", 1)} '
                          f'creator(s) expands reach to their audience.',
            })

        if raw_features.get('is_ugc'):
            explanations.append({
                'factor': 'User-Generated Content',
                'impact': 'positive',
                'detail': 'UGC posts often feel more authentic and drive engagement.',
            })

        # Caption analysis
        wc = raw_features.get('word_count', 0)
        if wc > 80:
            explanations.append({
                'factor': 'Long Caption',
                'impact': 'negative',
                'detail': f'Caption has {wc} words. Shorter captions tend to perform better.',
            })
        elif wc < 5 and wc > 0:
            explanations.append({
                'factor': 'Very Short Caption',
                'impact': 'neutral',
                'detail': 'Minimal caption may limit discoverability.',
            })

        if raw_features.get('has_cta'):
            explanations.append({
                'factor': 'Call-to-Action Present',
                'impact': 'positive',
                'detail': 'CTAs like "share", "tag", "comment" drive interaction.',
            })

        if raw_features.get('hashtag_count', 0) > 5:
            explanations.append({
                'factor': f'{raw_features["hashtag_count"]} Hashtags',
                'impact': 'neutral',
                'detail': 'Many hashtags can increase discoverability but may look spammy.',
            })

        if raw_features.get('has_question'):
            explanations.append({
                'factor': 'Question in Caption',
                'impact': 'positive',
                'detail': 'Questions encourage comments and boost engagement.',
            })

        # Visual brand presence
        if raw_features.get('has_brand_in_visual'):
            explanations.append({
                'factor': 'Brand Visible in Creative',
                'impact': 'positive',
                'detail': 'Brand logo/product visibility reinforces brand engagement.',
            })

        if raw_features.get('has_person_in_visual'):
            explanations.append({
                'factor': 'Person/Face in Creative',
                'impact': 'positive',
                'detail': 'Posts with faces tend to get more engagement.',
            })

        # Timing
        if raw_features.get('is_morning'):
            explanations.append({
                'factor': 'Posted in Morning (6-10 AM)',
                'impact': 'neutral',
                'detail': 'Morning posts catch early scrollers but may miss peak hours.',
            })
        elif raw_features.get('is_evening'):
            explanations.append({
                'factor': 'Posted in Evening (5-9 PM)',
                'impact': 'positive',
                'detail': 'Evening posts align with peak Instagram usage in India.',
            })

        if raw_features.get('is_weekend'):
            explanations.append({
                'factor': 'Weekend Post',
                'impact': 'positive',
                'detail': 'Weekend posts often get higher casual engagement.',
            })

        # Overall confidence note
        if probs.max() < 0.5:
            explanations.append({
                'factor': 'Low Model Confidence',
                'impact': 'neutral',
                'detail': 'The model is uncertain; this post has mixed signals.',
            })

        return explanations
