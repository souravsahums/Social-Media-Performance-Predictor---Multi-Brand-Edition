"""
Feedback Mechanism & Anti-Corruption Module
==============================================
Handles user feedback when predictions are wrong, stores corrections,
validates feedback integrity, and provides a retraining pipeline.

Anti-corruption measures:
  1. Rate limiting (max N feedback entries per IP/session per hour)
  2. Statistical anomaly detection (reject outlier feedback)
  3. Consistency checking (flag contradictory feedback for the same input)
  4. Feedback quality scoring
  5. Audit trail

Usage:
    from model.feedback import FeedbackManager
    fm = FeedbackManager('artifacts/feedback')
    fm.submit(prediction_id, correct_label, features, metadata)
    report = fm.get_summary()
    training_data = fm.export_for_retraining()
"""

import json
import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np


class FeedbackEntry:
    """A single feedback correction from a user."""

    def __init__(self, prediction_id: str, predicted_label: str,
                 correct_label: str, features: Dict, metadata: Dict = None):
        self.id = hashlib.sha256(
            f"{prediction_id}{time.time()}".encode()
        ).hexdigest()[:16]
        self.prediction_id = prediction_id
        self.predicted_label = predicted_label
        self.correct_label = correct_label
        self.features = features
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()
        self.quality_score = 1.0  # Will be adjusted by validation
        self.status = 'pending'  # pending | accepted | rejected | quarantined

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'predicted_label': self.predicted_label,
            'correct_label': self.correct_label,
            'features': self.features,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'quality_score': self.quality_score,
            'status': self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'FeedbackEntry':
        entry = cls(
            d['prediction_id'], d['predicted_label'],
            d['correct_label'], d['features'], d.get('metadata', {}),
        )
        entry.id = d['id']
        entry.timestamp = d['timestamp']
        entry.quality_score = d.get('quality_score', 1.0)
        entry.status = d.get('status', 'pending')
        return entry


class FeedbackManager:
    """Manages feedback collection, validation, and retraining pipeline."""

    VALID_LABELS = {'low', 'medium', 'high'}
    MAX_FEEDBACK_PER_HOUR = 30  # Rate limit per source
    MIN_QUALITY_FOR_TRAINING = 0.5

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._feedback: List[FeedbackEntry] = []
        self._rate_tracker: Dict[str, List[float]] = defaultdict(list)
        self._load()

    def _feedback_path(self) -> str:
        return os.path.join(self.storage_dir, 'feedback_data.json')

    def _audit_path(self) -> str:
        return os.path.join(self.storage_dir, 'audit_log.json')

    def _load(self):
        path = self._feedback_path()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self._feedback = [FeedbackEntry.from_dict(d) for d in data]

    def _save(self):
        with open(self._feedback_path(), 'w') as f:
            json.dump([e.to_dict() for e in self._feedback], f, indent=2)

    def _audit(self, action: str, details: Dict):
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'details': details,
        }
        audit_path = self._audit_path()
        if os.path.exists(audit_path):
            with open(audit_path) as f:
                log = json.load(f)
        else:
            log = []
        log.append(entry)
        # Keep last 1000 audit entries
        if len(log) > 1000:
            log = log[-1000:]
        with open(audit_path, 'w') as f:
            json.dump(log, f, indent=2)

    # ----- Anti-corruption checks -----

    def _check_rate_limit(self, source_id: str) -> Tuple[bool, str]:
        """Check if the source has exceeded the feedback rate limit."""
        now = time.time()
        hour_ago = now - 3600
        # Clean old entries
        self._rate_tracker[source_id] = [
            t for t in self._rate_tracker[source_id] if t > hour_ago
        ]
        if len(self._rate_tracker[source_id]) >= self.MAX_FEEDBACK_PER_HOUR:
            return False, f"Rate limit exceeded: {self.MAX_FEEDBACK_PER_HOUR}/hour"
        return True, ""

    def _check_label_validity(self, label: str) -> Tuple[bool, str]:
        """Check if the label is one of the valid classes."""
        if label not in self.VALID_LABELS:
            return False, f"Invalid label '{label}'. Must be one of {self.VALID_LABELS}"
        return True, ""

    def _check_consistency(self, features: Dict, correct_label: str) -> Tuple[float, str]:
        """
        Check if similar past feedback is consistent.
        Returns a quality score (0-1) and a message.
        """
        similar_feedback = []
        for entry in self._feedback:
            if entry.status == 'rejected':
                continue
            # Simple similarity: check if same brand and similar caption length
            same_brand = (
                entry.features.get('brand_cocacola_india') == features.get('brand_cocacola_india') and
                entry.features.get('brand_pepsiindia') == features.get('brand_pepsiindia') and
                entry.features.get('brand_sprite_india') == features.get('brand_sprite_india') and
                entry.features.get('brand_redbullindia') == features.get('brand_redbullindia') and
                entry.features.get('brand_thumsupofficial') == features.get('brand_thumsupofficial')
            )
            similar_caption = abs(
                entry.features.get('caption_length', 0) - features.get('caption_length', 0)
            ) < 50
            if same_brand and similar_caption:
                similar_feedback.append(entry)

        if not similar_feedback:
            return 1.0, "No similar past feedback for comparison."

        # Check if the correction is consistent with past corrections
        contradictions = sum(
            1 for e in similar_feedback
            if e.correct_label != correct_label and e.quality_score > 0.5
        )
        if contradictions > len(similar_feedback) * 0.5:
            return 0.3, (f"Warning: {contradictions}/{len(similar_feedback)} "
                        f"similar entries have different corrections.")
        return 1.0, "Consistent with past feedback."

    def _check_statistical_anomaly(self, features: Dict,
                                    correct_label: str) -> Tuple[float, str]:
        """
        Detect statistically anomalous feedback — e.g., marking a post with
        0 engagement features as 'high'.
        """
        suspicious = []

        # Check: marking as 'high' but no positive engagement signals
        if correct_label == 'high':
            if (features.get('is_reel', 0) == 0 and
                features.get('is_collaborated', 0) == 0 and
                features.get('has_cta', 0) == 0 and
                features.get('word_count', 0) == 0):
                suspicious.append("Post has no positive engagement signals but marked as 'high'")

        # Check: marking as 'low' but has strong positive signals
        if correct_label == 'low':
            positive_signals = sum([
                features.get('is_reel', 0),
                features.get('is_collaborated', 0),
                features.get('has_cta', 0),
                features.get('has_person_in_visual', 0),
                features.get('is_evening', 0),
            ])
            if positive_signals >= 4:
                suspicious.append("Post has 4+ positive signals but marked as 'low'")

        if suspicious:
            return 0.5, "Statistically unusual: " + "; ".join(suspicious)
        return 1.0, "No statistical anomalies detected."

    # ----- Public API -----

    def submit(self, prediction_id: str, predicted_label: str,
               correct_label: str, features: Dict,
               source_id: str = 'anonymous',
               metadata: Dict = None) -> Dict:
        """
        Submit a feedback correction.

        Returns:
            Dict with status, message, and quality_score.
        """
        # Validation checks
        valid, msg = self._check_label_validity(correct_label)
        if not valid:
            self._audit('rejected', {'reason': msg, 'source': source_id})
            return {'status': 'rejected', 'message': msg}

        rate_ok, rate_msg = self._check_rate_limit(source_id)
        if not rate_ok:
            self._audit('rate_limited', {'source': source_id, 'reason': rate_msg})
            return {'status': 'rejected', 'message': rate_msg}

        # Same label feedback is useless
        if predicted_label == correct_label:
            return {'status': 'rejected',
                    'message': 'Feedback label is same as prediction — no correction needed.'}

        # Quality scoring
        consistency_score, consistency_msg = self._check_consistency(features, correct_label)
        anomaly_score, anomaly_msg = self._check_statistical_anomaly(features, correct_label)
        quality = min(consistency_score, anomaly_score)

        entry = FeedbackEntry(
            prediction_id=prediction_id,
            predicted_label=predicted_label,
            correct_label=correct_label,
            features=features,
            metadata=metadata or {},
        )
        entry.quality_score = quality

        if quality < 0.3:
            entry.status = 'rejected'
            action = 'rejected'
            message = f"Feedback rejected (quality={quality:.2f}): {anomaly_msg} {consistency_msg}"
        elif quality < self.MIN_QUALITY_FOR_TRAINING:
            entry.status = 'quarantined'
            action = 'quarantined'
            message = (f"Feedback quarantined for review (quality={quality:.2f}): "
                      f"{anomaly_msg} {consistency_msg}")
        else:
            entry.status = 'accepted'
            action = 'accepted'
            message = f"Feedback accepted (quality={quality:.2f})."

        self._feedback.append(entry)
        self._rate_tracker[source_id].append(time.time())
        self._save()
        self._audit(action, {
            'feedback_id': entry.id,
            'source': source_id,
            'predicted': predicted_label,
            'corrected': correct_label,
            'quality': quality,
        })

        return {
            'status': entry.status,
            'feedback_id': entry.id,
            'quality_score': round(quality, 2),
            'message': message,
        }

    def get_summary(self) -> Dict:
        """Get a summary of all collected feedback."""
        if not self._feedback:
            return {'total': 0, 'accepted': 0, 'rejected': 0,
                    'quarantined': 0, 'pending': 0}

        counts = defaultdict(int)
        for e in self._feedback:
            counts[e.status] += 1

        # Confusion between predicted and correct labels
        corrections = defaultdict(int)
        for e in self._feedback:
            if e.status in ('accepted', 'pending'):
                key = f"{e.predicted_label} → {e.correct_label}"
                corrections[key] += 1

        return {
            'total': len(self._feedback),
            'accepted': counts.get('accepted', 0),
            'rejected': counts.get('rejected', 0),
            'quarantined': counts.get('quarantined', 0),
            'pending': counts.get('pending', 0),
            'correction_patterns': dict(corrections),
            'avg_quality': round(
                np.mean([e.quality_score for e in self._feedback]), 2
            ),
        }

    def export_for_retraining(self) -> List[Dict]:
        """
        Export high-quality accepted feedback as training data.

        Returns a list of dicts with 'features' and 'label' keys,
        suitable for augmenting the training dataset.

        Retraining Pipeline:
          1. Load original training data
          2. Call this method to get validated feedback
          3. Merge feedback features/labels into training data
          4. Re-fit scaler on combined data
          5. Retrain model
          6. Evaluate on held-out set
          7. Deploy if metrics improve
        """
        return [
            {
                'features': e.features,
                'label': e.correct_label,
                'feedback_id': e.id,
                'quality_score': e.quality_score,
            }
            for e in self._feedback
            if e.status == 'accepted' and e.quality_score >= self.MIN_QUALITY_FOR_TRAINING
        ]

    def get_retraining_pipeline(self) -> Dict:
        """
        Returns the recommended retraining pipeline configuration.
        """
        accepted = self.export_for_retraining()
        return {
            'available_feedback': len(accepted),
            'min_samples_for_retrain': 20,
            'ready_to_retrain': len(accepted) >= 20,
            'pipeline_steps': [
                '1. Load original training dataset',
                '2. Export accepted feedback via export_for_retraining()',
                '3. Convert feedback features to DataFrame rows',
                '4. Concatenate with original training data',
                '5. Re-compute brand stats with augmented data',
                '6. Re-fit StandardScaler on combined features',
                '7. Train model with cross-validation',
                '8. Compare new model metrics vs current model',
                '9. Deploy new model only if metrics improve',
                '10. Archive old model and reset feedback buffer',
            ],
            'auto_retrain_trigger': (
                'Automatic retraining recommended when:\n'
                '  - 20+ accepted feedback entries\n'
                '  - Accuracy drop detected via drift monitoring\n'
                '  - Feedback correction rate > 30%'
            ),
        }
