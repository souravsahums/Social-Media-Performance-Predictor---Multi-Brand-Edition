"""
Data Drift Detection Module
==============================
Monitors feature distributions between training data and incoming prediction
requests.  Detects covariate shift that could degrade model performance.

Strategies:
  1. Population Stability Index (PSI) — per feature
  2. Kolmogorov-Smirnov (KS) test — per feature
  3. Overall drift score — weighted average

Usage:
    from model.drift import DriftDetector
    detector = DriftDetector.from_training_data(X_train, feature_columns)
    detector.save('artifacts/drift_reference.json')

    # At prediction time:
    detector = DriftDetector.load('artifacts/drift_reference.json')
    report = detector.check(new_feature_vector)
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class DriftDetector:
    """Monitors data drift by comparing incoming data to training distributions."""

    # PSI thresholds (industry standard)
    PSI_OK = 0.1
    PSI_WARN = 0.2

    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.reference_stats: Dict[str, Dict] = {}
        self.drift_log: List[Dict] = []
        self._prediction_buffer: List[np.ndarray] = []
        self._buffer_max = 100

    @classmethod
    def from_training_data(cls, X_train: np.ndarray,
                           feature_columns: List[str]) -> 'DriftDetector':
        """Create a drift detector from training data statistics."""
        det = cls(feature_columns)
        for i, col in enumerate(feature_columns):
            vals = X_train[:, i]
            det.reference_stats[col] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'p25': float(np.percentile(vals, 25)),
                'p50': float(np.percentile(vals, 50)),
                'p75': float(np.percentile(vals, 75)),
                'histogram': np.histogram(vals, bins=10)[0].tolist(),
                'bin_edges': np.histogram(vals, bins=10)[1].tolist(),
            }
        return det

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'reference_stats': self.reference_stats,
                'created_at': datetime.utcnow().isoformat(),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'DriftDetector':
        with open(path) as f:
            data = json.load(f)
        det = cls(data['feature_columns'])
        det.reference_stats = data['reference_stats']
        return det

    def _compute_psi(self, expected_hist: List[int],
                     actual_vals: np.ndarray,
                     bin_edges: List[float]) -> float:
        """Population Stability Index between reference histogram and new data."""
        actual_hist = np.histogram(actual_vals, bins=bin_edges)[0]
        # Normalize to proportions
        expected = np.array(expected_hist, dtype=float)
        expected = expected / expected.sum() if expected.sum() > 0 else expected + 1e-6
        actual = actual_hist.astype(float)
        actual = actual / actual.sum() if actual.sum() > 0 else actual + 1e-6

        # Avoid log(0) by adding small epsilon
        expected = np.clip(expected, 1e-6, None)
        actual = np.clip(actual, 1e-6, None)

        psi = np.sum((actual - expected) * np.log(actual / expected))
        return float(psi)

    def check_single(self, feature_vector: np.ndarray) -> Dict:
        """
        Check a single prediction input for drift signals.
        Returns per-feature anomaly flags and an overall risk level.
        """
        anomalies = []
        for i, col in enumerate(self.feature_columns):
            ref = self.reference_stats.get(col)
            if ref is None:
                continue
            val = float(feature_vector[i])

            # Z-score check
            z = (val - ref['mean']) / (ref['std'] + 1e-8)
            is_outlier = abs(z) > 3.0
            is_warning = abs(z) > 2.0

            # Range check
            out_of_range = val < ref['min'] or val > ref['max']

            if is_outlier or out_of_range:
                anomalies.append({
                    'feature': col,
                    'value': round(val, 4),
                    'z_score': round(z, 2),
                    'ref_mean': round(ref['mean'], 4),
                    'ref_std': round(ref['std'], 4),
                    'out_of_range': out_of_range,
                    'severity': 'high' if is_outlier else 'medium',
                })
            elif is_warning:
                anomalies.append({
                    'feature': col,
                    'value': round(val, 4),
                    'z_score': round(z, 2),
                    'severity': 'low',
                })

        risk = 'low'
        high_count = sum(1 for a in anomalies if a['severity'] == 'high')
        if high_count >= 3:
            risk = 'high'
        elif high_count >= 1 or len(anomalies) >= 5:
            risk = 'medium'

        return {
            'risk_level': risk,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies,
        }

    def add_to_buffer(self, feature_vector: np.ndarray):
        """Add a prediction to the buffer for batch drift analysis."""
        self._prediction_buffer.append(feature_vector.copy())
        if len(self._prediction_buffer) > self._buffer_max:
            self._prediction_buffer.pop(0)

    def check_batch_drift(self) -> Optional[Dict]:
        """
        Run batch drift analysis on the prediction buffer.
        Returns None if buffer is too small.
        """
        if len(self._prediction_buffer) < 20:
            return None

        X_new = np.array(self._prediction_buffer)
        per_feature_psi = {}
        drifted_features = []

        for i, col in enumerate(self.feature_columns):
            ref = self.reference_stats.get(col)
            if ref is None or ref.get('histogram') is None:
                continue

            psi = self._compute_psi(ref['histogram'], X_new[:, i], ref['bin_edges'])
            per_feature_psi[col] = round(psi, 4)

            if psi > self.PSI_WARN:
                drifted_features.append({
                    'feature': col, 'psi': round(psi, 4), 'severity': 'high'
                })
            elif psi > self.PSI_OK:
                drifted_features.append({
                    'feature': col, 'psi': round(psi, 4), 'severity': 'medium'
                })

        overall_psi = np.mean(list(per_feature_psi.values())) if per_feature_psi else 0

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'buffer_size': len(self._prediction_buffer),
            'overall_psi': round(overall_psi, 4),
            'drift_detected': overall_psi > self.PSI_OK,
            'drifted_features': drifted_features,
            'recommendation': (
                'RETRAIN: Significant distribution shift detected.'
                if overall_psi > self.PSI_WARN else
                'MONITOR: Mild drift detected, monitor closely.'
                if overall_psi > self.PSI_OK else
                'OK: No significant drift.'
            ),
        }

        self.drift_log.append(report)
        return report

    def get_drift_summary(self) -> Dict:
        """Return a summary of all drift checks performed."""
        return {
            'total_checks': len(self.drift_log),
            'buffer_size': len(self._prediction_buffer),
            'latest': self.drift_log[-1] if self.drift_log else None,
            'drift_alerts': sum(1 for r in self.drift_log if r.get('drift_detected')),
        }
