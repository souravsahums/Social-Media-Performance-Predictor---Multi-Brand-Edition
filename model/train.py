"""
PyTorch model definition and training script.
Trains an engagement-performance classifier (low / medium / high).
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, List

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.loader import load_and_clean
from model.features import (
    build_dataframe, compute_brand_stats, categorize_performance,
    FEATURE_COLUMNS,
)
from model.visual_features import extract_visual_features_batch

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
LABEL_MAP = {'low': 0, 'medium': 1, 'high': 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ----------------------------------------------
# Dataset
# ----------------------------------------------

class PostDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------------------------
# Neural Network
# ----------------------------------------------

class EngagementNet(nn.Module):
    """Multi-layer classifier for engagement performance prediction."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------
# Training helpers
# ----------------------------------------------

def prepare_data(dataset: List[Dict],
                 visual_features: List[Dict] = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler, Dict]:
    """Prepare feature matrix and labels.
    
    Args:
        dataset: List of post dicts
        visual_features: Pre-computed visual features from batch extraction.
                        If None, zeros are used for visual columns.
    """
    features_df, targets_df = build_dataframe(dataset, visual_features=visual_features)
    brand_stats = compute_brand_stats(dataset)

    # Build labels
    labels = []
    for i, post in enumerate(dataset):
        data = post.get('data', post)
        brand = data.get('profile_stats', {}).get('username', 'unknown')
        er = targets_df.iloc[i]['engagement_rate']
        labels.append(LABEL_MAP[categorize_performance(er, brand, brand_stats)])
    labels = np.array(labels)

    X = features_df[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, labels, scaler, brand_stats


def train_model(X: np.ndarray, y: np.ndarray,
                epochs: int = 200, lr: float = 1e-3,
                batch_size: int = 32, device: str = 'cpu') -> EngagementNet:
    ds = PostDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # Handle class imbalance
    class_counts = np.bincount(y, minlength=3).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights /= class_weights.sum()
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = EngagementNet(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.4f}")

    return model


# ----------------------------------------------
# Cross-validation
# ----------------------------------------------

def cross_validate(X: np.ndarray, y: np.ndarray,
                   n_splits: int = 5, epochs: int = 150) -> Dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = train_model(X_train, y_train, epochs=epochs)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_val, dtype=torch.float32))
            preds = logits.argmax(1).numpy()

        acc = (preds == y_val).mean()
        fold_results.append({'fold': fold + 1, 'accuracy': float(acc)})
        print(f"  Fold {fold+1} accuracy: {acc:.4f}")

    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    print(f"\nMean CV accuracy: {mean_acc:.4f}")
    return {'folds': fold_results, 'mean_accuracy': float(mean_acc)}


# ----------------------------------------------
# Save / load
# ----------------------------------------------

def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def save_artifacts(model: EngagementNet, scaler: StandardScaler,
                   brand_stats: Dict, cv_results: Dict,
                   X_train: np.ndarray = None,
                   rf_model: RandomForestClassifier = None):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, 'model.pt'))
    with open(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(ARTIFACTS_DIR, 'brand_stats.json'), 'w') as f:
        json.dump(brand_stats, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(FEATURE_COLUMNS, f)

    # Save Random Forest model
    if rf_model is not None:
        with open(os.path.join(ARTIFACTS_DIR, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
        print("Random Forest model saved.")

    # Save drift reference data
    if X_train is not None:
        from model.drift import DriftDetector
        detector = DriftDetector.from_training_data(X_train, FEATURE_COLUMNS)
        detector.save(os.path.join(ARTIFACTS_DIR, 'drift_reference.json'))
        print("Drift reference saved.")

    print(f"Artifacts saved to {ARTIFACTS_DIR}")


def load_artifacts():
    state = torch.load(os.path.join(ARTIFACTS_DIR, 'model.pt'),
                       map_location='cpu', weights_only=True)
    with open(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'brand_stats.json')) as f:
        brand_stats = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.json')) as f:
        feature_cols = json.load(f)

    input_dim = len(feature_cols)
    model = EngagementNet(input_dim=input_dim)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, brand_stats, feature_cols


# ----------------------------------------------
# Main
# ----------------------------------------------

def main():
    print("Loading dataset...")
    dataset = load_and_clean()
    print(f"  {len(dataset)} posts loaded")

    print("\nExtracting visual features from thumbnails (parallel download)...")
    visual_features = extract_visual_features_batch(
        dataset, max_workers=8,
        progress_callback=lambda done, total: print(f"  Downloaded {done}/{total} images")
    )
    # Count successful extractions
    success = sum(1 for vf in visual_features if vf.get('img_width', 0) > 0)
    print(f"  Visual features extracted: {success}/{len(dataset)} images loaded successfully")

    print("\nPreparing features...")
    X, y, scaler, brand_stats = prepare_data(dataset, visual_features=visual_features)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("\nRunning 5-fold cross-validation (Neural Network)...")
    cv_results = cross_validate(X, y, n_splits=5, epochs=150)

    print("\nTraining final Neural Network on full data...")
    final_model = train_model(X, y, epochs=200)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X, y)
    # Quick CV for RF
    from sklearn.model_selection import cross_val_score
    rf_cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=300, class_weight='balanced',
                               random_state=42, n_jobs=-1),
        X, y, cv=5, scoring='accuracy'
    )
    print(f"  RF 5-fold CV accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
    cv_results['rf_cv_accuracy'] = float(rf_cv_scores.mean())
    cv_results['rf_cv_std'] = float(rf_cv_scores.std())

    print("\nSaving artifacts...")
    save_artifacts(final_model, scaler, brand_stats, cv_results,
                   X_train=X, rf_model=rf_model)
    print("Done!")


if __name__ == '__main__':
    main()
