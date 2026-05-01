"""
Evaluation module – computes metrics, baselines, and failure analysis.

Reports:
  1. Baselines (majority class, random)
  2. 5-fold cross-validation (NN + RF + ensemble)
  3. Held-out test set (stratified 80/20 split, never seen during training)
  4. Per-brand accuracy
  5. Failure analysis
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.loader import load_and_clean
from model.features import build_dataframe, compute_brand_stats, categorize_performance, FEATURE_COLUMNS
from model.train import (
    prepare_data, train_model, train_random_forest, EngagementNet,
    LABEL_MAP, INV_LABEL_MAP,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')

# Ensemble weights — must match model/predictor.py
RF_WEIGHT = 0.85
NN_WEIGHT = 0.15


def baseline_majority(y: np.ndarray) -> float:
    """Accuracy of always-predict-majority-class baseline."""
    counts = np.bincount(y, minlength=3)
    return float(counts.max() / len(y))


def baseline_random(y: np.ndarray, seed: int = 42) -> float:
    """Accuracy of random uniform baseline."""
    rng = np.random.RandomState(seed)
    preds = rng.randint(0, 3, size=len(y))
    return float((preds == y).mean())


def evaluate_model(X: np.ndarray, y: np.ndarray, model: EngagementNet) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(1).numpy()

    acc = accuracy_score(y, preds)
    f1_macro = f1_score(y, preds, average='macro')
    f1_weighted = f1_score(y, preds, average='weighted')
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    report = classification_report(
        y, preds, target_names=['low', 'medium', 'high'], output_dict=True
    )

    return {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': preds.tolist(),
        'probabilities': probs.tolist(),
    }


def held_out_evaluation(X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2, seed: int = 42) -> dict:
    """Train on a stratified train split, evaluate on a held-out test split.

    The test set is NOT used during training, hyper-parameter tuning, or
    feature engineering — it provides an honest generalisation estimate.
    Reports NN, RF, and ensemble metrics on the same held-out split.
    """
    print("\n" + "=" * 60)
    print(f"HELD-OUT TEST SET EVALUATION (stratified {int((1-test_size)*100)}/{int(test_size*100)} split)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )
    print(f"Train: {len(y_train)} samples | Test: {len(y_test)} samples")
    print(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- Train NN on train split only ---
    nn_model = train_model(X_train, y_train, epochs=200)
    nn_model.eval()
    with torch.no_grad():
        nn_logits = nn_model(torch.tensor(X_test, dtype=torch.float32))
        nn_probs = torch.softmax(nn_logits, dim=1).numpy()
        nn_preds = nn_logits.argmax(1).numpy()

    # --- Train RF on train split only ---
    rf_model = train_random_forest(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)
    rf_preds = rf_probs.argmax(1)

    # --- Ensemble ---
    ens_probs = NN_WEIGHT * nn_probs + RF_WEIGHT * rf_probs
    ens_preds = ens_probs.argmax(1)

    def _metrics(name, preds):
        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average='macro')
        f1w = f1_score(y_test, preds, average='weighted')
        print(f"\n{name}:")
        print(f"  accuracy  : {acc:.4f}")
        print(f"  f1_macro  : {f1m:.4f}")
        print(f"  f1_weighted: {f1w:.4f}")
        print(classification_report(
            y_test, preds, target_names=['low', 'medium', 'high'], zero_division=0,
        ))
        return {'accuracy': float(acc), 'f1_macro': float(f1m), 'f1_weighted': float(f1w)}

    print("\n--- Held-Out Test Metrics ---")
    nn_m = _metrics('Neural Network (held-out)', nn_preds)
    rf_m = _metrics('Random Forest (held-out, PRIMARY)', rf_preds)
    ens_m = _metrics('Ensemble 0.15*NN + 0.85*RF (held-out)', ens_preds)

    cm_rf = confusion_matrix(y_test, rf_preds, labels=[0, 1, 2])
    print("\nRandom Forest Confusion Matrix (held-out):")
    print(f"  {'':>8} pred_low  pred_med  pred_high")
    for i, label in enumerate(['low', 'medium', 'high']):
        print(f"  {label:>8}  {cm_rf[i][0]:>7}  {cm_rf[i][1]:>8}  {cm_rf[i][2]:>9}")

    return {
        'test_size': float(test_size),
        'seed': int(seed),
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'neural_network': nn_m,
        'random_forest': rf_m,
        'ensemble': ens_m,
        'rf_confusion_matrix': cm_rf.tolist(),
    }


def full_evaluation():
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    dataset = load_and_clean()
    X, y, scaler, brand_stats = prepare_data(dataset)

    # Baselines
    maj = baseline_majority(y)
    rnd = baseline_random(y)
    print(f"\nBaseline – Majority class: {maj:.4f}")
    print(f"Baseline – Random:         {rnd:.4f}")

    # Cross-validated evaluation
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros_like(y)
    all_probs = np.zeros((len(y), 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = train_model(X[train_idx], y[train_idx], epochs=150)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X[val_idx], dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
            preds = logits.argmax(1).numpy()
        all_preds[val_idx] = preds
        all_probs[val_idx] = probs
        acc = (preds == y[val_idx]).mean()
        print(f"  Fold {fold+1}: accuracy={acc:.4f}")

    cv_acc = accuracy_score(y, all_preds)
    cv_f1 = f1_score(y, all_preds, average='macro')
    print(f"\nCV Accuracy:  {cv_acc:.4f}")
    print(f"CV F1 (macro): {cv_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, all_preds, target_names=['low', 'medium', 'high']))

    cm = confusion_matrix(y, all_preds, labels=[0, 1, 2])
    print("Confusion Matrix:")
    print(f"  {'':>8} pred_low  pred_med  pred_high")
    for i, label in enumerate(['low', 'medium', 'high']):
        print(f"  {label:>8}  {cm[i][0]:>7}  {cm[i][1]:>8}  {cm[i][2]:>9}")

    # Per-brand analysis
    print("\n--- Per-Brand Analysis ---")
    features_df, targets_df = build_dataframe(dataset)
    for brand in brand_stats:
        mask = features_df[f'brand_{brand}'] == 1.0 if f'brand_{brand}' in features_df.columns else None
        if mask is None or mask.sum() == 0:
            continue
        idxs = np.where(mask.values)[0]
        brand_acc = (all_preds[idxs] == y[idxs]).mean()
        print(f"  {brand:>25}: n={len(idxs):3d}  accuracy={brand_acc:.4f}")

    # Failure analysis
    print("\n--- Failure Analysis ---")
    misclassified = np.where(all_preds != y)[0]
    print(f"  Total misclassified: {len(misclassified)} / {len(y)} "
          f"({len(misclassified)/len(y)*100:.1f}%)")

    if len(misclassified) > 0:
        print("\n  Sample misclassifications:")
        for idx in misclassified[:5]:
            data = dataset[idx].get('data', dataset[idx])
            brand = data.get('profile_stats', {}).get('username', '?')
            er = data.get('engagements', {}).get('engagement_rate', 0)
            true_label = INV_LABEL_MAP[y[idx]]
            pred_label = INV_LABEL_MAP[all_preds[idx]]
            caption = data.get('metadata_content', {}).get('caption', '')[:60]
            # Strip non-ASCII to avoid Windows cp1252 console encoding errors
            caption_ascii = caption.encode('ascii', 'replace').decode('ascii')
            print(f"    [{brand}] ER={er:.2f}  true={true_label}  pred={pred_label}  "
                  f'caption="{caption_ascii}..."')

    # Save evaluation results
    results = {
        'baselines': {'majority': maj, 'random': rnd},
        'cv_accuracy': float(cv_acc),
        'cv_f1_macro': float(cv_f1),
        'confusion_matrix': cm.tolist(),
        'per_brand': {},
    }
    for brand in brand_stats:
        col = f'brand_{brand}'
        if col in features_df.columns:
            mask = features_df[col] == 1.0
            idxs = np.where(mask.values)[0]
            if len(idxs) > 0:
                results['per_brand'][brand] = {
                    'count': int(len(idxs)),
                    'accuracy': float((all_preds[idxs] == y[idxs]).mean()),
                }

    # Held-out test set evaluation (honest generalisation estimate)
    held_out = held_out_evaluation(X, y, test_size=0.2, seed=42)
    results['held_out_test'] = held_out

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to {ARTIFACTS_DIR}/evaluation_results.json")


if __name__ == '__main__':
    full_evaluation()
