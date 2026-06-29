"""
=============================================================================
Train Model — Multi-Class Malware Category Classifier
=============================================================================
Trains a Random Forest on the merged CIC-MalMem2022 output files.
Uses Stratified K-Fold CV for honest evaluation (no 100% scores).

Output artefacts saved to models/v2/:
    rf_model.pkl          — trained Random Forest
    scaler.pkl            — StandardScaler (for inference)
    feature_names.json    — ordered list of feature names
    metadata.json         — class names, benign index, n_classes
    label_encoder.pkl     — LabelEncoder (int ↔ class name)
=============================================================================
"""

import os
import sys
import json
import warnings

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ── Resolve paths ─────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_V2_DIR   = os.path.join(_BASE_DIR, "models", "v2")

sys.path.insert(0, _SRC_DIR)
from data_preprocessing import load_and_preprocess_data


def train(dataset_dir: str, model_dir: str = _V2_DIR):
    print("=" * 60)
    print("  Malware Detection — Multi-Class Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading & preprocessing dataset...")
    X, y_str = load_and_preprocess_data(dataset_dir)

    # ── 2. Encode labels ───────────────────────────────────────────────────────
    print("\n[2/5] Encoding class labels...")
    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    class_names = list(le.classes_)
    benign_idx  = class_names.index("Benign") if "Benign" in class_names else 0
    print(f"  Classes: {class_names}")

    # ── 3. Scale features ─────────────────────────────────────────────────────
    print("\n[3/5] Fitting StandardScaler...")
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 4. Cross-validation (5-fold) ──────────────────────────────────────────
    print("\n[4/5] Running 5-Fold Stratified Cross-Validation...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,          # restrict depth → prevents overfit / 100% accuracy
        min_samples_leaf=4,    # requires at least 4 samples per leaf node
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        rf, X_scaled, y, cv=cv,
        scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        return_train_score=True,
    )

    print("\n  -- Cross-Validation Results (5 folds) --")
    print(f"  Train Accuracy : {cv_results['train_accuracy'].mean():.4f}  ± {cv_results['train_accuracy'].std():.4f}")
    print(f"  Val   Accuracy : {cv_results['test_accuracy'].mean():.4f}  ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  Val   F1-Macro : {cv_results['test_f1_macro'].mean():.4f}  ± {cv_results['test_f1_macro'].std():.4f}")
    print(f"  Val   Precision: {cv_results['test_precision_macro'].mean():.4f}  ± {cv_results['test_precision_macro'].std():.4f}")
    print(f"  Val   Recall   : {cv_results['test_recall_macro'].mean():.4f}  ± {cv_results['test_recall_macro'].std():.4f}")

    # ── 5. Final model on full dataset ────────────────────────────────────────
    print("\n[5/5] Training final model on full dataset & saving artefacts...")
    rf.fit(X_scaled, y)

    # Final hold-out eval (20%)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.20, stratify=y, random_state=42
    )
    rf_eval = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=4,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    rf_eval.fit(X_tr, y_tr)
    y_pred = rf_eval.predict(X_te)

    print("\n  -- Hold-Out Test Set (20%) Results --")
    acc = accuracy_score(y_te, y_pred)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=class_names, zero_division=0))

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    feature_names = list(X.columns)

    joblib.dump(rf, os.path.join(model_dir, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

    with open(os.path.join(model_dir, "feature_names.json"), "w") as fh:
        json.dump(feature_names, fh)

    metadata = {
        "class_names"     : class_names,
        "n_classes"       : len(class_names),
        "benign_class_idx": benign_idx,
        "model_type"      : "RandomForest_MultiClass_v2",
        "cv_val_accuracy" : round(float(cv_results["test_accuracy"].mean()), 4),
        "cv_val_f1_macro" : round(float(cv_results["test_f1_macro"].mean()), 4),
        "holdout_accuracy": round(float(acc), 4),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\n  Artefacts saved to: {model_dir}")
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    dataset_path = os.path.join(_BASE_DIR, "dataset")
    train(dataset_path)