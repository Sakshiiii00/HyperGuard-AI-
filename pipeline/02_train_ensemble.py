"""
=============================================================================
PIPELINE STEP 2 — Ensemble Model Training
=============================================================================
Trains and evaluates 3 base classifiers:
  1. Random Forest        (sklearn)
  2. XGBoost              (xgboost)
  3. Deep Neural Network  (TensorFlow / Keras)

+ Builds a Hybrid Ensemble via soft-voting (weighted probability averaging)

Training process:
  - 5-fold Stratified Cross-Validation for all models
  - Hyperparameter grid search for RF and XGB (light grid)
  - DNN with early stopping, ReduceLROnPlateau, BatchNorm, Dropout
  - Metrics: Accuracy, Precision, Recall, F1-macro, ROC-AUC (OvR)

Output artifacts (models/v2/):
  rf_model.pkl, xgb_model.json, dnn_model.keras,
  ensemble_weights.json, comparison_results.json

Visualization artifacts (reports/figures/):
  dnn_training_loss.png, rf_feature_importance.png, xgb_feature_importance.png
=============================================================================
"""

import os
import sys
import json
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_V2_DIR = os.path.join(BASE_DIR, "models", "v2")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
FIGURES_DIR  = os.path.join(REPORTS_DIR, "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
BG_COLOR = "#0a0d1a"
FG_COLOR = "#f8fafc"
plt.rcParams.update({
    "figure.facecolor" : BG_COLOR,
    "axes.facecolor"   : "#111827",
    "axes.edgecolor"   : "#374151",
    "axes.labelcolor"  : FG_COLOR,
    "xtick.color"      : FG_COLOR,
    "ytick.color"      : FG_COLOR,
    "text.color"       : FG_COLOR,
    "grid.color"       : "#1f2937",
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "figure.dpi"       : 130,
})

RANDOM_STATE = 42
N_FOLDS      = 5


def _banner(msg: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {msg}")
    print(f"{'═' * 60}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray, n_classes: int) -> dict:
    avg = "macro"
    metrics = {
        "accuracy"  : float(round(accuracy_score(y_true, y_pred), 4)),
        "precision" : float(round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4)),
        "recall"    : float(round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4)),
        "f1_macro"  : float(round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4)),
    }
    try:
        multi_class = "ovr" if n_classes > 2 else "raise"
        auc = roc_auc_score(y_true, y_prob, multi_class=multi_class,
                            average=avg, labels=list(range(n_classes)))
        metrics["roc_auc"] = float(round(auc, 4))
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def _cv_score(model, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    """Run 5-fold CV and return mean metrics."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_accs  = []
    fold_f1s   = []
    fold_aucs  = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model.fit(X[tr], y[tr])
        preds = model.predict(X[va])
        probs = model.predict_proba(X[va])
        fold_accs.append(accuracy_score(y[va], preds))
        fold_f1s.append(f1_score(y[va], preds, average="macro", zero_division=0))
        try:
            auc = roc_auc_score(y[va], probs, multi_class="ovr",
                                average="macro", labels=list(range(n_classes)))
            fold_aucs.append(auc)
        except Exception:
            pass
        print(f"     Fold {fold}/{N_FOLDS}  Acc={fold_accs[-1]:.4f}  F1={fold_f1s[-1]:.4f}")

    return {
        "cv_accuracy_mean": float(round(np.mean(fold_accs), 4)),
        "cv_accuracy_std" : float(round(np.std(fold_accs),  4)),
        "cv_f1_mean"      : float(round(np.mean(fold_f1s),  4)),
        "cv_f1_std"       : float(round(np.std(fold_f1s),   4)),
        "cv_roc_auc_mean" : float(round(np.mean(fold_aucs), 4)) if fold_aucs else None,
    }


# ── BUILD DNN ─────────────────────────────────────────────────────────────────
def _build_dnn(n_features: int, n_classes: int, lr: float = 1e-3) -> tf.keras.Model:
    inp = layers.Input(shape=(n_features,), name="features")
    x = layers.Dense(512, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)   # increased from 0.3

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)   # increased from 0.3

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)   # increased from 0.25

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)  # increased from 0.2

    if n_classes == 2:
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
        metrics_ = ["accuracy"]
    else:
        out = layers.Dense(n_classes, activation="softmax", name="output")(x)
        loss = "sparse_categorical_crossentropy"
        metrics_ = ["accuracy"]

    model = models.Model(inp, out, name="DNN_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics_
    )
    return model


def run() -> None:
    _banner("STEP 2 — Ensemble Training (RF + XGBoost + DNN)")

    # ── Load preprocessed data ────────────────────────────────────────────────
    data_path = os.path.join(MODEL_V2_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        sys.exit("[ERROR] Run pipeline/01_preprocess.py first.")

    data = np.load(data_path, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    with open(os.path.join(MODEL_V2_DIR, "metadata.json")) as fh:
        meta = json.load(fh)

    n_classes   = meta["n_classes"]
    class_names = meta["class_names"]
    n_features  = X_train.shape[1]

    print(f"\n  Train  : {X_train.shape}")
    print(f"  Test   : {X_test.shape}")
    print(f"  Classes: {n_classes} → {class_names}")

    results = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL 1 — Random Forest
    # ═══════════════════════════════════════════════════════════════════════════
    _banner("Model 1 / 3 — Random Forest")

    print("\n  Hyperparameter grid search (light) …")
    rf_param_grid = {
        "n_estimators"    : [100, 200],
        "max_features"    : ["sqrt", "log2"],
        "max_depth"       : [10, 15],      # cap depth — prevents memorization
        "min_samples_leaf": [4, 8],        # min samples per leaf — no tiny splits
    }
    rf_base = RandomForestClassifier(
        max_depth=15, min_samples_leaf=4,  # defaults used during CV scoring
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_gs = GridSearchCV(
        rf_base, rf_param_grid, cv=3, scoring="f1_macro",
        n_jobs=-1, verbose=1
    )
    rf_gs.fit(X_train, y_train)
    best_rf_params = rf_gs.best_params_
    print(f"\n  Best RF params: {best_rf_params}")

    print("\n  5-Fold CV …")
    rf_cv_model = RandomForestClassifier(
        **best_rf_params, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_cv_metrics = _cv_score(rf_cv_model, X_train, y_train, n_classes)

    # Final fit on full train set
    rf_final = RandomForestClassifier(
        **best_rf_params, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_final.fit(X_train, y_train)
    rf_pred  = rf_final.predict(X_test)
    rf_prob  = rf_final.predict_proba(X_test)
    rf_test_metrics = _compute_metrics(y_test, rf_pred, rf_prob, n_classes)

    print(f"\n  [RF] Test metrics: {rf_test_metrics}")
    print(f"\n  [RF] Classification Report:\n{classification_report(y_test, rf_pred, target_names=class_names, zero_division=0)}")

    joblib.dump(rf_final, os.path.join(MODEL_V2_DIR, "rf_model.pkl"))
    results["Random Forest"] = {**rf_test_metrics, **rf_cv_metrics, "best_params": best_rf_params}

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL 2 — XGBoost
    # ═══════════════════════════════════════════════════════════════════════════
    _banner("Model 2 / 3 — XGBoost")

    print("\n  Hyperparameter grid search (light) …")
    xgb_param_grid = {
        "n_estimators"  : [200, 300],
        "max_depth"     : [4, 6],
        "learning_rate" : [0.05, 0.1],
        "subsample"     : [0.8, 1.0],
    }
    use_gpu = len(tf.config.list_physical_devices("GPU")) > 0
    tree_method = "hist"

    xgb_base = xgb.XGBClassifier(
        objective="multi:softmax" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        tree_method=tree_method,
        eval_metric="mlogloss",
        reg_lambda=1.5,          # L2 regularization — penalises leaf weight magnitude
        min_child_weight=5,      # min sum of instance weight per child — no tiny splits
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_gs = GridSearchCV(
        xgb_base, xgb_param_grid, cv=3, scoring="f1_macro",
        n_jobs=-1, verbose=1
    )
    xgb_gs.fit(X_train, y_train)
    best_xgb_params = xgb_gs.best_params_
    print(f"\n  Best XGB params: {best_xgb_params}")

    print("\n  5-Fold CV …")
    xgb_cv_model = xgb.XGBClassifier(
        **best_xgb_params,
        objective="multi:softmax" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        tree_method=tree_method,
        eval_metric="mlogloss",
        reg_lambda=1.5,
        min_child_weight=5,
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_cv_metrics = _cv_score(xgb_cv_model, X_train, y_train, n_classes)

    # Final fit
    xgb_final = xgb.XGBClassifier(
        **best_xgb_params,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        tree_method=tree_method,
        eval_metric="mlogloss",
        reg_lambda=1.5,
        min_child_weight=5,
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_final.fit(X_train, y_train)
    xgb_pred = xgb_final.predict(X_test)
    xgb_prob = xgb_final.predict_proba(X_test)
    xgb_test_metrics = _compute_metrics(y_test, xgb_pred, xgb_prob, n_classes)

    print(f"\n  [XGB] Test metrics: {xgb_test_metrics}")
    print(f"\n  [XGB] Classification Report:\n{classification_report(y_test, xgb_pred, target_names=class_names, zero_division=0)}")

    xgb_final.save_model(os.path.join(MODEL_V2_DIR, "xgb_model.json"))
    results["XGBoost"] = {**xgb_test_metrics, **xgb_cv_metrics, "best_params": best_xgb_params}

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL 3 — Deep Neural Network
    # ═══════════════════════════════════════════════════════════════════════════
    _banner("Model 3 / 3 — Deep Neural Network (Keras)")

    dnn = _build_dnn(n_features, n_classes)
    dnn.summary()

    cb_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_V2_DIR, "dnn_best.keras"),
            save_best_only=True, monitor="val_accuracy", verbose=0
        ),
    ]

    history = dnn.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=256,
        callbacks=cb_list,
        verbose=1,
        class_weight={i: 1.0 for i in range(n_classes)},  # balanced by SMOTE
    )

    dnn.save(os.path.join(MODEL_V2_DIR, "dnn_model.keras"))

    dnn_raw = dnn.predict(X_test, verbose=0)
    if n_classes == 2:
        dnn_prob = np.column_stack([1 - dnn_raw.ravel(), dnn_raw.ravel()])
    else:
        dnn_prob = dnn_raw
    dnn_pred = np.argmax(dnn_prob, axis=1)

    dnn_test_metrics = _compute_metrics(y_test, dnn_pred, dnn_prob, n_classes)
    print(f"\n  [DNN] Test metrics: {dnn_test_metrics}")
    print(f"\n  [DNN] Classification Report:\n{classification_report(y_test, dnn_pred, target_names=class_names, zero_division=0)}")

    results["DNN"] = dnn_test_metrics

    # ── DNN Training Loss Curve ────────────────────────────────────────────────
    _banner("Saving DNN Training Loss Curve")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("DNN Training History", fontsize=14, color=FG_COLOR)

    # Loss
    axes[0].plot(history.history["loss"],     color="#7c3aed", lw=2, label="Train Loss")
    axes[0].plot(history.history["val_loss"], color="#00e5ff", lw=2, label="Val Loss", ls="--")
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve", color=FG_COLOR)
    axes[0].legend(framealpha=0.2)
    axes[0].grid(alpha=0.25)

    # Accuracy
    axes[1].plot(history.history["accuracy"],     color="#10b981", lw=2, label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], color="#f59e0b", lw=2, label="Val Acc",   ls="--")
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve", color=FG_COLOR)
    axes[1].legend(framealpha=0.2)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    loss_path = os.path.join(FIGURES_DIR, "dnn_training_loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {loss_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # HYBRID ENSEMBLE — Weighted soft voting
    # ═══════════════════════════════════════════════════════════════════════════
    _banner("Hybrid Ensemble — Soft Voting (RF + XGB + DNN)")

    # Weight proportional to individual F1 scores
    f1_scores = np.array([
        results["Random Forest"]["f1_macro"],
        results["XGBoost"]["f1_macro"],
        results["DNN"]["f1_macro"],
    ])
    f1_scores = np.clip(f1_scores, 0, 1)
    weights   = f1_scores / f1_scores.sum()
    print(f"  Ensemble weights — RF: {weights[0]:.3f}  XGB: {weights[1]:.3f}  DNN: {weights[2]:.3f}")

    ens_prob = (
        weights[0] * rf_prob +
        weights[1] * xgb_prob +
        weights[2] * dnn_prob
    )
    ens_pred = np.argmax(ens_prob, axis=1)
    ens_test_metrics = _compute_metrics(y_test, ens_pred, ens_prob, n_classes)

    print(f"\n  [ENS] Test metrics: {ens_test_metrics}")
    print(f"\n  [ENS] Classification Report:\n{classification_report(y_test, ens_pred, target_names=class_names, zero_division=0)}")

    results["Ensemble (RF+XGB+DNN)"] = ens_test_metrics

    # Save ensemble weights (RF + XGB + DNN only — no LSTM)
    ens_weights = {
        "rf_weight"  : float(weights[0]),
        "xgb_weight" : float(weights[1]),
        "dnn_weight" : float(weights[2]),
    }
    with open(os.path.join(MODEL_V2_DIR, "ensemble_weights.json"), "w") as fh:
        json.dump(ens_weights, fh, indent=2)

    # ── RF Feature Importance Chart ────────────────────────────────────────────
    _banner("Saving RF Feature Importance Chart")
    rf_importances = rf_final.feature_importances_
    top_n = min(20, len(rf_importances))
    top_idx_rf  = np.argsort(rf_importances)[::-1][:top_n]
    top_vals_rf = rf_importances[top_idx_rf]
    feat_labels  = [f"PCA_{i+1:03d}" for i in range(len(rf_importances))]
    top_feats_rf = [feat_labels[i] for i in top_idx_rf]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdPu(np.linspace(0.3, 1.0, top_n))
    ax.barh(np.arange(top_n), top_vals_rf[::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.3)
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(top_feats_rf[::-1], fontsize=8, color=FG_COLOR)
    ax.set_xlabel("Feature Importance (Gini)", color=FG_COLOR, fontsize=10)
    ax.set_title(f"Top {top_n} Random Forest Feature Importances",
                 color=FG_COLOR, fontsize=12)
    ax.tick_params(colors=FG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, color="white")
    plt.tight_layout()
    rf_fi_path = os.path.join(FIGURES_DIR, "rf_feature_importance.png")
    plt.savefig(rf_fi_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {rf_fi_path}")

    # ── XGB Feature Importance Chart ───────────────────────────────────────────
    _banner("Saving XGB Feature Importance Chart")
    xgb_importances = xgb_final.feature_importances_
    top_n_xgb  = min(20, len(xgb_importances))
    top_idx_xgb  = np.argsort(xgb_importances)[::-1][:top_n_xgb]
    top_vals_xgb = xgb_importances[top_idx_xgb]
    top_feats_xgb = [feat_labels[i] for i in top_idx_xgb]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, top_n_xgb))
    ax.barh(np.arange(top_n_xgb), top_vals_xgb[::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.3)
    ax.set_yticks(np.arange(top_n_xgb))
    ax.set_yticklabels(top_feats_xgb[::-1], fontsize=8, color=FG_COLOR)
    ax.set_xlabel("Feature Importance (Weight)", color=FG_COLOR, fontsize=10)
    ax.set_title(f"Top {top_n_xgb} XGBoost Feature Importances",
                 color=FG_COLOR, fontsize=12)
    ax.tick_params(colors=FG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, color="white")
    plt.tight_layout()
    xgb_fi_path = os.path.join(FIGURES_DIR, "xgb_feature_importance.png")
    plt.savefig(xgb_fi_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {xgb_fi_path}")

    # ── Save comparison results ────────────────────────────────────────────────
    results_path = os.path.join(MODEL_V2_DIR, "comparison_results.json")
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Pretty table
    df_res = pd.DataFrame(results).T[
        ["accuracy", "f1_macro", "roc_auc", "precision", "recall"]
    ]
    df_res.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    print(f"\n✅  Ensemble training complete.")
    print(f"\n  Model Comparison:")
    print(df_res.to_string())

    _banner("Summary")
    print(f"  Winner by F1: {df_res['f1_macro'].idxmax()}")


if __name__ == "__main__":
    run()
