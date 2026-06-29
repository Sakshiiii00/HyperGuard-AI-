"""
=============================================================================
PIPELINE STEP 5 — Comprehensive Evaluation & Visualization
=============================================================================
Loads all trained models and generates research-grade evaluation artifacts:

  ✦ Confusion matrices (per model, normalized)
  ✦ ROC curves (all models overlaid)
  ✦ Precision–Recall curves
  ✦ Model comparison table (CSV + JSON)
  ✦ Ablation study table
  ✦ Autoencoder reconstruction error histogram

Output directory: reports/figures/
=============================================================================
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_V2_DIR = os.path.join(BASE_DIR, "models", "v2")
FIGURES_DIR  = os.path.join(BASE_DIR, "reports", "figures")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
PALETTE  = ["#7c3aed", "#00e5ff", "#ef4444", "#10b981", "#f59e0b"]
BG_COLOR = "#0a0d1a"
FG_COLOR = "#f8fafc"
plt.rcParams.update({
    "figure.facecolor"   : BG_COLOR,
    "axes.facecolor"     : "#111827",
    "axes.edgecolor"     : "#374151",
    "axes.labelcolor"    : FG_COLOR,
    "xtick.color"        : FG_COLOR,
    "ytick.color"        : FG_COLOR,
    "text.color"         : FG_COLOR,
    "grid.color"         : "#1f2937",
    "font.family"        : "DejaVu Sans",
    "font.size"          : 10,
    "figure.dpi"         : 130,
})


def _banner(msg: str) -> None:
    print(f"\n{'═' * 60}\n  {msg}\n{'═' * 60}")


def _load_models(meta: dict):
    """Load all available model artifacts."""
    mdls = {}

    rf_path = os.path.join(MODEL_V2_DIR, "rf_model.pkl")
    if os.path.exists(rf_path):
        mdls["Random Forest"] = joblib.load(rf_path)
        print("  ✔  Random Forest loaded")

    xgb_path = os.path.join(MODEL_V2_DIR, "xgb_model.json")
    if os.path.exists(xgb_path):
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(xgb_path)
        mdls["XGBoost"] = m
        print("  ✔  XGBoost loaded")

    dnn_path = os.path.join(MODEL_V2_DIR, "dnn_model.keras")
    if os.path.exists(dnn_path):
        mdls["DNN"] = tf.keras.models.load_model(dnn_path)
        print("  ✔  DNN loaded")

    ae_path = os.path.join(MODEL_V2_DIR, "autoencoder.keras")
    if os.path.exists(ae_path):
        mdls["_autoencoder"] = tf.keras.models.load_model(ae_path)
        print("  ✔  Autoencoder loaded")

    return mdls


def _get_probs(model, X: np.ndarray,
               model_name: str, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, y_prob) for any model type."""
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        pred = np.argmax(prob, axis=1)
        return pred, prob

    # For Keras models
    raw = model.predict(X, verbose=0)
    if hasattr(raw, "ravel") and (raw.ndim == 1 or raw.shape[1] == 1):
        prob = np.column_stack([1 - raw.ravel(), raw.ravel()])
    else:
        prob = raw

    pred = np.argmax(prob, axis=1)
    return pred, prob


def _plot_confusion_matrix(y_true, y_pred, class_names, model_name, ax):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="RdPu",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar=True, linewidths=0.5, linecolor="#1f2937",
        annot_kws={"size": 8}
    )
    ax.set_title(f"{model_name}", color=FG_COLOR, fontsize=11, pad=8)
    ax.set_xlabel("Predicted", color=FG_COLOR)
    ax.set_ylabel("True",      color=FG_COLOR)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)


def run() -> None:
    _banner("STEP 5 — Evaluation & Visualization")

    # ── Load data ──────────────────────────────────────────────────────────────
    data_path = os.path.join(MODEL_V2_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        sys.exit("[ERROR] Run pipeline/01_preprocess.py first.")

    data     = np.load(data_path, allow_pickle=True)
    X_test   = data["X_test"]
    y_test   = data["y_test"]

    with open(os.path.join(MODEL_V2_DIR, "metadata.json")) as fh:
        meta = json.load(fh)

    n_classes   = meta["n_classes"]
    class_names = meta["class_names"]

    # ── Load models ────────────────────────────────────────────────────────────
    print("\n  Loading models …")
    mdls = _load_models(meta)

    # ── Per-model metrics ──────────────────────────────────────────────────────
    model_metrics = {}
    model_preds   = {}
    model_probs   = {}

    for name, model in mdls.items():
        if name.startswith("_"):
            continue
        print(f"\n  Evaluating: {name} …")
        y_pred, y_prob = _get_probs(model, X_test, name, n_classes)
        model_preds[name] = y_pred
        model_probs[name] = y_prob

        try:
            auc_val = roc_auc_score(
                y_test, y_prob, multi_class="ovr",
                average="macro", labels=list(range(n_classes))
            )
        except Exception:
            auc_val = None

        model_metrics[name] = {
            "Accuracy"  : round(accuracy_score(y_test, y_pred),                          4),
            "Precision" : round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "Recall"    : round(recall_score(y_test, y_pred, average="macro", zero_division=0),    4),
            "F1-macro"  : round(f1_score(y_test, y_pred, average="macro", zero_division=0),        4),
            "ROC-AUC"   : round(auc_val, 4) if auc_val else "N/A",
        }
        print(f"    {model_metrics[name]}")

    # ── HYBRID ENSEMBLE ────────────────────────────────────────────────────────
    ens_path = os.path.join(MODEL_V2_DIR, "ensemble_weights.json")
    if os.path.exists(ens_path):
        with open(ens_path) as fh:
            ens_w = json.load(fh)

        ens_prob = None
        wsum     = 0.0
        for nm, wk in [("Random Forest", "rf_weight"), ("XGBoost", "xgb_weight"),
                       ("DNN", "dnn_weight")]:
            if nm in model_probs and ens_w.get(wk, 0) > 0:
                w = ens_w[wk]
                ens_prob = model_probs[nm] * w if ens_prob is None else ens_prob + model_probs[nm] * w
                wsum += w

        if ens_prob is not None:
            ens_prob  = ens_prob / wsum
            ens_pred  = np.argmax(ens_prob, axis=1)
            model_preds["Hybrid Ensemble"] = ens_pred
            model_probs["Hybrid Ensemble"] = ens_prob
            try:
                auc_val = roc_auc_score(y_test, ens_prob, multi_class="ovr",
                                        average="macro", labels=list(range(n_classes)))
            except Exception:
                auc_val = None
            model_metrics["Hybrid Ensemble"] = {
                "Accuracy"  : round(accuracy_score(y_test, ens_pred), 4),
                "Precision" : round(precision_score(y_test, ens_pred, average="macro", zero_division=0), 4),
                "Recall"    : round(recall_score(y_test, ens_pred, average="macro", zero_division=0),    4),
                "F1-macro"  : round(f1_score(y_test, ens_pred, average="macro", zero_division=0),        4),
                "ROC-AUC"   : round(auc_val, 4) if auc_val else "N/A",
            }

    # ── Print + save comparison table ──────────────────────────────────────────
    df_cmp = pd.DataFrame(model_metrics).T
    print(f"\n{'─'*60}")
    print(f"  MODEL COMPARISON TABLE")
    print(f"{'─'*60}")
    print(df_cmp.to_string())
    df_cmp.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    # ── ABLATION STUDY ────────────────────────────────────────────────────────
    ablation = {}

    # A1: Baseline RF only
    if "Random Forest" in model_metrics:
        ablation["A1: Baseline RF"] = model_metrics["Random Forest"]

    # A2: RF + XGB (no DNN, no LSTM)
    if "Random Forest" in model_probs and "XGBoost" in model_probs:
        ab2_prob = (model_probs["Random Forest"] + model_probs["XGBoost"]) / 2
        ab2_pred = np.argmax(ab2_prob, axis=1)
        ablation["A2: RF+XGB"] = {
            "Accuracy": round(accuracy_score(y_test, ab2_pred), 4),
            "F1-macro": round(f1_score(y_test, ab2_pred, average="macro", zero_division=0), 4),
        }

    # A3: Full Ensemble (no LSTM)
    if "DNN" in model_probs and "Random Forest" in model_probs and "XGBoost" in model_probs:
        ab3_prob = (model_probs["Random Forest"] + model_probs["XGBoost"] + model_probs["DNN"]) / 3
        ab3_pred = np.argmax(ab3_prob, axis=1)
        ablation["A3: RF+XGB+DNN"] = {
            "Accuracy": round(accuracy_score(y_test, ab3_pred), 4),
            "F1-macro": round(f1_score(y_test, ab3_pred, average="macro", zero_division=0), 4),
        }

    # A4: Full Hybrid
    if "Hybrid Ensemble" in model_metrics:
        ablation["A4: Full Hybrid"] = {
            "Accuracy": model_metrics["Hybrid Ensemble"]["Accuracy"],
            "F1-macro": model_metrics["Hybrid Ensemble"]["F1-macro"],
        }

    df_abl = pd.DataFrame(ablation).T
    print(f"\n  ABLATION STUDY:")
    print(df_abl.to_string())
    df_abl.to_csv(os.path.join(REPORTS_DIR, "ablation_study.csv"))

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Confusion Matrices
    # ══════════════════════════════════════════════════════════════════════════
    _banner("Figure 1 — Confusion Matrices")
    plot_models = {k: v for k, v in model_preds.items()}
    n_models    = len(plot_models)
    n_cols      = min(3, n_models)
    n_rows      = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, color=FG_COLOR, y=1.01)
    axes_flat = np.array(axes).ravel()

    for i, (nm, y_pred) in enumerate(plot_models.items()):
        _plot_confusion_matrix(y_test, y_pred, class_names, nm, axes_flat[i])

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {cm_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — ROC Curves
    # ══════════════════════════════════════════════════════════════════════════
    _banner("Figure 2 — ROC Curves (OvR, Macro)")
    fig, ax = plt.subplots(figsize=(9, 7))

    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    for idx, (nm, prob) in enumerate(model_probs.items()):
        color = PALETTE[idx % len(PALETTE)]
        try:
            if n_classes == 2:
                fpr_, tpr_, _ = roc_curve(y_test, prob[:, 1])
                roc_auc_val   = auc(fpr_, tpr_)
                ax.plot(fpr_, tpr_, color=color, lw=2,
                        label=f"{nm}  (AUC={roc_auc_val:.3f})")
            else:
                for ci in range(n_classes):
                    fpr_, tpr_, _ = roc_curve(y_bin[:, ci], prob[:, ci])
                    roc_auc_val   = auc(fpr_, tpr_)
                # Plot macro-average
                all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, ci], prob[:, ci])[0]
                                                    for ci in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for ci in range(n_classes):
                    fpr_, tpr_, _ = roc_curve(y_bin[:, ci], prob[:, ci])
                    mean_tpr += np.interp(all_fpr, fpr_, tpr_)
                mean_tpr  /= n_classes
                macro_auc  = auc(all_fpr, mean_tpr)
                ax.plot(all_fpr, mean_tpr, color=color, lw=2,
                        label=f"{nm}  (macro AUC={macro_auc:.3f})")
        except Exception as e:
            print(f"    [WARN] ROC for {nm}: {e}")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random baseline")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, color=FG_COLOR)
    ax.legend(loc="lower right", framealpha=0.2, fontsize=9)
    ax.grid(True, alpha=0.3)

    roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {roc_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Model Comparison Bar Chart
    # ══════════════════════════════════════════════════════════════════════════
    _banner("Figure 3 — Model Comparison")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-macro"]
    df_plot = df_cmp[metrics_to_plot].apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 6))
    x    = np.arange(len(df_plot))
    w    = 0.18
    offsets = np.linspace(-1.5, 1.5, len(metrics_to_plot))

    for i, col in enumerate(metrics_to_plot):
        bars = ax.bar(x + offsets[i] * w, df_plot[col].values,
                      width=w, label=col, color=PALETTE[i], alpha=0.9,
                      edgecolor="white", linewidth=0.4)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, color=FG_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, color=FG_COLOR)
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(axis="y", alpha=0.3)

    cmp_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(cmp_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {cmp_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 4 — Autoencoder Anomaly Score Histogram
    # ══════════════════════════════════════════════════════════════════════════
    ae_thresh_path = os.path.join(MODEL_V2_DIR, "ae_threshold.json")
    ae_model_path  = os.path.join(MODEL_V2_DIR, "autoencoder.keras")
    if os.path.exists(ae_thresh_path) and os.path.exists(ae_model_path):
        _banner("Figure 4 — Autoencoder Reconstruction Error")
        with open(ae_thresh_path) as fh:
            ae_cfg = json.load(fh)
        threshold = ae_cfg["anomaly_threshold"]

        ae_model = tf.keras.models.load_model(ae_model_path)
        X_recon  = ae_model.predict(X_test, verbose=0)
        mse_all  = np.mean((X_test - X_recon) ** 2, axis=1)

        benign_idx = meta["benign_class_idx"]
        ben_mse = mse_all[y_test == benign_idx]
        mal_mse = mse_all[y_test != benign_idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(ben_mse, bins=60, alpha=0.7, color="#10b981", label="Benign",    density=True)
        ax.hist(mal_mse, bins=60, alpha=0.6, color="#ef4444", label="Malware",   density=True)
        ax.axvline(threshold, color="#f59e0b", lw=2, ls="--",
                   label=f"Threshold = {threshold:.4f}")
        ax.set_xlabel("Reconstruction Error (MSE)",   fontsize=11)
        ax.set_ylabel("Density",                       fontsize=11)
        ax.set_title("Autoencoder Anomaly Score Distribution", fontsize=13, color=FG_COLOR)
        ax.legend(fontsize=9, framealpha=0.2)
        ax.grid(alpha=0.3)

        ae_path_fig = os.path.join(FIGURES_DIR, "autoencoder_anomaly_scores.png")
        plt.tight_layout()
        plt.savefig(ae_path_fig, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {ae_path_fig}")

    _banner("Evaluation Complete ✅")
    print(f"  All figures in : {FIGURES_DIR}")
    print(f"  Comparison CSV : {os.path.join(REPORTS_DIR, 'model_comparison.csv')}")
    print(f"  Ablation CSV   : {os.path.join(REPORTS_DIR, 'ablation_study.csv')}")


if __name__ == "__main__":
    run()
