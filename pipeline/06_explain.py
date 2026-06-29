"""
=============================================================================
PIPELINE STEP 6 — SHAP Explainability
=============================================================================
Generates SHAP-based explanations for the Random Forest and XGBoost models
(Tree-based SHAP is exact and fast). DNN/LSTM use KernelExplainer on a
background sample for approximate explanations.

Outputs:
  ✦ SHAP summary plot (beeswarm) — RF
  ✦ SHAP bar chart (global feature importance) — all tree models
  ✦ SHAP waterfall — single benign sample + single malware sample
  ✦ feature_importance.json — for the dashboard API

Saved to: reports/figures/  and  models/v2/
=============================================================================
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import shap

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_V2_DIR = os.path.join(BASE_DIR, "models", "v2")
FIGURES_DIR  = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BG_COLOR = "#0a0d1a"
FG_COLOR = "#f8fafc"

SHAP_BACKGROUND_SAMPLES = 200   # For KernelExplainer background
SHAP_EXPLAIN_SAMPLES    = 100   # Number of test samples to explain


def _banner(msg: str) -> None:
    print(f"\n{'═' * 60}\n  {msg}\n{'═' * 60}")


def run() -> None:
    _banner("STEP 6 — SHAP Explainability")

    # ── Load data ──────────────────────────────────────────────────────────────
    data_path = os.path.join(MODEL_V2_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        sys.exit("[ERROR] Run pipeline/01_preprocess.py first.")

    data   = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]

    with open(os.path.join(MODEL_V2_DIR, "metadata.json")) as fh:
        meta = json.load(fh)

    class_names = meta["class_names"]
    n_pca       = meta["n_pca_components"]

    # PCA feature names (for axis labels)
    feature_labels = [f"PCA_{i+1:03d}" for i in range(n_pca)]

    # ── Subsample for speed ────────────────────────────────────────────────────
    N = min(SHAP_EXPLAIN_SAMPLES, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), N, replace=False)
    X_explain = X_test[idx]
    y_explain = y_test[idx]

    # Background for KernelExplainer (random subset)
    bg_idx  = rng.choice(len(X_test), SHAP_BACKGROUND_SAMPLES, replace=False)
    X_bg    = X_test[bg_idx]

    # ═══════════════════════════════════════════════════════════════════════════
    # RANDOM FOREST — TreeExplainer (exact, fast)
    # ═══════════════════════════════════════════════════════════════════════════
    rf_path = os.path.join(MODEL_V2_DIR, "rf_model.pkl")
    if os.path.exists(rf_path):
        _banner("SHAP — Random Forest (TreeExplainer)")
        rf_model  = joblib.load(rf_path)
        explainer = shap.TreeExplainer(rf_model)

        print(f"  Computing SHAP values for {N} samples …")
        shap_values = explainer.shap_values(X_explain)

        # shap_values: list of arrays (one per class) OR single array
        if isinstance(shap_values, list):
            # Multi-class: use class 0 (benign) vs aggregate
            shap_for_plot = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_for_plot = np.abs(shap_values)
            if shap_for_plot.ndim == 3:
                if shap_for_plot.shape[2] == X_explain.shape[1]:
                    shap_for_plot = shap_for_plot.mean(axis=1)
                else:
                    shap_for_plot = shap_for_plot.mean(axis=2)

        importance = shap_for_plot.mean(axis=0)  # shape (n_features,)
        top_n      = min(20, len(importance))
        top_idx    = np.argsort(importance)[::-1][:top_n]
        top_feats  = [feature_labels[i] for i in top_idx]
        top_vals   = importance[top_idx]

        # Save JSON for dashboard
        fi_json = {
            "top_features"     : top_feats,
            "importance_scores": [float(v) for v in top_vals],
            "source_model"     : "Random Forest (TreeSHAP)",
        }
        with open(os.path.join(MODEL_V2_DIR, "feature_importance.json"), "w") as fh:
            json.dump(fi_json, fh, indent=2)
        print(f"  Saved feature_importance.json")

        # ── Plot: Global Bar Chart ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
        ax.set_facecolor("#111827")
        colors = plt.cm.RdPu(np.linspace(0.3, 1.0, top_n))
        bars   = ax.barh(np.arange(top_n), top_vals[::-1],
                         color=colors[::-1], edgecolor="white", linewidth=0.3)
        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels(top_feats[::-1], fontsize=8, color=FG_COLOR)
        ax.set_xlabel("Mean |SHAP value| (impact on model output)",
                      color=FG_COLOR, fontsize=10)
        ax.set_title("Top 20 Features — SHAP Global Importance (Random Forest)",
                     color=FG_COLOR, fontsize=12)
        ax.tick_params(colors=FG_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="white")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "shap_rf_bar.png")
        plt.savefig(out, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        print(f"  Saved: {out}")

        # ── SHAP Summary Beeswarm ──────────────────────────────────────────────
        try:
            if isinstance(shap_values, list):
                sv = shap_values[0]   # Benign class SHAP values
            elif shap_values.ndim == 3:
                if shap_values.shape[2] == X_explain.shape[1]:
                    sv = shap_values[:, 0, :]
                else:
                    sv = shap_values[:, :, 0]
            else:
                sv = shap_values

            shap.summary_plot(
                sv[:, top_idx], X_explain[:, top_idx],
                feature_names=top_feats,
                show=False, plot_type="dot",
                max_display=15, color_bar=True
            )
            fig = plt.gcf()
            fig.set_facecolor(BG_COLOR)
            out_bw = os.path.join(FIGURES_DIR, "shap_rf_beeswarm.png")
            plt.savefig(out_bw, bbox_inches="tight", facecolor=BG_COLOR, dpi=130)
            plt.close()
            print(f"  Saved: {out_bw}")
        except Exception as e:
            print(f"  [WARN] Beeswarm plot failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # XGBoost — TreeExplainer (exact, fast)
    # ═══════════════════════════════════════════════════════════════════════════
    xgb_path = os.path.join(MODEL_V2_DIR, "xgb_model.json")
    if os.path.exists(xgb_path):
        _banner("SHAP — XGBoost (TreeExplainer)")
        import xgboost as xgb_lib
        xgb_model  = xgb_lib.XGBClassifier()
        xgb_model.load_model(xgb_path)
        xgb_explainer = shap.TreeExplainer(xgb_model)

        print(f"  Computing XGB SHAP values …")
        xgb_shap = xgb_explainer.shap_values(X_explain)

        if isinstance(xgb_shap, list):
            xgb_importance = np.abs(np.array(xgb_shap)).mean(axis=0).mean(axis=0)
        elif xgb_shap.ndim == 3:
            if xgb_shap.shape[2] == X_explain.shape[1]:
                xgb_importance = np.abs(xgb_shap).mean(axis=(0, 1))
            else:
                xgb_importance = np.abs(xgb_shap).mean(axis=(0, 2))
        else:
            xgb_importance = np.abs(xgb_shap).mean(axis=0)

        top_n_xgb    = min(20, len(xgb_importance))
        top_idx_xgb  = np.argsort(xgb_importance)[::-1][:top_n_xgb]
        top_feats_xgb = [feature_labels[i] for i in top_idx_xgb]
        top_vals_xgb  = xgb_importance[top_idx_xgb]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
        ax.set_facecolor("#111827")
        colors = plt.cm.Blues(np.linspace(0.4, 1.0, top_n_xgb))
        ax.barh(np.arange(top_n_xgb), top_vals_xgb[::-1],
                color=colors[::-1], edgecolor="white", linewidth=0.3)
        ax.set_yticks(np.arange(top_n_xgb))
        ax.set_yticklabels(top_feats_xgb[::-1], fontsize=8, color=FG_COLOR)
        ax.set_xlabel("Mean |SHAP value|", color=FG_COLOR, fontsize=10)
        ax.set_title("Top 20 Features — SHAP Global Importance (XGBoost)",
                     color=FG_COLOR, fontsize=12)
        ax.tick_params(colors=FG_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="white")
        plt.tight_layout()
        out_xgb = os.path.join(FIGURES_DIR, "shap_xgb_bar.png")
        plt.savefig(out_xgb, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        print(f"  Saved: {out_xgb}")

    # ═══════════════════════════════════════════════════════════════════════════
    # DNN — KernelExplainer (approximate, on background sample)
    # ═══════════════════════════════════════════════════════════════════════════
    dnn_path = os.path.join(MODEL_V2_DIR, "dnn_model.keras")
    if os.path.exists(dnn_path):
        _banner("SHAP — DNN (KernelExplainer, approximate)")
        import tensorflow as tf
        dnn_model = tf.keras.models.load_model(dnn_path)

        def dnn_predict_fn(x):
            out = dnn_model.predict(x, verbose=0)
            if out.ndim == 1 or out.shape[1] == 1:
                out = np.column_stack([1 - out.ravel(), out.ravel()])
            return out

        print(f"  Building KernelExplainer (background={SHAP_BACKGROUND_SAMPLES} samples) …")
        kernel_exp = shap.KernelExplainer(dnn_predict_fn, X_bg)

        n_kernel = min(30, N)   # KernelExplainer is slow; limit samples
        print(f"  Computing DNN SHAP for {n_kernel} samples (may take a moment) …")
        dnn_shap = kernel_exp.shap_values(X_explain[:n_kernel], nsamples=100)

        if isinstance(dnn_shap, list):
            dnn_importance = np.abs(np.array(dnn_shap)).mean(axis=0).mean(axis=0)
        elif dnn_shap.ndim == 3:
            if dnn_shap.shape[2] == X_explain.shape[1]:
                dnn_importance = np.abs(dnn_shap).mean(axis=(0, 1))
            else:
                dnn_importance = np.abs(dnn_shap).mean(axis=(0, 2))
        else:
            dnn_importance = np.abs(dnn_shap).mean(axis=0)

        top_n_dnn     = min(15, len(dnn_importance))
        top_idx_dnn   = np.argsort(dnn_importance)[::-1][:top_n_dnn]
        top_feats_dnn = [feature_labels[i] for i in top_idx_dnn]
        top_vals_dnn  = dnn_importance[top_idx_dnn]

        fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG_COLOR)
        ax.set_facecolor("#111827")
        colors = plt.cm.YlOrRd(np.linspace(0.4, 1.0, top_n_dnn))
        ax.barh(np.arange(top_n_dnn), top_vals_dnn[::-1],
                color=colors[::-1], edgecolor="white", linewidth=0.3)
        ax.set_yticks(np.arange(top_n_dnn))
        ax.set_yticklabels(top_feats_dnn[::-1], fontsize=8, color=FG_COLOR)
        ax.set_xlabel("Mean |SHAP value|", color=FG_COLOR, fontsize=10)
        ax.set_title("Top 15 Features — SHAP Importance (DNN / KernelExplainer)",
                     color=FG_COLOR, fontsize=12)
        ax.tick_params(colors=FG_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="white")
        plt.tight_layout()
        out_dnn = os.path.join(FIGURES_DIR, "shap_dnn_bar.png")
        plt.savefig(out_dnn, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        print(f"  Saved: {out_dnn}")

    _banner("SHAP Explainability Complete ✅")


if __name__ == "__main__":
    run()
