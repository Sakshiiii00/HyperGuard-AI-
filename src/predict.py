"""
=============================================================================
HybridPredictor — Core Prediction Engine
=============================================================================
Loads all trained v2 models (RF, XGBoost, DNN, Autoencoder) and
performs a weighted-ensemble hybrid prediction.

Decision Pipeline:
  ┌─────────────────────────────────────────────────────────┐
  │ Raw CSV sample                                          │
  │   → InferencePreprocessor (scale + PCA)                 │
  │      ↓                                                  │
  │   RF prob  ─┐                                           │
  │   XGB prob ─┤ weighted soft vote → classification        │
  │   DNN prob ─┘                                           │
  │                                                          │
  │   Autoencoder MSE → anomaly score → zero-day flag        │
  │      ↓                                                  │
  │   Hybrid threat level: BENIGN / LOW / MEDIUM / HIGH / CRITICAL │
  └─────────────────────────────────────────────────────────┘

Falls back gracefully: if v2 models are absent, uses legacy RF pkl.
=============================================================================
"""

import os
import sys
import json
import warnings
import joblib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Resolve paths ─────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_V2_DIR   = os.path.join(_BASE_DIR, "models", "v2")
_OLD_PKL  = os.path.join(_BASE_DIR, "models", "malware_model.pkl")

sys.path.insert(0, _SRC_DIR)
from data_preprocessing import InferencePreprocessor


# ── Threat-level mapping ──────────────────────────────────────────────────────
def _threat_level(class_name: str, confidence: float, is_anomaly: bool) -> str:
    if class_name.lower() in ("benign",):
        return "BENIGN" if not is_anomaly else "LOW"
    if is_anomaly and confidence > 0.85:
        return "CRITICAL"
    if confidence > 0.90:
        return "HIGH"
    if confidence > 0.70:
        return "MEDIUM"
    return "LOW"


class HybridPredictor:
    """
    Loads and orchestrates all v2 models for hybrid malware prediction.
    Models: Random Forest, XGBoost, DNN, Autoencoder (anomaly detector).
    Thread-safe for multi-worker Flask/Gunicorn (read-only after init).
    """

    def __init__(self, model_dir: str = _V2_DIR):
        self.model_dir    = model_dir
        self.preprocessor = InferencePreprocessor(model_dir)

        # Model slots
        self.rf   = None
        self.xgb  = None
        self.dnn  = None
        self.lstm = None
        self.ae   = None

        # Configs
        self.lstm_cfg       : dict      = {}  
        self.class_names    : list[str] = []
        self.n_classes      : int       = 0
        self.benign_idx     : int       = 0
        self.ensemble_weights: dict     = {}
        self.ae_threshold   : float     = 0.05
        self.feature_imp    : dict      = {}

        # Mode flags
        self.v2_available   : bool = False
        self.legacy_rf      = None   # fallback

        self._load_all()

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        if not os.path.exists(os.path.join(self.model_dir, "metadata.json")):
            print("[HybridPredictor] v2 models not found → using legacy RF fallback.")
            self._load_legacy()
            return

        # Load metadata
        with open(os.path.join(self.model_dir, "metadata.json")) as fh:
            meta = json.load(fh)
        self.class_names = meta["class_names"]
        self.n_classes   = meta["n_classes"]
        self.benign_idx  = meta["benign_class_idx"]

        # Preprocessor
        try:
            self.preprocessor.load()
        except Exception as e:
            print(f"[HybridPredictor] Preprocessor load failed: {e}")
            self._load_legacy()
            return

        # Ensemble weights
        ew_path = os.path.join(self.model_dir, "ensemble_weights.json")
        if os.path.exists(ew_path):
            with open(ew_path) as fh:
                self.ensemble_weights = json.load(fh)
        else:
            self.ensemble_weights = {
                "rf_weight": 0.30, "xgb_weight": 0.40, "dnn_weight": 0.30
            }

        # ── Try to import deep-learning libs (optional; app works without them) ──
        try:
            import tensorflow as tf
            _tf_available = True
        except ImportError:
            tf = None
            _tf_available = False
            print("[HybridPredictor] WARNING: TensorFlow not found. DNN/LSTM/AE models will be skipped.")
            print("[HybridPredictor]          Run:  .venv\\Scripts\\pip install tensorflow")

        try:
            import xgboost as xgb_lib
            _xgb_available = True
        except ImportError:
            xgb_lib = None
            _xgb_available = False
            print("[HybridPredictor] WARNING: XGBoost not found. XGB model will be skipped.")

        # ── Load individual models (each optional) ────────────────────────────
        rf_path = os.path.join(self.model_dir, "rf_model.pkl")
        if os.path.exists(rf_path):
            self.rf = joblib.load(rf_path)
            print("[HybridPredictor] OK Random Forest loaded")

        xgb_path = os.path.join(self.model_dir, "xgb_model.json")
        if os.path.exists(xgb_path) and _xgb_available:
            self.xgb = xgb_lib.XGBClassifier()
            self.xgb.load_model(xgb_path)
            print("[HybridPredictor] OK XGBoost loaded")

        if _tf_available:
            dnn_path = os.path.join(self.model_dir, "dnn_model.keras")
            if os.path.exists(dnn_path):
                self.dnn = tf.keras.models.load_model(dnn_path)
                print("[HybridPredictor] OK DNN loaded")

            lstm_path = os.path.join(self.model_dir, "lstm_model.keras")
            if os.path.exists(lstm_path):
                self.lstm = tf.keras.models.load_model(lstm_path)
                lstm_cfg_path = os.path.join(self.model_dir, "lstm_config.json")
                if os.path.exists(lstm_cfg_path):
                    with open(lstm_cfg_path) as fh:
                        self.lstm_cfg = json.load(fh)
                print("[HybridPredictor] OK LSTM loaded")

            ae_path = os.path.join(self.model_dir, "autoencoder.keras")
            if os.path.exists(ae_path):
                self.ae = tf.keras.models.load_model(ae_path)
                ae_thr_path = os.path.join(self.model_dir, "ae_threshold.json")
                if os.path.exists(ae_thr_path):
                    with open(ae_thr_path) as fh:
                        ae_cfg = json.load(fh)
                    self.ae_threshold = ae_cfg["anomaly_threshold"]
                print(f"[HybridPredictor] OK Autoencoder loaded (threshold={self.ae_threshold:.6f})")

        # SHAP feature importance (for dashboard)
        fi_path = os.path.join(self.model_dir, "feature_importance.json")
        if os.path.exists(fi_path):
            with open(fi_path) as fh:
                self.feature_imp = json.load(fh)

        if any([self.rf, self.xgb, self.dnn, self.lstm]):
            self.v2_available = True
            print(f"[HybridPredictor] Hybrid system ready ({self.n_classes} classes).")
        else:
            print("[HybridPredictor] No v2 models found — falling back to legacy RF.")
            self._load_legacy()

    def _load_legacy(self) -> None:
        if os.path.exists(_OLD_PKL):
            self.legacy_rf = joblib.load(_OLD_PKL)
            if hasattr(self.legacy_rf, "classes_"):
                classes = list(self.legacy_rf.classes_)
                self.class_names = [str(c) for c in classes]
                self.n_classes   = len(classes)
            print("[HybridPredictor] ✔ Legacy RF loaded (fallback mode)")
        else:
            print("[HybridPredictor] ✘ No models available. Train first.")

    # ── Core Prediction ───────────────────────────────────────────────────────

    def predict(self, raw_df: pd.DataFrame) -> list[dict]:
        """
        Main prediction method.

        Parameters
        ----------
        raw_df : pd.DataFrame
            One or more rows of raw memory features (must contain required columns).

        Returns
        -------
        list[dict]  — one prediction dict per row:
            {
              "classification"       : str,   # malware class name
              "confidence"           : float, # ensemble confidence [0-1]
              "threat_level"         : str,   # BENIGN / LOW / MEDIUM / HIGH / CRITICAL
              "is_anomaly"           : bool,  # autoencoder anomaly flag
              "anomaly_score"        : float, # reconstruction MSE
              "class_probs"          : dict,  # {class_name: probability}
              "model_contributions"  : dict,  # per-model top confidence
              "top_features"         : list,  # SHAP feature names (from training)
              "model_used"           : str,   # which prediction path was taken
            }
        """
        if self.v2_available:
            return self._predict_hybrid(raw_df)
        elif self.legacy_rf is not None:
            return self._predict_legacy(raw_df)
        else:
            raise RuntimeError("No models loaded. Run the training pipeline first.")

    def _predict_hybrid(self, raw_df: pd.DataFrame) -> list[dict]:
        # ── Preprocess ────────────────────────────────────────────────────────
        X = self.preprocessor.validate_and_transform(raw_df)
        n = X.shape[0]

        # ── Ensemble soft-voting ───────────────────────────────────────────────
        ens_prob = None
        wsum     = 0.0
        model_contributions = {}  # per-model top confidence tracking

        def _model_prob(prob_raw):
            if prob_raw.ndim == 1 or prob_raw.shape[1] == 1:
                p = np.column_stack([1 - prob_raw.ravel(), prob_raw.ravel()])
            else:
                p = prob_raw
            return p

        if self.rf is not None:
            w = self.ensemble_weights.get("rf_weight", 0.30)
            p = self.rf.predict_proba(X)
            ens_prob = p * w if ens_prob is None else ens_prob + p * w
            wsum += w
            model_contributions["Random Forest"] = {
                "weight": round(w, 3),
                "top_confidence": float(round(float(np.max(p[0])), 4)),
                "predicted_class": self.class_names[int(np.argmax(p[0]))] if self.class_names else str(int(np.argmax(p[0]))),
            }

        if self.xgb is not None:
            w = self.ensemble_weights.get("xgb_weight", 0.40)
            p = self.xgb.predict_proba(X)
            p = _model_prob(p)
            ens_prob = p * w if ens_prob is None else ens_prob + p * w
            wsum += w
            model_contributions["XGBoost"] = {
                "weight": round(w, 3),
                "top_confidence": float(round(float(np.max(p[0])), 4)),
                "predicted_class": self.class_names[int(np.argmax(p[0]))] if self.class_names else str(int(np.argmax(p[0]))),
            }

        if self.dnn is not None:
            w   = self.ensemble_weights.get("dnn_weight", 0.30)
            raw = self.dnn.predict(X, verbose=0)
            p   = _model_prob(raw)
            ens_prob = p * w if ens_prob is None else ens_prob + p * w
            wsum += w
            model_contributions["DNN"] = {
                "weight": round(w, 3),
                "top_confidence": float(round(float(np.max(p[0])), 4)),
                "predicted_class": self.class_names[int(np.argmax(p[0]))] if self.class_names else str(int(np.argmax(p[0]))),
            }

        if ens_prob is None:
            raise RuntimeError("No models produced probabilities.")

        ens_prob /= wsum  # normalize

        # ── Anomaly detection (Autoencoder) ───────────────────────────────────
        if self.ae is not None:
            X_recon = self.ae.predict(X, verbose=0)
            ae_mse  = np.mean((X - X_recon) ** 2, axis=1)
        else:
            ae_mse  = np.zeros(n)

        # ── Per-sample results ─────────────────────────────────────────────────
        results = []
        for i in range(n):
            prob_vec   = ens_prob[i]
            pred_idx   = int(np.argmax(prob_vec))
            confidence = float(prob_vec[pred_idx])
            class_name = (self.class_names[pred_idx]
                          if pred_idx < len(self.class_names)
                          else f"Class_{pred_idx}")
            anomaly_score = float(ae_mse[i]) if self.ae else None
            is_anomaly    = (anomaly_score > self.ae_threshold) if anomaly_score is not None else False

            class_probs = {
                (self.class_names[j] if j < len(self.class_names) else f"Class_{j}"): float(round(float(prob_vec[j]), 4))
                for j in range(len(prob_vec))
            }

            results.append({
                "classification"     : class_name,
                "confidence"         : float(round(confidence, 4)),
                "threat_level"       : _threat_level(class_name, confidence, is_anomaly),
                "is_anomaly"         : bool(is_anomaly),
                "anomaly_score"      : float(round(anomaly_score, 6)) if anomaly_score is not None else None,
                "ae_threshold"       : float(round(self.ae_threshold, 6)),
                "class_probs"        : class_probs,
                "model_contributions": model_contributions,
                "top_features"       : self.feature_imp.get("top_features", [])[:10],
                "model_used"         : "Hybrid (RF+XGB+DNN+AE)",
            })

        return results

    def _predict_legacy(self, raw_df: pd.DataFrame) -> list[dict]:
        """Fallback to original RF model when v2 models are absent."""
        df = raw_df.copy()

        # Strip Filename + Label columns if present
        if df.columns[0].lower() in ("filename", "file_name"):
            df = df.drop(columns=[df.columns[0]])
        last_col = df.columns[-1]
        if last_col.lower() in ("class", "label", "category", "malware"):
            df = df.drop(columns=[last_col])

        if hasattr(self.legacy_rf, "feature_names_in_"):
            feats   = list(self.legacy_rf.feature_names_in_)
            missing = set(feats) - set(df.columns)
            if missing:
                raise ValueError(f"Missing features: {list(missing)[:5]}")
            df = df[feats]

        preds  = self.legacy_rf.predict(df)
        probas = (self.legacy_rf.predict_proba(df)
                  if hasattr(self.legacy_rf, "predict_proba") else None)

        # Legacy label mapping
        legacy_map = {0: "Benign", 1: "Ransomware", 2: "Spyware",
                      3: "Trojan", 4: "Worm", 8: "Spyware-T"}

        results = []
        for i, pred in enumerate(preds):
            try:
                class_name = str(legacy_map.get(int(pred), str(pred)))
            except Exception:
                class_name = str(pred)

            confidence = float(max(probas[i])) if probas is not None else 1.0
            results.append({
                "classification"     : class_name,
                "confidence"         : round(confidence, 4),
                "threat_level"       : _threat_level(class_name, confidence, False),
                "is_anomaly"         : False,
                "anomaly_score"      : None,
                "ae_threshold"       : None,
                "class_probs"        : {},
                "model_contributions": {},
                "top_features"       : [],
                "model_used"         : "Legacy Random Forest",
            })

        return results

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "v2_available"    : self.v2_available,
            "models_loaded"   : {
                "random_forest" : self.rf  is not None,
                "xgboost"       : self.xgb is not None,
                "dnn"           : self.dnn is not None,
                "autoencoder"   : self.ae  is not None,
            },
            "n_classes"       : self.n_classes,
            "class_names"     : self.class_names,
            "ae_threshold"    : self.ae_threshold,
            "ensemble_weights": self.ensemble_weights,
        }


# ---------------------------------------------------------------------------
# Legacy helper functions (preserve backward compatibility with old app.py)
# ---------------------------------------------------------------------------
def load_malware_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_sample(model, sample_features):
    if isinstance(sample_features, dict):
        df = pd.DataFrame([sample_features])
    elif isinstance(sample_features, (list, tuple)):
        df = pd.DataFrame([sample_features])
    else:
        df = sample_features
    return model.predict(df)[0]
