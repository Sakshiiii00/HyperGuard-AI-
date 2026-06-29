"""
=============================================================================
PIPELINE STEP 4 — Autoencoder for Anomaly / Zero-Day Detection
=============================================================================
Trains a deep autoencoder on *benign-only* samples.

Logic:
  - Autoencoder learns to reconstruct normal (benign) memory patterns
  - On malware / zero-day input → reconstruction error (MSE) spikes
  - Anomaly threshold = mean + 2 * std of benign validation reconstruction error
  - Samples exceeding threshold flagged as "Anomalous / Zero-Day"

Architecture (symmetric):
  Encoder: N → 128 → 64 → 32 → 16
  Decoder: 16 → 32 → 64 → 128 → N

Output artifacts (models/v2/):
  autoencoder.keras, ae_threshold.json
=============================================================================
"""

import os
import sys
import json
import warnings

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_V2_DIR = os.path.join(BASE_DIR, "models", "v2")

RANDOM_STATE  = 42


def _banner(msg: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {msg}")
    print(f"{'═' * 60}")


def _build_autoencoder(n_features: int) -> tf.keras.Model:
    """
    Symmetric deep autoencoder.
    Bottleneck at 16 dims forces learning of compressed benign representation.
    """
    # ── Encoder ──────────────────────────────────────────────────────────────
    inp = layers.Input(shape=(n_features,), name="ae_input")

    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    encoded = layers.Dense(16, activation="relu", name="bottleneck")(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = layers.Dense(32, activation="relu")(encoded)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    decoded = layers.Dense(n_features, activation="linear", name="ae_output")(x)

    autoencoder = models.Model(inp, decoded, name="Autoencoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return autoencoder


def reconstruction_error(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Per-sample Mean Squared Reconstruction Error."""
    X_recon = model.predict(X, verbose=0)
    mse     = np.mean((X - X_recon) ** 2, axis=1)
    return mse


def run() -> None:
    _banner("STEP 4 — Autoencoder (Zero-Day / Anomaly Detection)")

    # ── Load benign training data ──────────────────────────────────────────────
    benign_path = os.path.join(MODEL_V2_DIR, "X_benign_train.npy")
    if not os.path.exists(benign_path):
        sys.exit("[ERROR] Run pipeline/01_preprocess.py first.")

    X_benign = np.load(benign_path)
    n_features = X_benign.shape[1]
    print(f"\n  Benign train set : {X_benign.shape}")
    print(f"  Input features   : {n_features}")

    # ── Also load full test set for threshold validation ─────────────────────
    data_path = os.path.join(MODEL_V2_DIR, "processed_data.npz")
    data      = np.load(data_path, allow_pickle=True)
    X_test    = data["X_test"]
    y_test    = data["y_test"]

    with open(os.path.join(MODEL_V2_DIR, "metadata.json")) as fh:
        meta = json.load(fh)
    benign_idx = meta["benign_class_idx"]

    # Split benign into train/val (80/20)
    n_val       = int(len(X_benign) * 0.2)
    X_ben_train = X_benign[n_val:]
    X_ben_val   = X_benign[:n_val]

    print(f"  Benign train     : {X_ben_train.shape}")
    print(f"  Benign val       : {X_ben_val.shape}")

    # ── Build autoencoder ──────────────────────────────────────────────────────
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    ae = _build_autoencoder(n_features)
    ae.summary()

    # ── Train ──────────────────────────────────────────────────────────────────
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-7, verbose=1
        ),
    ]

    print("\n  Training Autoencoder on benign data …")
    ae.fit(
        X_ben_train, X_ben_train,
        validation_data=(X_ben_val, X_ben_val),
        epochs=100,
        batch_size=256,
        callbacks=cb_list,
        verbose=1,
    )

    # ── Compute anomaly threshold from benign validation reconstruction error ──
    val_mse = reconstruction_error(ae, X_ben_val)
    threshold_mean   = float(np.mean(val_mse))
    threshold_std    = float(np.std(val_mse))
    anomaly_threshold = threshold_mean + 2.0 * threshold_std

    print(f"\n  Benign val MSE — mean : {threshold_mean:.6f}")
    print(f"  Benign val MSE — std  : {threshold_std:.6f}")
    print(f"  Anomaly threshold     : {anomaly_threshold:.6f}  (mean + 2σ)")

    # ── Validate on test set ───────────────────────────────────────────────────
    test_mse        = reconstruction_error(ae, X_test)
    predicted_benign = test_mse <= anomaly_threshold
    true_benign      = y_test == benign_idx

    anomaly_accuracy = float(np.mean(predicted_benign == true_benign))
    # True Positive Rate (malware correctly flagged as anomaly)
    mal_mask  = ~true_benign
    tpr       = float(np.mean(~predicted_benign[mal_mask])) if mal_mask.sum() else 0.0
    # False Positive Rate (benign incorrectly flagged as anomaly)
    fpr       = float(np.mean(~predicted_benign[true_benign])) if true_benign.sum() else 0.0

    print(f"\n  Autoencoder Anomaly Detection on Test Set:")
    print(f"    Overall accuracy : {anomaly_accuracy:.4f}")
    print(f"    Malware TPR      : {tpr:.4f}  (fraction of malware flagged anomalous)")
    print(f"    Benign FPR       : {fpr:.4f}  (fraction of benign falsely flagged)")

    # ── Save ──────────────────────────────────────────────────────────────────
    ae.save(os.path.join(MODEL_V2_DIR, "autoencoder.keras"))

    ae_config = {
        "anomaly_threshold" : float(round(anomaly_threshold, 8)),
        "threshold_mean"    : float(round(threshold_mean,    8)),
        "threshold_std"     : float(round(threshold_std,     8)),
        "n_input_features"  : int(n_features),
        "test_anomaly_acc"  : float(round(anomaly_accuracy,  4)),
        "malware_tpr"       : float(round(tpr, 4)),
        "benign_fpr"        : float(round(fpr, 4)),
    }
    with open(os.path.join(MODEL_V2_DIR, "ae_threshold.json"), "w") as fh:
        json.dump(ae_config, fh, indent=2)

    _banner("Autoencoder Training Complete ✅")
    print(f"  anomaly_threshold saved: {anomaly_threshold:.6f}")


if __name__ == "__main__":
    run()
