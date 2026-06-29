# HyperGuard AI — Hybrid Intelligent Malware Detection Framework

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![Dataset](https://img.shields.io/badge/Dataset-Obfuscated%20MalMem2022-purple.svg)](https://www.unb.ca/cic/datasets/)

> Research-grade hybrid malware detection system combining Random Forest, XGBoost, and a Deep Neural Network in a weighted ensemble, with an Autoencoder for zero-day anomaly detection — featuring SHAP explainability and a premium REST API dashboard.

---

## Architecture

```
Obfuscated-MalMem2022 Dataset (58,596 rows · 57 cols)
        │
        ▼
┌─────────────────────────────────────────────┐
│  pipeline/01_preprocess.py                  │
│  ● SMOTE class balancing                    │
│  ● StandardScaler                           │
│  ● PCA (95% variance retained)              │
│  ● LabelEncoder → class_mapping.json        │
└──────────────────┬──────────────────────────┘
                   │  models/v2/ artifacts
         ┌─────────┴─────────┐
         ▼                   ▼
  02_train_ensemble    03_train_autoencoder
  RF + XGB + DNN       Benign-only AE
         └─────────┬─────────┘
                   ▼
          04_evaluate → plots, tables
          05_explain  → SHAP figures
                   │
                   ▼
          app/app.py  (Flask REST API)
          app/templates/index.html  (Dashboard)
```

---

## Model Architecture Details

### 1. Random Forest (Baseline + Ensemble member)
- 200–300 estimators, `class_weight='balanced'`
- GridSearchCV: `n_estimators`, `max_features`, `min_samples_split`
- 5-fold stratified CV

### 2. XGBoost
- Multi-class `softprob` objective
- GridSearchCV: `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- 5-fold stratified CV

### 3. Deep Neural Network (Keras)
```
Input(n_pca) → Dense(512) → BN → Dropout(0.3)
             → Dense(256) → BN → Dropout(0.3)
             → Dense(128) → BN → Dropout(0.25)
             → Dense(64)  → BN → Dropout(0.2)
             → Softmax(n_classes)
```
- EarlyStopping + ReduceLROnPlateau
- Adam optimizer (lr=1e-3)

### 4. Autoencoder (Zero-Day Detection)
```
Encoder: N → 128 → 64 → 32 → 16  (bottleneck)
Decoder: 16 → 32 → 64 → 128 → N
```
- Trained **only on BENIGN** samples
- Reconstruction error (MSE) = anomaly score
- Threshold = mean + 2σ of benign validation MSE
- Exceeding threshold → flagged as zero-day / anomalous

### 5. Hybrid Ensemble
Weighted soft-voting across RF, XGB, and DNN:
```
P_final = w_rf × P_rf + w_xgb × P_xgb + w_dnn × P_dnn
```
Weights are set proportional to individual F1-macro scores after training.

### Threat Level Assignment
| Classification | Confidence | Anomaly | → Threat Level |
|---|---|---|---|
| Benign | any | No | BENIGN |
| Benign | any | Yes | LOW |
| Malware | < 0.70 | No | LOW |
| Malware | 0.70–0.90 | any | MEDIUM |
| Malware | > 0.90 | No | HIGH |
| Malware | > 0.85 | Yes | CRITICAL |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA GPU (optional but recommended for DNN training)

### 1. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify dataset
Ensure this file exists:
```
dataset/Obfuscated-MalMem2022.csv
```
> **Dataset:** Obfuscated-MalMem2022 — 58,596 memory forensic samples (29,298 Benign + 29,298 Malware)
> comprising Ransomware, Spyware, and Trojan families, extracted using Volatility3.
> Source: [CIC Datasets — University of New Brunswick](https://www.unb.ca/cic/datasets/malmem-2022.html)

---

## Running the Training Pipeline

### Full pipeline (recommended)
```bash
python pipeline_runner.py
```

Runs 5 steps in order:
| Step | Script | Description | Est. Time (CPU) |
|------|--------|-------------|-----------------|
| 1 | `01_preprocess.py` | SMOTE, PCA, Scaler | ~2–5 min |
| 2 | `02_train_ensemble.py` | RF + XGB + DNN | ~15–30 min |
| 3 | `04_train_autoencoder.py` | Zero-day AE | ~5–10 min |
| 4 | `05_evaluate.py` | Plots + metrics | ~2–5 min |
| 5 | `06_explain.py` | SHAP figures | ~5–15 min |

### Individual steps
```bash
python pipeline/01_preprocess.py          # Step 1
python pipeline/02_train_ensemble.py      # Step 2
python pipeline/04_train_autoencoder.py   # Step 3
python pipeline/05_evaluate.py            # Step 4
python pipeline/06_explain.py             # Step 5
```

### Resume / partial runs
```bash
python pipeline_runner.py --from 3     # Resume from Autoencoder
python pipeline_runner.py --only 4     # Only run evaluation
python pipeline_runner.py --skip-shap  # Skip SHAP (slow on large datasets)
```

---

## Starting the Web Application

```bash
python app/app.py
```

Then open: **http://127.0.0.1:5000**

> **Note:** If v2 models haven't been trained yet, run `python pipeline_runner.py` first to generate all model artifacts before starting the app.

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Premium dashboard UI |
| POST | `/api/predict` | Real-time single CSV prediction |
| POST | `/api/predict/batch` | Batch CSV → downloadable results CSV |
| GET  | `/api/models/status` | Model status + comparison metrics |
| GET  | `/api/reports` | List evaluation plots + tables |
| GET  | `/api/health` | Health check |

### Example: Single Prediction
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -F "csv_file=@benign_sample.csv"
```

Response:
```json
{
  "success": true,
  "prediction": {
    "classification": "Benign",
    "confidence": 0.9712,
    "threat_level": "BENIGN",
    "is_anomaly": false,
    "anomaly_score": 0.000812,
    "ae_threshold": 0.002314,
    "class_probs": { "Benign": 0.9712, "Ransomware": 0.018, "..." : "..." },
    "model_contributions": {
      "Random Forest": { "weight": 0.3, "top_confidence": 0.97, "predicted_class": "Benign" },
      "XGBoost":       { "weight": 0.4, "top_confidence": 0.96, "predicted_class": "Benign" },
      "DNN":           { "weight": 0.3, "top_confidence": 0.98, "predicted_class": "Benign" }
    },
    "top_features": ["PCA_001", "PCA_005", "..."],
    "model_used": "Hybrid (RF+XGB+DNN+AE)",
    "inference_ms": 42.3
  }
}
```

---

## Output Artifacts

| Path | Description |
|------|-------------|
| `models/v2/rf_model.pkl` | Trained Random Forest |
| `models/v2/xgb_model.json` | Trained XGBoost |
| `models/v2/dnn_model.keras` | Trained DNN |
| `models/v2/autoencoder.keras` | Trained Autoencoder (zero-day) |
| `models/v2/scaler.pkl` | StandardScaler |
| `models/v2/pca.pkl` | PCA transformer |
| `models/v2/label_encoder.pkl` | Class label encoder |
| `models/v2/ensemble_weights.json` | Per-model ensemble weights |
| `models/v2/comparison_results.json` | Per-model metrics |
| `models/v2/feature_importance.json` | SHAP feature ranking |
| `reports/figures/confusion_matrices.png` | All confusion matrices |
| `reports/figures/roc_curves.png` | Overlaid ROC curves |
| `reports/figures/model_comparison.png` | Bar chart comparison |
| `reports/figures/dnn_training_loss.png` | DNN loss + accuracy curves |
| `reports/figures/rf_feature_importance.png` | RF feature importance (Gini) |
| `reports/figures/xgb_feature_importance.png` | XGB feature importance |
| `reports/figures/shap_rf_bar.png` | RF SHAP global importance |
| `reports/figures/shap_xgb_bar.png` | XGB SHAP global importance |
| `reports/figures/autoencoder_anomaly_scores.png` | AE score distribution |
| `reports/model_comparison.csv` | Comparison table (CSV) |
| `reports/ablation_study.csv` | Ablation study (CSV) |

---

## Research Notes (For Paper Writing)

### Training Results (Obfuscated-MalMem2022 — 4-class)

> **Problem:** 4-class multi-label classification (Benign / Ransomware / Spyware / Trojan)
> **Features:** 51 memory forensic features (55 raw − 4 leaky removed)
> **PCA components:** 14 (95.7% variance retained)

| Model | Accuracy | F1-Macro | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 80.9% | 71.4% | 71.6% | 71.4% |
| XGBoost | 81.6% | 72.5% | 72.5% | 72.5% |
| DNN | 76.0% | 62.3% | 66.9% | 63.8% |
| **Hybrid Ensemble** | **81.2%** | **71.8%** | **71.9%** | **71.8%** |

**Autoencoder (Anomaly Detection):**
- Anomaly threshold: 0.173 (mean + 2σ)
- Malware TPR: 97.41% · Benign FPR: 0.20%

### Key Contributions
1. **Hybrid Detection Framework** combining signature-based (ML ensemble: RF + XGB + DNN) and anomaly-based (Autoencoder) detection
2. **Per-model contributions** exposed at inference time for transparency and explainability
3. **SHAP explainability** for compliance with regulatory requirements (EU AI Act transparency provisions)
4. **Zero-day detection** via autoencoder reconstruction error without requiring malware labels at inference time
5. **Obfuscated dataset**: trained on `Obfuscated-MalMem2022`, a harder benchmark where malware authors apply obfuscation techniques to evade detection

### Suggested Paper Sections
- Section III — Dataset (Obfuscated-MalMem2022 description, class distribution, Table I)
- Section IV — Proposed Framework (Architecture diagram, model descriptions)
- Section V — Experiments (Table II: model comparison, Figure 2: ROC, Figure 3: Confusion matrices)
- Section VI — Ablation Study (Table III: ablation results from `reports/ablation_study.csv`)
- Section VII — Explainability (Figure 4: SHAP summary plots + DNN loss curves)
- Section VIII — Discussion (zero-day detection analysis using Figure 5: AE anomaly scores)

### Baseline Comparison Table (template)
| Model | Accuracy | F1-Macro | ROC-AUC | Training Time |
|---|---|---|---|---|
| Random Forest (baseline) | see reports | see reports | see reports | — |
| XGBoost | — | — | — | — |
| DNN | — | — | — | — |
| **Hybrid Ensemble (RF+XGB+DNN)** | — | — | — | — |

---

## Project Structure

```
malware-detection-ml/
├── app/
│   ├── app.py                   ← Flask REST API (6 endpoints)
│   └── templates/
│       └── index.html           ← Premium cybersecurity dashboard
├── pipeline/
│   ├── 01_preprocess.py         ← SMOTE, PCA, Scaler
│   ├── 02_train_ensemble.py     ← RF + XGB + DNN (+ visualizations)
│   ├── 04_train_autoencoder.py  ← Zero-day Autoencoder
│   ├── 05_evaluate.py           ← Evaluation plots + metrics
│   └── 06_explain.py            ← SHAP explainability
├── src/
│   ├── data_preprocessing.py    ← InferencePreprocessor (scaler + PCA)
│   └── predict.py               ← HybridPredictor (RF+XGB+DNN+AE)
├── dataset/
│   └── Obfuscated-MalMem2022.csv ← Dataset (58,596 samples)
├── models/
│   └── v2/                       ← All v2 model artifacts (after training)
├── reports/
│   └── figures/                 ← Generated plots (after pipeline run)
├── pipeline_runner.py           ← Master training orchestrator
├── requirements.txt
└── README.md
```
