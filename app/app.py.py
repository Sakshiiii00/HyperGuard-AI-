"""
=============================================================================
Flask REST API — Hybrid Malware Detection Framework
=============================================================================
Endpoints:
  GET  /                   → Premium dashboard UI
  POST /api/predict        → Real-time single prediction (CSV upload)
  POST /api/predict/batch  → Batch CSV prediction (returns downloadable CSV)
  GET  /api/models/status  → Model loading status + version info
  GET  /api/health         → Health check
  GET  /api/reports        → Fetch evaluation metrics + plots manifest

CORS enabled for all /api/ routes.
=============================================================================
"""

import os
import sys
import json
import time
import io
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    send_file, make_response
)
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from predict import HybridPredictor

# ── Flask Setup ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "templates"),
    static_folder=os.path.join(APP_DIR, "static"),
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Paths ─────────────────────────────────────────────────────────────────────
V2_DIR      = os.path.join(BASE_DIR, "models", "v2")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# ── Load Predictor at startup ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Hybrid Malware Detection Framework — API Server")
print("=" * 60)
predictor = HybridPredictor(model_dir=V2_DIR)

_startup_time = datetime.utcnow().isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _error(msg: str, code: int = 400) -> tuple:
    return jsonify({"success": False, "error": msg}), code


def _safe_read_csv(file_obj) -> pd.DataFrame:
    """Read CSV from file object; raise ValueError on parse failure."""
    try:
        df = pd.read_csv(file_obj)
        if df.empty:
            raise ValueError("Uploaded CSV is empty.")
        return df
    except Exception as e:
        raise ValueError(f"CSV parse error: {e}")


def _strip_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Filename and Class columns if present (for inference)."""
    drops = []
    if len(df.columns) > 0:
        fc = df.columns[0]
        if fc.lower() in ("filename", "file_name", "file"):
            drops.append(fc)
    if len(df.columns) > 1:
        lc = df.columns[-1]
        if lc.lower() in ("class", "label", "category", "malware", "type"):
            drops.append(lc)
    if drops:
        df = df.drop(columns=drops)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Serve the premium cybersecurity dashboard."""
    status = predictor.status()

    # Load comparison results if available
    comp_results   = {}
    comp_path = os.path.join(V2_DIR, "comparison_results.json")
    if os.path.exists(comp_path):
        with open(comp_path) as fh:
            comp_results = json.load(fh)

    # Ablation study
    ablation = {}
    abl_path = os.path.join(REPORTS_DIR, "ablation_study.csv")
    if os.path.exists(abl_path):
        try:
            ablation = pd.read_csv(abl_path, index_col=0).to_dict()
        except Exception:
            pass

    # Feature importance for dashboard
    fi = predictor.feature_imp or {}

    return render_template(
        "index.html",
        status=status,
        comp_results=comp_results,
        ablation=ablation,
        feature_importance=fi,
        startup_time=_startup_time,
    )


# ─── POST /api/predict ────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Real-time single-sample prediction.
    Accepts:  multipart/form-data with field 'csv_file'
    Returns:  JSON prediction with full details
    """
    if "csv_file" not in request.files:
        return _error("No file part — send CSV as field 'csv_file'.")

    file = request.files["csv_file"]
    if file.filename == "":
        return _error("No file selected.")
    if not file.filename.lower().endswith(".csv"):
        return _error("Only .csv files are accepted.")

    t0 = time.perf_counter()

    try:
        df = _safe_read_csv(file.stream)
        df = _strip_meta_columns(df)

        # Use first row only for real-time endpoint
        sample = df.iloc[0:1].reset_index(drop=True)

        results = predictor.predict(sample)
        result  = results[0]

    except ValueError as e:
        return _error(str(e), 422)
    except Exception as e:
        traceback.print_exc()
        return _error(f"Prediction failed: {str(e)}", 500)

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    result["inference_ms"] = elapsed
    result["timestamp"]    = datetime.utcnow().isoformat()

    return jsonify({"success": True, "prediction": result})


# ─── POST /api/predict/batch ──────────────────────────────────────────────────
@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    """
    Batch prediction for all rows in an uploaded CSV.
    Returns a downloadable CSV with prediction columns appended.
    """
    if "csv_file" not in request.files:
        return _error("No file part — send CSV as field 'csv_file'.")

    file = request.files["csv_file"]
    if not file.filename.lower().endswith(".csv"):
        return _error("Only .csv files are accepted.")

    t0 = time.perf_counter()

    try:
        df_original = _safe_read_csv(file.stream)
        df_features = _strip_meta_columns(df_original.copy())

        results = predictor.predict(df_features)

    except ValueError as e:
        return _error(str(e), 422)
    except Exception as e:
        traceback.print_exc()
        return _error(f"Batch prediction failed: {str(e)}", 500)

    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    # Append prediction columns to original dataframe
    df_out = df_original.copy()
    df_out["pred_classification"] = [r["classification"] for r in results]
    df_out["pred_confidence"]     = [r["confidence"]     for r in results]
    df_out["pred_threat_level"]   = [r["threat_level"]   for r in results]
    df_out["pred_is_anomaly"]     = [r["is_anomaly"]     for r in results]
    if results[0]["anomaly_score"] is not None:
        df_out["pred_anomaly_score"] = [r["anomaly_score"] for r in results]

    # Stream as CSV download
    output_buf = io.StringIO()
    df_out.to_csv(output_buf, index=False)
    output_buf.seek(0)

    resp = make_response(output_buf.getvalue())
    resp.headers["Content-Disposition"] = (
        f"attachment; filename=batch_predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    resp.headers["Content-Type"]  = "text/csv"
    resp.headers["X-Inference-Ms"] = str(elapsed)
    resp.headers["X-Row-Count"]    = str(len(results))

    return resp


# ─── GET /api/models/status ───────────────────────────────────────────────────
@app.route("/api/models/status", methods=["GET"])
def api_models_status():
    """Return model loading status and version metadata."""
    status = predictor.status()

    # Append file sizes
    model_files = {
        "rf_model.pkl"        : "random_forest",
        "xgb_model.json"      : "xgboost",
        "dnn_model.keras"     : "dnn",
        "lstm_model.keras"    : "lstm",
        "autoencoder.keras"   : "autoencoder",
    }
    file_info = {}
    for fname, key in model_files.items():
        fpath = os.path.join(V2_DIR, fname)
        if os.path.exists(fpath):
            size_mb = round(os.path.getsize(fpath) / 1024 / 1024, 2)
            file_info[key] = {"file": fname, "size_mb": size_mb}

    # Comparison results
    comp = {}
    cp = os.path.join(V2_DIR, "comparison_results.json")
    if os.path.exists(cp):
        with open(cp) as fh:
            comp = json.load(fh)

    return jsonify({
        "success"      : True,
        "status"       : status,
        "model_files"  : file_info,
        "comparison"   : comp,
        "startup_time" : _startup_time,
        "server_time"  : datetime.utcnow().isoformat(),
    })


# ─── GET /api/reports ──────────────────────────────────────────────────────────
@app.route("/api/reports", methods=["GET"])
def api_reports():
    """Return available report files and comparison metrics."""
    figures = []
    if os.path.isdir(FIGURES_DIR):
        figures = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]

    comp = {}
    cp = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(cp):
        try:
            comp = pd.read_csv(cp, index_col=0).to_dict()
        except Exception:
            pass

    ablation = {}
    ap = os.path.join(REPORTS_DIR, "ablation_study.csv")
    if os.path.exists(ap):
        try:
            ablation = pd.read_csv(ap, index_col=0).to_dict()
        except Exception:
            pass

    return jsonify({
        "success"       : True,
        "figure_files"  : figures,
        "comparison"    : comp,
        "ablation_study": ablation,
    })


# ─── GET /api/health ──────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def api_health():
    status = predictor.status()
    healthy = status["v2_available"] or predictor.legacy_rf is not None
    return jsonify({
        "status"        : "healthy" if healthy else "degraded",
        "models_loaded" : status["models_loaded"],
        "v2_available"  : status["v2_available"],
        "timestamp"     : datetime.utcnow().isoformat(),
    }), 200 if healthy else 503


# ─── Error handlers ───────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed."}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
