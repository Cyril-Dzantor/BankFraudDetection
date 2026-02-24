"""
autoencoder service — Port 8003
=================================
Wraps the trained Lightweight Autoencoder model + scaler + threshold.
Accepts a TransactionRequest, builds the 13-feature unsupervised vector,
and returns the reconstruction error plus anomaly flag.
"""

import os
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from shared.schemas import TransactionRequest, AEScore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AE_DIR         = os.path.join(PROJECT_ROOT, "Anomally Detection", "Lightweight Autoencoder")
AE_MODEL_PATH  = os.path.join(AE_DIR, "autoencoder_model.joblib")
AE_SCALER_PATH = os.path.join(AE_DIR, "autoencoder_scaler.joblib")
AE_THRESH_PATH = os.path.join(AE_DIR, "ae_threshold.txt")

# ---------------------------------------------------------------------------
# Feature definitions (same 13 features as Isolation Forest)
# ---------------------------------------------------------------------------
UNSUPERVISED_FEATURES = [
    'tx_count_last_5m', 'tx_count_last_1h', 'tx_frequency_ratio',
    'amount', 'avg_tx_amount_7d', 'amount_to_avg_ratio',
    'failed_login_count_last_1h', 'accounts_per_device',
    'accounts_per_ip_24h',
    'geo_country', 'currency', 'channel', 'auth_method',
]

CATEGORICAL_COLS = ['geo_country', 'currency', 'channel', 'auth_method']

KNOWN_CATEGORIES = {
    'geo_country': ['GH', 'NG', 'KE', 'ZA', 'US', 'GB', 'Unknown'],
    'currency':    ['GHS', 'NGN', 'KES', 'ZAR', 'USD', 'GBP', 'Unknown'],
    'channel':     ['mobile', 'web', 'atm', 'pos', 'Unknown'],
    'auth_method': ['pin', 'biometric', 'otp', 'password', 'Unknown'],
}

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
def _load_models():
    for path in [AE_MODEL_PATH, AE_SCALER_PATH, AE_THRESH_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[autoencoder] File not found: {path}")
    model  = joblib.load(AE_MODEL_PATH)
    scaler = joblib.load(AE_SCALER_PATH)
    with open(AE_THRESH_PATH, 'r') as f:
        threshold = float(f.read().strip())
    return model, scaler, threshold

_ae_model, _ae_scaler, _ae_threshold = _load_models()
print(f"[autoencoder] Model loaded. Threshold = {_ae_threshold:.6f}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _encode_category(value, col: str) -> int:
    known = KNOWN_CATEGORIES.get(col, [])
    val = str(value).strip() if value is not None else 'Unknown'
    if val in known:
        return known.index(val)
    return known.index('Unknown') if 'Unknown' in known else 0


def _build_feature_vector(tx: dict) -> pd.DataFrame:
    row = {}
    for feat in UNSUPERVISED_FEATURES:
        if feat in CATEGORICAL_COLS:
            row[feat] = _encode_category(tx.get(feat), feat)
        else:
            row[feat] = float(tx.get(feat, 0) or 0)
    return pd.DataFrame([row])

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Autoencoder Service",
    description="ML Layer: returns the Autoencoder reconstruction error for a transaction.",
    version="1.0.0",
)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "autoencoder", "port": 8003}


@app.post("/score", response_model=AEScore, tags=["Scoring"])
def score(transaction: TransactionRequest):
    """
    Score a transaction with the Lightweight Autoencoder.

    Returns:
    - ae_error: MSE reconstruction error (higher = more anomalous)
    - ae_threshold: the trained anomaly threshold
    - ae_anomaly: True if ae_error > ae_threshold
    """
    try:
        X = _build_feature_vector(transaction.model_dump())
        X_scaled      = _ae_scaler.transform(X)
        reconstruction = _ae_model.predict(X_scaled)
        ae_error       = float(np.mean(np.power(X_scaled - reconstruction, 2)))
        ae_anomaly     = ae_error > _ae_threshold
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autoencoder scoring error: {str(e)}")

    return AEScore(
        ae_error=round(ae_error, 6),
        ae_threshold=round(_ae_threshold, 6),
        ae_anomaly=ae_anomaly,
    )
