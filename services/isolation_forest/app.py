"""
isolation_forest service — Port 8002
======================================
Wraps the trained Isolation Forest model. Accepts a TransactionRequest,
builds the 13-feature unsupervised vector, and returns the IF decision
function score.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from shared.schemas import TransactionRequest, IFScore

# ---------------------------------------------------------------------------
# Paths — model file accessed via Docker volume mount or local path
# ---------------------------------------------------------------------------
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IF_MODEL_PATH = os.path.join(PROJECT_ROOT, "Anomally Detection", "IsolationForest", "isolation_forest_model.joblib")

# ---------------------------------------------------------------------------
# Feature definitions (must match training exactly)
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
def _load_model():
    if not os.path.exists(IF_MODEL_PATH):
        raise FileNotFoundError(f"[isolation_forest] Model not found: {IF_MODEL_PATH}")
    return joblib.load(IF_MODEL_PATH)

_model = _load_model()
print(f"[isolation_forest] Model loaded from {IF_MODEL_PATH}")

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
    title="Isolation Forest Service",
    description="ML Layer: returns the Isolation Forest anomaly score for a transaction.",
    version="1.0.0",
)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "isolation_forest", "port": 8002}


@app.post("/score", response_model=IFScore, tags=["Scoring"])
def score(transaction: TransactionRequest):
    """
    Score a transaction with the Isolation Forest model.

    Returns:
    - if_score: decision_function value (lower / more negative = more anomalous)
    """
    try:
        X = _build_feature_vector(transaction.model_dump())
        if_score = float(_model.decision_function(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Isolation Forest scoring error: {str(e)}")

    return IFScore(if_score=round(if_score, 6))
