"""
xgboost_classifier service — Port 8004
========================================
Wraps the trained XGBoost classifier. Uses 17 raw transaction features only
(no anomaly scores appended) — matching exactly how the model was trained.
Accepts a TransactionRequest and returns the fraud probability.
"""

import os
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from shared.schemas import TransactionRequest, XGBScore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
XGB_MODEL_PATH = os.path.join(PROJECT_ROOT, "Anomally Detection", "XGBoost Classifier", "xgboost_model.joblib")

# ---------------------------------------------------------------------------
# Feature definitions — 17 raw transaction features (matches train_xgboost.py)
# ---------------------------------------------------------------------------
XGBOOST_BASE_FEATURES = [
    'tx_count_last_5m', 'tx_count_last_1h', 'tx_frequency_ratio',
    'amount', 'avg_tx_amount_7d', 'amount_to_avg_ratio',
    'new_device_flag', 'device_seen_before', 'country_mismatch_flag',
    'accounts_per_device', 'accounts_per_ip_24h',
    'failed_login_count_last_1h', 'failed_logins_then_success',
    'channel', 'auth_method', 'geo_country', 'currency',
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
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(f"[xgboost_classifier] Model not found: {XGB_MODEL_PATH}")
    return joblib.load(XGB_MODEL_PATH)

_model = _load_model()
print(f"[xgboost_classifier] Model loaded from {XGB_MODEL_PATH}")

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
    """Build the 17-feature vector — raw transaction features only."""
    row = {}
    for feat in XGBOOST_BASE_FEATURES:
        if feat in CATEGORICAL_COLS:
            row[feat] = _encode_category(tx.get(feat), feat)
        else:
            row[feat] = float(tx.get(feat, 0) or 0)
    return pd.DataFrame([row])

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="XGBoost Classifier Service",
    description="ML Layer: returns the XGBoost fraud probability for a transaction (17 features).",
    version="1.0.0",
)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "xgboost_classifier", "port": 8004}


@app.post("/score", response_model=XGBScore, tags=["Scoring"])
def score(transaction: TransactionRequest):
    """
    Score a transaction with the XGBoost classifier.

    Returns:
    - fraud_score: probability of fraud (0.0–1.0)
    - is_fraud: True if fraud_score >= 0.5
    """
    try:
        X = _build_feature_vector(transaction.model_dump())
        fraud_score = float(_model.predict_proba(X)[0, 1])
        is_fraud    = fraud_score >= 0.5
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XGBoost scoring error: {str(e)}")

    return XGBScore(fraud_score=round(fraud_score, 6), is_fraud=is_fraud)
