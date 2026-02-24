"""
predict.py — Bank Fraud Detection: ML Inference Pipeline
=========================================================
Loads all trained models once and exposes a single `score()` method
that accepts a raw transaction dict and returns a structured fraud score.

Inference Flow:
  Raw transaction dict
        │
        ├─► Isolation Forest  → isolation_forest_score
        ├─► Autoencoder       → autoencoder_reconstruction_error
        └─► XGBoost Classifier (17 raw transaction features — independent)
                │
                ▼
        {fraud_score, is_fraud, confidence, if_score, ae_error, ae_threshold}
"""

import os
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IF_MODEL_PATH  = os.path.join(BASE_DIR, 'IsolationForest',        'isolation_forest_model.joblib')
AE_MODEL_PATH  = os.path.join(BASE_DIR, 'Lightweight Autoencoder', 'autoencoder_model.joblib')
AE_SCALER_PATH = os.path.join(BASE_DIR, 'Lightweight Autoencoder', 'autoencoder_scaler.joblib')
AE_THRESH_PATH = os.path.join(BASE_DIR, 'Lightweight Autoencoder', 'ae_threshold.txt')
XGB_MODEL_PATH = os.path.join(BASE_DIR, 'XGBoost Classifier',     'xgboost_model.joblib')

# ---------------------------------------------------------------------------
# Feature Definitions  (must match training scripts exactly)
# ---------------------------------------------------------------------------

# Features consumed by Isolation Forest and Autoencoder (13 features)
UNSUPERVISED_FEATURES = [
    'tx_count_last_5m', 'tx_count_last_1h', 'tx_frequency_ratio',
    'amount', 'avg_tx_amount_7d', 'amount_to_avg_ratio',
    'failed_login_count_last_1h', 'accounts_per_device',
    'accounts_per_ip_24h',
    'geo_country', 'currency', 'channel', 'auth_method',
]

# Feature set for XGBoost (17 raw transaction features — trained without anomaly scores)
XGBOOST_BASE_FEATURES = [
    'tx_count_last_5m', 'tx_count_last_1h', 'tx_frequency_ratio',
    'amount', 'avg_tx_amount_7d', 'amount_to_avg_ratio',
    'new_device_flag', 'device_seen_before', 'country_mismatch_flag',
    'accounts_per_device', 'accounts_per_ip_24h',
    'failed_login_count_last_1h', 'failed_logins_then_success',
    'channel', 'auth_method', 'geo_country', 'currency',
]

CATEGORICAL_COLS = ['geo_country', 'currency', 'channel', 'auth_method']

# Known categories from training data (for consistent ordinal encoding at inference)
# Unknown values will be mapped to 0.
KNOWN_CATEGORIES = {
    'geo_country': ['GH', 'NG', 'KE', 'ZA', 'US', 'GB', 'Unknown'],
    'currency':    ['GHS', 'NGN', 'KES', 'ZAR', 'USD', 'GBP', 'Unknown'],
    'channel':     ['mobile', 'web', 'atm', 'pos', 'Unknown'],
    'auth_method': ['pin', 'biometric', 'otp', 'password', 'Unknown'],
}


# ---------------------------------------------------------------------------
# Helper: Encode a single categorical value to integer
# ---------------------------------------------------------------------------
def _encode_category(value, col: str) -> int:
    """
    Maps a category string to its index in KNOWN_CATEGORIES.
    Unknown or missing values map to index of 'Unknown' (last entry = safe default).
    """
    known = KNOWN_CATEGORIES.get(col, [])
    val = str(value).strip() if value is not None else 'Unknown'
    if val in known:
        return known.index(val)
    return known.index('Unknown') if 'Unknown' in known else 0


# ---------------------------------------------------------------------------
# FraudScorer
# ---------------------------------------------------------------------------
class FraudScorer:
    """
    Loads all trained models on construction and exposes a single `score()`
    method for real-time inference.

    Usage:
        scorer = FraudScorer()
        result = scorer.score(transaction)
        print(result['fraud_score'])
    """

    def __init__(self):
        print("[FraudScorer] Loading models...")
        self._check_paths()

        self.if_model   = joblib.load(IF_MODEL_PATH)
        self.ae_model   = joblib.load(AE_MODEL_PATH)
        self.ae_scaler  = joblib.load(AE_SCALER_PATH)
        self.xgb_model  = joblib.load(XGB_MODEL_PATH)

        with open(AE_THRESH_PATH, 'r') as f:
            self.ae_threshold = float(f.read().strip())

        print(f"[FraudScorer] All models loaded. AE threshold = {self.ae_threshold:.6f}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, transaction: dict) -> dict:
        """
        Scores a single transaction.

        Parameters
        ----------
        transaction : dict
            Raw transaction fields. At minimum supply all fields in
            UNSUPERVISED_FEATURES and XGBOOST_BASE_FEATURES.
            Missing fields default to 0 / 'Unknown'.

        Returns
        -------
        dict with keys:
            fraud_score  : float  (0.0 – 1.0)
            is_fraud     : bool   (True if score ≥ 0.5)
            confidence   : str    ("HIGH" | "MEDIUM" | "LOW")
            if_score     : float  (raw Isolation Forest decision_function value)
            ae_error     : float  (Autoencoder reconstruction MSE)
            ae_threshold : float
            ae_anomaly   : bool   (True if ae_error > ae_threshold)
        """
        # 1. Build unsupervised feature vector
        X_unsup = self._build_unsupervised_vector(transaction)

        # 2. Isolation Forest score (lower = more anomalous)
        if_score = float(self.if_model.decision_function(X_unsup)[0])

        # 3. Autoencoder reconstruction error (higher = more anomalous)
        X_scaled = self.ae_scaler.transform(X_unsup)
        reconstruction = self.ae_model.predict(X_scaled)
        ae_error = float(np.mean(np.power(X_scaled - reconstruction, 2)))
        ae_anomaly = ae_error > self.ae_threshold

        # 4. Build XGBoost feature vector (17 raw features — independent of IF/AE)
        X_xgb = self._build_xgboost_vector(transaction)

        # 5. XGBoost fraud probability
        fraud_score = float(self.xgb_model.predict_proba(X_xgb)[0, 1])
        is_fraud = fraud_score >= 0.5

        # 6. Confidence band
        if fraud_score >= 0.85 or fraud_score <= 0.15:
            confidence = "HIGH"
        elif fraud_score >= 0.65 or fraud_score <= 0.35:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "fraud_score":  round(fraud_score, 6),
            "is_fraud":     is_fraud,
            "confidence":   confidence,
            "if_score":     round(if_score, 6),
            "ae_error":     round(ae_error, 6),
            "ae_threshold": round(self.ae_threshold, 6),
            "ae_anomaly":   ae_anomaly,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_unsupervised_vector(self, tx: dict) -> pd.DataFrame:
        """
        Builds a single-row DataFrame with the 13 features used by IF and AE,
        with categoricals label-encoded using the known-category lists.
        """
        row = {}
        for feat in UNSUPERVISED_FEATURES:
            if feat in CATEGORICAL_COLS:
                row[feat] = _encode_category(tx.get(feat), feat)
            else:
                row[feat] = float(tx.get(feat, 0) or 0)
        return pd.DataFrame([row])

    def _build_xgboost_vector(self, tx: dict) -> pd.DataFrame:
        """
        Builds the 17-feature vector for XGBoost.
        Raw transaction features only — no anomaly scores appended.
        Matches the feature set used during training (train_xgboost.py).
        """
        row = {}
        for feat in XGBOOST_BASE_FEATURES:
            if feat in CATEGORICAL_COLS:
                row[feat] = _encode_category(tx.get(feat), feat)
            else:
                row[feat] = float(tx.get(feat, 0) or 0)
        return pd.DataFrame([row])

    def _check_paths(self):
        """Raises FileNotFoundError early if any model file is missing."""
        paths = {
            'Isolation Forest':  IF_MODEL_PATH,
            'Autoencoder model': AE_MODEL_PATH,
            'Autoencoder scaler': AE_SCALER_PATH,
            'AE threshold':      AE_THRESH_PATH,
            'XGBoost model':     XGB_MODEL_PATH,
        }
        missing = [name for name, path in paths.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"[FraudScorer] Missing model files: {missing}\n"
                "Please run the training scripts before using predict.py."
            )


# ---------------------------------------------------------------------------
# Convenience wrapper for one-shot use
# ---------------------------------------------------------------------------
_default_scorer: FraudScorer | None = None


def score_transaction(transaction: dict) -> dict:
    """
    Module-level convenience function.
    Loads models on first call (singleton), then reuses them.

    Example:
        from predict import score_transaction
        result = score_transaction({"amount": 5000, "geo_country": "GH", ...})
    """
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = FraudScorer()
    return _default_scorer.score(transaction)
