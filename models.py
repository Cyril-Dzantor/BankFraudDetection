"""
models.py — API Request and Response Schemas
=============================================
Pydantic models define the exact shape of data flowing into and out of the API.

Why Pydantic?
  - FastAPI uses these to automatically validate incoming JSON.
  - If a required field is missing or the wrong type, the API returns a clear
    422 error before our fraud code even runs.
  - The fields and types are also used to auto-generate the /docs Swagger UI.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# REQUEST MODEL — what the client sends us
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """
    A single transaction to be scored.
    All fields match the feature names used during model training.
    """

    # --- Velocity features ---
    # How many times has this account transacted recently?
    tx_count_last_5m:    float = Field(..., description="Number of transactions in the last 5 minutes")
    tx_count_last_1h:    float = Field(..., description="Number of transactions in the last 1 hour")
    tx_frequency_ratio:  float = Field(..., description="Current frequency vs. historical average ratio")

    # --- Amount features ---
    amount:              float = Field(..., description="Transaction amount")
    avg_tx_amount_7d:    float = Field(..., description="Average transaction amount over last 7 days")
    amount_to_avg_ratio: float = Field(..., description="This transaction's amount divided by the 7-day average")

    # --- Device / identity features ---
    new_device_flag:             int   = Field(..., description="1 if this is a device not seen before, else 0")
    device_seen_before:          int   = Field(..., description="1 if device is known, else 0")
    country_mismatch_flag:       int   = Field(..., description="1 if transaction country differs from account's usual country")
    accounts_per_device:         float = Field(..., description="Number of accounts linked to this device")
    accounts_per_ip_24h:         float = Field(..., description="Number of unique accounts seen from this IP in 24 hours")

    # --- Authentication / login features ---
    failed_login_count_last_1h:  float = Field(..., description="Number of failed login attempts in the last 1 hour")
    failed_logins_then_success:  int   = Field(..., description="1 if there were failed logins immediately before this success")

    # --- Categorical features ---
    channel:     str = Field(..., description="Transaction channel: 'mobile', 'web', 'atm', 'pos'")
    auth_method: str = Field(..., description="Authentication method: 'pin', 'biometric', 'otp', 'password'")
    geo_country: str = Field(..., description="Country code of the transaction, e.g. 'GH', 'NG'")
    currency:    str = Field(..., description="Currency code, e.g. 'GHS', 'USD'")

    # Example payload shown in the /docs Swagger UI
    model_config = {
        "json_schema_extra": {
            "example": {
                "tx_count_last_5m": 1,
                "tx_count_last_1h": 2,
                "tx_frequency_ratio": 0.9,
                "amount": 95.00,
                "avg_tx_amount_7d": 110.00,
                "amount_to_avg_ratio": 0.86,
                "new_device_flag": 0,
                "device_seen_before": 1,
                "country_mismatch_flag": 0,
                "accounts_per_device": 1,
                "accounts_per_ip_24h": 1,
                "failed_login_count_last_1h": 0,
                "failed_logins_then_success": 0,
                "channel": "mobile",
                "auth_method": "biometric",
                "geo_country": "GH",
                "currency": "GHS",
            }
        }
    }


# ---------------------------------------------------------------------------
# RESPONSE MODEL — what we send back to the client
# ---------------------------------------------------------------------------
class FraudVerdict(BaseModel):
    """
    The fraud scoring result returned to the client after a /score call.
    """

    # Final decision
    decision:    str  = Field(..., description="'APPROVE', 'CHALLENGE', or 'DECLINE'")
    is_early_exit: bool = Field(..., description="True if the Rules Engine issued a BLOCK and the ML layer was skipped")

    # ML scores
    fraud_score:          Optional[float] = Field(None, description="XGBoost fraud probability (0.0–1.0). None if early exit.")
    anomaly_score:        Optional[float] = Field(None, description="Isolation Forest score (lower = more anomalous). None if early exit.")
    reconstruction_error: Optional[float] = Field(None, description="Autoencoder MSE (higher = more anomalous). None if early exit.")

    # Decision metadata
    confidence:   str = Field(..., description="'HIGH', 'MEDIUM', or 'LOW' — certainty of the decision")
    triggered_by: str = Field(..., description="What caused this decision: ML_SCORE, RULES_FLAG, RULES_BLOCK_EARLY_EXIT, etc.")
    reason:       str = Field(..., description="Human-readable explanation of the verdict")
    timestamp:    str = Field(..., description="UTC timestamp of when the decision was made")
