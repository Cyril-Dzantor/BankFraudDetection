"""
schemas.py — Shared Pydantic Models
=====================================
Single source of truth for all request/response schemas
used across the microservice network.

Import in any service with:
    from shared.schemas import TransactionRequest, FraudVerdict, ...
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ---------------------------------------------------------------------------
# INPUT — Transaction (same fields as root models.py)
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """A single transaction to be scored. Fields match model training feature names."""

    # Velocity
    tx_count_last_5m:    float = Field(..., description="Number of transactions in the last 5 minutes")
    tx_count_last_1h:    float = Field(..., description="Number of transactions in the last 1 hour")
    tx_frequency_ratio:  float = Field(..., description="Current frequency vs. historical average ratio")

    # Amount
    amount:              float = Field(..., description="Transaction amount")
    avg_tx_amount_7d:    float = Field(..., description="Average transaction amount over last 7 days")
    amount_to_avg_ratio: float = Field(..., description="This transaction's amount divided by the 7-day average")

    # Device / identity
    new_device_flag:            int   = Field(..., description="1 if this is a device not seen before, else 0")
    device_seen_before:         int   = Field(..., description="1 if device is known, else 0")
    country_mismatch_flag:      int   = Field(..., description="1 if transaction country differs from account's usual country")
    accounts_per_device:        float = Field(..., description="Number of accounts linked to this device")
    accounts_per_ip_24h:        float = Field(..., description="Number of unique accounts seen from this IP in 24 hours")

    # Auth / login
    failed_login_count_last_1h: float = Field(..., description="Number of failed login attempts in the last 1 hour")
    failed_logins_then_success: int   = Field(..., description="1 if there were failed logins immediately before this success")

    # Categorical
    channel:     str = Field(..., description="Transaction channel: 'mobile', 'web', 'atm', 'pos'")
    auth_method: str = Field(..., description="Authentication method: 'pin', 'biometric', 'otp', 'password'")
    geo_country: str = Field(..., description="Country code of the transaction, e.g. 'GH', 'NG'")
    currency:    str = Field(..., description="Currency code, e.g. 'GHS', 'USD'")

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
# RULES ENGINE — output of POST /evaluate
# ---------------------------------------------------------------------------
class RulesResult(BaseModel):
    """Output from the Rules Engine service (POST /evaluate)."""
    rules_flag:      str       = Field(..., description="'ALLOW' | 'FLAG' | 'BLOCK'")
    triggered_rules: List[str] = Field(default_factory=list, description="Names of all rules that fired")
    block_rules_hit: List[str] = Field(default_factory=list, description="BLOCK-severity rules that fired")
    flag_rules_hit:  List[str] = Field(default_factory=list, description="FLAG-severity rules that fired")
    total_rules:     int       = Field(..., description="Total number of rules evaluated")
    rules_failed:    int       = Field(..., description="Number of rules that fired")
    failure_pct:     float     = Field(..., description="rules_failed / total_rules")


# ---------------------------------------------------------------------------
# ML SCORES — outputs of each individual ML service
# ---------------------------------------------------------------------------
class IFScore(BaseModel):
    """Output from the Isolation Forest service (POST /score)."""
    if_score: float = Field(..., description="Isolation Forest decision_function score (lower = more anomalous)")


class AEScore(BaseModel):
    """Output from the Autoencoder service (POST /score)."""
    ae_error:     float = Field(..., description="Autoencoder MSE reconstruction error (higher = more anomalous)")
    ae_threshold: float = Field(..., description="Threshold above which ae_error is flagged as anomalous")
    ae_anomaly:   bool  = Field(..., description="True if ae_error > ae_threshold")


class XGBScore(BaseModel):
    """Output from the XGBoost Classifier service (POST /score)."""
    fraud_score: float = Field(..., description="XGBoost fraud probability (0.0–1.0)")
    is_fraud:    bool  = Field(..., description="True if fraud_score >= 0.5")


# ---------------------------------------------------------------------------
# DECISION ENGINE — input + output
# ---------------------------------------------------------------------------
class DecideRequest(BaseModel):
    """Input for the Decision Engine service (POST /decide)."""
    fraud_score:          float     = Field(..., description="XGBoost fraud probability")
    if_score:             float     = Field(..., description="Isolation Forest score")
    ae_error:             float     = Field(..., description="Autoencoder reconstruction error")
    rules_flag:           str       = Field(..., description="'ALLOW' | 'FLAG'")
    triggered_rules:      List[str] = Field(default_factory=list, description="Rules that fired")


class FraudVerdict(BaseModel):
    """Final fraud decision — returned to the client by the Orchestrator."""
    decision:             str            = Field(..., description="'APPROVE', 'CHALLENGE', or 'DECLINE'")
    is_early_exit:        bool           = Field(..., description="True if Rules Engine issued BLOCK and ML was skipped")
    fraud_score:          Optional[float] = Field(None, description="XGBoost score. None if early exit.")
    anomaly_score:        Optional[float] = Field(None, description="Isolation Forest score. None if early exit.")
    reconstruction_error: Optional[float] = Field(None, description="Autoencoder MSE. None if early exit.")
    confidence:           str            = Field(..., description="'HIGH', 'MEDIUM', or 'LOW'")
    triggered_by:         str            = Field(..., description="What caused this decision")
    reason:               str            = Field(..., description="Human-readable explanation of the verdict")
    timestamp:            str            = Field(..., description="UTC ISO-8601 timestamp")


# ---------------------------------------------------------------------------
# ACTION LAYER — output of POST /execute
# ---------------------------------------------------------------------------
class ActionAck(BaseModel):
    """Acknowledgement from the Action Layer service (POST /execute)."""
    action_taken: str = Field(..., description="Action that was taken, e.g. 'LOGGED', 'OTP_TRIGGERED'")
    timestamp:    str = Field(..., description="UTC ISO-8601 timestamp")


# ---------------------------------------------------------------------------
# POST-AUTH INTELLIGENCE — input of POST /record
# ---------------------------------------------------------------------------
class PostAuthPayload(BaseModel):
    """Payload sent to the Post-Auth Intelligence service (POST /record)."""
    transaction: TransactionRequest
    verdict:     FraudVerdict
    timestamp:   str = Field(..., description="UTC ISO-8601 timestamp of the scoring event")
