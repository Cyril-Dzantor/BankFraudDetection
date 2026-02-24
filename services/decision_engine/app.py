"""
decision_engine service — Port 8005
=====================================
Wraps the existing DecisionEngine class from Decision Layer/decision_engine.py.
Accepts ML scores + rules flag and returns a FraudVerdict.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Decision Layer"))

from fastapi import FastAPI, HTTPException
from shared.schemas import DecideRequest, FraudVerdict
from decision_engine import make_decision   # real DecisionEngine — no logic changes

app = FastAPI(
    title="Decision Engine Service",
    description="Layer 3: Combines ML scores and rules flag to produce a final FraudVerdict.",
    version="1.0.0",
)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "decision_engine", "port": 8005}


@app.post("/decide", response_model=FraudVerdict, tags=["Decision"])
def decide(payload: DecideRequest):
    """
    Produce a final FraudVerdict from the ML scores and Rules Engine flag.

    Input:
    - fraud_score: XGBoost probability
    - if_score: Isolation Forest decision_function value
    - ae_error: Autoencoder reconstruction MSE
    - rules_flag: "ALLOW" | "FLAG"
    - triggered_rules: list of rules that fired

    Returns a FraudVerdict (APPROVE / CHALLENGE / DECLINE).
    """
    try:
        verdict = make_decision(
            fraud_score          = payload.fraud_score,
            rules_flag           = payload.rules_flag,
            anomaly_score        = payload.if_score,
            reconstruction_error = payload.ae_error,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision engine error: {str(e)}")

    return FraudVerdict(
        decision             = verdict["decision"],
        is_early_exit        = verdict.get("is_early_exit", False),
        fraud_score          = verdict.get("fraud_score"),
        anomaly_score        = verdict.get("anomaly_score"),
        reconstruction_error = verdict.get("reconstruction_error"),
        confidence           = verdict["confidence"],
        triggered_by         = verdict["triggered_by"],
        reason               = verdict["reason"],
        timestamp            = verdict["timestamp"],
    )
