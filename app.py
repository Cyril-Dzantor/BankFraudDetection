"""
app.py — Bank Fraud Detection API (MONOLITH — DEPRECATED)
===========================================================
⚠  This file is the original single-process entry point.
   It is kept for reference only and is NO LONGER the live API.

   The live entry point is: services/orchestrator/app.py
   Run the full microservices stack with:  docker compose up --build

   DO NOT add new features here. All changes go into services/.
==================================================================

To run:
    uvicorn app:app --reload

Then open:
    http://127.0.0.1:8000/health        ← health check
    http://127.0.0.1:8000/docs          ← Swagger UI (try /score interactively)
"""

import sys
import os
from fastapi import FastAPI, HTTPException

from models import TransactionRequest, FraudVerdict

# ---------------------------------------------------------------------------
# Import the two core modules from our pipeline
# ---------------------------------------------------------------------------
# predict.py lives inside "Anomally Detection/" — we add it to the path
# so Python can find it from app.py which is in the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Anomally Detection"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Decision Layer"))

from predict import score_transaction          # ML layer: IF + AE + XGBoost
from decision_engine import make_decision, make_early_exit_decision  # Decision Layer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Rules Engine"))
from rules_engine import evaluate_rules        # Rules Engine: ALLOW / FLAG / BLOCK

# ---------------------------------------------------------------------------
# Create the FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bank Fraud Detection API",
    description="Real-time transaction fraud scoring using Isolation Forest, Autoencoder, and XGBoost.",
    version="0.2.0",
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Returns a simple status message.
    Monitoring tools and load balancers ping this to confirm the server is alive.
    """
    return {
        "status": "ok",
        "service": "Bank Fraud Detection API",
        "version": "0.2.0",
    }


# ---------------------------------------------------------------------------
# POST /score
# ---------------------------------------------------------------------------
@app.post("/score", response_model=FraudVerdict, tags=["Fraud Detection"])
def score(transaction: TransactionRequest):
    """
    Score a single transaction for fraud.

    **Flow:**
    1. FastAPI validates the incoming JSON against `TransactionRequest`
    2. The transaction is passed to the ML pipeline (Isolation Forest + Autoencoder + XGBoost)
    3. The ML scores are passed to the Decision Engine
    4. A `FraudVerdict` is returned with the final decision

    **Decision outcomes:**
    - `APPROVE` — transaction appears legitimate
    - `CHALLENGE` — suspicious, trigger MFA / OTP verification
    - `DECLINE` — high fraud probability, block the transaction
    """

    # ------------------------------------------------------------------
    # Step A: Run the transaction through the Rules Engine
    # ------------------------------------------------------------------
    try:
        rules_result = evaluate_rules(transaction.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rules engine error: {str(e)}")

    rules_flag      = rules_result["rules_flag"]
    triggered_rules = rules_result["triggered_rules"]

    # BLOCK → early exit; skip the ML layer entirely
    if rules_flag == "BLOCK":
        verdict = make_early_exit_decision(rules_triggered=triggered_rules)
        return FraudVerdict(
            decision             = verdict["decision"],
            is_early_exit        = verdict.get("is_early_exit", True),
            fraud_score          = verdict.get("fraud_score"),
            anomaly_score        = verdict.get("anomaly_score"),
            reconstruction_error = verdict.get("reconstruction_error"),
            confidence           = verdict["confidence"],
            triggered_by         = verdict["triggered_by"],
            reason               = verdict["reason"],
            timestamp            = verdict["timestamp"],
        )

    # ------------------------------------------------------------------
    # Step B: Run the transaction through the ML pipeline
    # ------------------------------------------------------------------
    # score_transaction() expects a plain dict, so we convert the Pydantic
    # model using .model_dump()
    try:
        ml_result = score_transaction(transaction.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML pipeline error: {str(e)}")

    # ml_result looks like:
    # {
    #   "fraud_score": 0.921,
    #   "is_fraud": True,
    #   "if_score": -0.08,              ← Isolation Forest raw score
    #   "ae_error": 0.45,               ← Autoencoder reconstruction error
    #   "ae_threshold": 0.310989,
    #   "ae_anomaly": True,
    # }

    # ------------------------------------------------------------------
    # Step C: Pass ML scores to the Decision Engine
    # ------------------------------------------------------------------
    # Rules Engine flag is now live (ALLOW or FLAG at this point — BLOCK
    # was already handled above as an early exit).
    try:
        verdict = make_decision(
            fraud_score          = ml_result["fraud_score"],
            rules_flag           = rules_flag,
            anomaly_score        = ml_result["if_score"],
            reconstruction_error = ml_result["ae_error"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision engine error: {str(e)}")

    # ------------------------------------------------------------------
    # Step C: Build and return the FraudVerdict response
    # ------------------------------------------------------------------
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
