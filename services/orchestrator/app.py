"""
orchestrator service — Port 8000
==================================
Public-facing gateway. Coordinates the full fraud detection pipeline:

  1. Rules Engine   — POST rules_engine:8001/evaluate
     [BLOCK]  → early exit: call Action Layer + return DECLINE verdict
  2. ML Layer       — asyncio.gather (all 3 in parallel):
       POST isolation_forest:8002/score
       POST autoencoder:8003/score
       POST xgboost_classifier:8004/score
  3. Decision Engine — POST decision_engine:8005/decide
  4. Fire-and-forget (background):
       POST action_layer:8006/execute
       POST post_auth_intelligence:8007/record
  5. Return FraudVerdict to the client
"""

import asyncio
import os
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from shared.schemas import (
    TransactionRequest, FraudVerdict,
    RulesResult, IFScore, AEScore, XGBScore,
    DecideRequest, ActionAck, PostAuthPayload,
)

# ---------------------------------------------------------------------------
# Service URLs — override via environment variables in docker-compose
# ---------------------------------------------------------------------------
RULES_ENGINE_URL          = os.getenv("RULES_ENGINE_URL",          "http://rules_engine:8001")
ISOLATION_FOREST_URL      = os.getenv("ISOLATION_FOREST_URL",      "http://isolation_forest:8002")
AUTOENCODER_URL           = os.getenv("AUTOENCODER_URL",           "http://autoencoder:8003")
XGBOOST_URL               = os.getenv("XGBOOST_URL",               "http://xgboost_classifier:8004")
DECISION_ENGINE_URL       = os.getenv("DECISION_ENGINE_URL",       "http://decision_engine:8005")
ACTION_LAYER_URL          = os.getenv("ACTION_LAYER_URL",          "http://action_layer:8006")
POST_AUTH_INTELLIGENCE_URL= os.getenv("POST_AUTH_INTELLIGENCE_URL","http://post_auth_intelligence:8007")

TIMEOUT = httpx.Timeout(10.0)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bank Fraud Detection — Orchestrator",
    description="Microservices gateway. Submit a transaction to POST /score.",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Helper: call a downstream service, raise on error
# ---------------------------------------------------------------------------
async def _post(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    try:
        resp = await client.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream error from {url}: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach {url}: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Background fire-and-forget helper
# ---------------------------------------------------------------------------
async def _fire_and_forget(url: str, payload: dict):
    """POST to a service but do not block the response on result."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload, timeout=TIMEOUT)
    except Exception:
        pass  # non-critical — do not surface errors to the client


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "ok",
        "service": "orchestrator",
        "version": "2.0.0",
        "port": 8000,
    }


# ---------------------------------------------------------------------------
# POST /score  — full pipeline
# ---------------------------------------------------------------------------
@app.post("/score", response_model=FraudVerdict, tags=["Fraud Detection"])
async def score(transaction: TransactionRequest):
    """
    Score a single transaction for fraud through the full microservices pipeline.

    **Pipeline:**
    1. Rules Engine → ALLOW / FLAG / BLOCK
    2. [BLOCK] → immediate DECLINE (early exit, ML skipped)
    3. [ALLOW / FLAG] → Isolation Forest + Autoencoder + XGBoost (parallel)
    4. Decision Engine → APPROVE / CHALLENGE / DECLINE
    5. Action Layer + Post-Auth Intelligence (fire-and-forget)
    """
    tx_dict = transaction.model_dump()

    async with httpx.AsyncClient() as client:

        # ------------------------------------------------------------------
        # Step 1: Rules Engine
        # ------------------------------------------------------------------
        rules_raw = await _post(client, f"{RULES_ENGINE_URL}/evaluate", tx_dict)
        rules     = RulesResult(**rules_raw)

        # ------------------------------------------------------------------
        # Step 2: BLOCK → early exit
        # ------------------------------------------------------------------
        if rules.rules_flag == "BLOCK":
            verdict = FraudVerdict(
                decision             = "DECLINE",
                is_early_exit        = True,
                fraud_score          = None,
                anomaly_score        = None,
                reconstruction_error = None,
                confidence           = "HIGH",
                triggered_by         = "RULES_BLOCK_EARLY_EXIT",
                reason               = (
                    f"Transaction blocked by Rules Engine. "
                    f"Rules violated: [{', '.join(rules.triggered_rules)}]."
                ),
                timestamp            = datetime.now(timezone.utc).isoformat(),
            )
            # Fire-and-forget: notify Action Layer
            asyncio.create_task(
                _fire_and_forget(f"{ACTION_LAYER_URL}/execute", verdict.model_dump())
            )
            return verdict

        # ------------------------------------------------------------------
        # Step 3: ML Layer — all three models in parallel
        # ------------------------------------------------------------------
        if_raw, ae_raw, xgb_raw = await asyncio.gather(
            _post(client, f"{ISOLATION_FOREST_URL}/score",  tx_dict),
            _post(client, f"{AUTOENCODER_URL}/score",       tx_dict),
            _post(client, f"{XGBOOST_URL}/score",           tx_dict),
        )
        if_score  = IFScore(**if_raw)
        ae_score  = AEScore(**ae_raw)
        xgb_score = XGBScore(**xgb_raw)

        # ------------------------------------------------------------------
        # Step 4: Decision Engine
        # ------------------------------------------------------------------
        decide_payload = DecideRequest(
            fraud_score     = xgb_score.fraud_score,
            if_score        = if_score.if_score,
            ae_error        = ae_score.ae_error,
            rules_flag      = rules.rules_flag,
            triggered_rules = rules.triggered_rules,
        )
        verdict_raw = await _post(
            client, f"{DECISION_ENGINE_URL}/decide", decide_payload.model_dump()
        )
        verdict = FraudVerdict(**verdict_raw)

    # ------------------------------------------------------------------
    # Step 5: Fire-and-forget — Action Layer + Post-Auth Intelligence
    # ------------------------------------------------------------------
    now_ts = datetime.now(timezone.utc).isoformat()

    asyncio.create_task(_fire_and_forget(
        f"{ACTION_LAYER_URL}/execute",
        verdict.model_dump(),
    ))
    asyncio.create_task(_fire_and_forget(
        f"{POST_AUTH_INTELLIGENCE_URL}/record",
        PostAuthPayload(
            transaction=transaction,
            verdict=verdict,
            timestamp=now_ts,
        ).model_dump(),
    ))

    return verdict
