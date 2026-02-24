"""
post_auth_intelligence service — Port 8007
============================================
STUB: Receives a completed scoring event (transaction + verdict)
and appends it as a JSON line to retraining_data.jsonl.

Full profiling and model-monitoring logic (action_list.txt items 5.1–5.4)
will be built separately once retraining infrastructure is ready.
"""

import os
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from shared.schemas import PostAuthPayload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("post_auth_intelligence")

# Append scored events here for future retraining / drift monitoring
DATA_FILE = os.path.join(os.path.dirname(__file__), "retraining_data.jsonl")

app = FastAPI(
    title="Post-Auth Intelligence Service",
    description="Layer 5 (stub): Records scoring events for model retraining and drift monitoring.",
    version="1.0.0",
)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "post_auth_intelligence", "port": 8007}


@app.post("/record", tags=["Intelligence"])
def record(payload: PostAuthPayload):
    """
    Record a completed scoring event for post-authorisation analysis.

    Appends a JSON line to retraining_data.jsonl for later use in:
    - Model drift detection          (action_list.txt item 5.1)
    - Automated retraining triggers  (item 5.2)
    - Fraud pattern analysis         (item 5.3)
    - False-positive feedback loop   (item 5.4)
    """
    try:
        record_dict = {
            "timestamp":   payload.timestamp,
            "decision":    payload.verdict.decision,
            "is_early_exit": payload.verdict.is_early_exit,
            "fraud_score": payload.verdict.fraud_score,
            "transaction": payload.transaction.model_dump(),
        }
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict) + "\n")

        logger.info(
            "RECORDED | decision=%s fraud_score=%s ts=%s",
            payload.verdict.decision,
            payload.verdict.fraud_score,
            payload.timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-auth record error: {str(e)}")

    return {"recorded": True, "timestamp": datetime.now(timezone.utc).isoformat()}
