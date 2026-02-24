"""
action_layer service — Port 8006
==================================
STUB: Accepts a FraudVerdict, logs it, and returns an ActionAck.

Full action implementations (OTP trigger, card block, fraud alert)
are tracked in action_list.txt items 4.1–4.3 and will be wired in
once the downstream integrations are ready.
"""

import os
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from shared.schemas import FraudVerdict, ActionAck

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("action_layer")

app = FastAPI(
    title="Action Layer Service",
    description="Layer 4 (stub): Receives a FraudVerdict and dispatches the appropriate action.",
    version="1.0.0",
)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "action_layer", "port": 8006}


@app.post("/execute", response_model=ActionAck, tags=["Actions"])
def execute(verdict: FraudVerdict):
    """
    Execute the appropriate action based on the fraud verdict.

    Current stub behaviour: logs the verdict and returns LOGGED.

    Future actions (from action_list.txt):
    - DECLINE  → block card / freeze account
    - CHALLENGE → trigger OTP / step-up authentication
    - APPROVE  → no action required
    """
    try:
        logger.info(
            "ACTION | decision=%s is_early_exit=%s confidence=%s triggered_by=%s",
            verdict.decision,
            verdict.is_early_exit,
            verdict.confidence,
            verdict.triggered_by,
        )

        action_taken = "LOGGED"

        # ------------------------------------------------------------------
        # TODO: Replace stubs below with real integrations
        # ------------------------------------------------------------------
        # if verdict.decision == "DECLINE":
        #     action_taken = block_card(verdict)          # item 4.1
        # elif verdict.decision == "CHALLENGE":
        #     action_taken = trigger_otp(verdict)         # item 4.2
        # else:
        #     action_taken = "NO_ACTION_REQUIRED"         # item 4.3
        # ------------------------------------------------------------------

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action layer error: {str(e)}")

    return ActionAck(
        action_taken=action_taken,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
