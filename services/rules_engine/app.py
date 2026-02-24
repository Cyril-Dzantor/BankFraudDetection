"""
rules_engine service — Port 8001
=================================
Wraps the existing RulesEngine class from the monolith's Rules Engine layer.
Exposes a single POST /evaluate endpoint that accepts a TransactionRequest
and returns a RulesResult.
"""

import sys
import os

# Make the project root importable so we can reach Rules Engine/rules_engine.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Rules Engine"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Decision Layer"))  # needed by rules_engine.py for config

from fastapi import FastAPI, HTTPException
from shared.schemas import TransactionRequest, RulesResult
from rules_engine import evaluate_rules   # real RulesEngine — no stub

app = FastAPI(
    title="Rules Engine Service",
    description="Layer 1: Evaluates business rules and returns ALLOW / FLAG / BLOCK.",
    version="1.0.0",
)


@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "rules_engine", "port": 8001}


@app.post("/evaluate", response_model=RulesResult, tags=["Rules"])
def evaluate(transaction: TransactionRequest):
    """
    Evaluate all business rules against the incoming transaction.

    Returns a RulesResult with:
    - rules_flag: "ALLOW" | "FLAG" | "BLOCK"
    - triggered_rules: list of rule names that fired
    - block_rules_hit, flag_rules_hit, total_rules, rules_failed, failure_pct
    """
    try:
        result = evaluate_rules(transaction.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rules engine error: {str(e)}")

    return RulesResult(**result)
