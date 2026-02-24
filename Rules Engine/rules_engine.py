"""
rules_engine.py — Bank Fraud Detection: Rules Engine (Layer 1)
==============================================================
Evaluates all defined rules against an incoming transaction and produces a
final Rules Engine flag:

  ALLOW  — all rules passed; transaction forwarded to ML layer
  FLAG   — one or more rules fired; transaction forwarded to ML layer with a
           caution flag (Decision Engine will escalate to CHALLENGE)
  BLOCK  — a hard-stop rule fired OR >= block_threshold_pct of all rules failed;
           transaction is immediately DECLINED (early exit — ML layer skipped)

Usage
-----
    from rules_engine import evaluate_rules

    result = evaluate_rules(transaction_dict)
    # result = {
    #     "rules_flag":      "FLAG",          # "ALLOW" | "FLAG" | "BLOCK"
    #     "triggered_rules": ["velocity_high_1h", "geo_country_mismatch"],
    #     "block_rules_hit": [],              # BLOCK-severity rules that fired
    #     "total_rules":     16,
    #     "rules_failed":    2,
    #     "failure_pct":     0.125,
    # }
"""

import os
import json
from typing import List, Dict, Any

from rules import ALL_RULES, BLOCK as SEVERITY_BLOCK, FLAG as SEVERITY_FLAG


# ---------------------------------------------------------------------------
# Load threshold config from decision_config.json
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "Decision Layer",
    "decision_config.json",
)

def _load_block_threshold() -> float:
    """Read block_threshold_pct from decision_config.json."""
    try:
        with open(_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        return float(cfg.get("rules_engine", {}).get("block_threshold_pct", 0.40))
    except (FileNotFoundError, KeyError, ValueError):
        return 0.40   # safe default


BLOCK_THRESHOLD_PCT: float = _load_block_threshold()


# ---------------------------------------------------------------------------
# RulesEngine class
# ---------------------------------------------------------------------------
class RulesEngine:
    """
    Evaluates the full rule set against a transaction dict.

    Parameters
    ----------
    block_threshold_pct : float
        Fraction of rules that must fail for an automatic BLOCK escalation.
        Loaded from decision_config.json; default = 0.40.

    Example
    -------
        engine = RulesEngine()
        result = engine.evaluate(transaction)
    """

    def __init__(self, block_threshold_pct: float = BLOCK_THRESHOLD_PCT):
        self.rules               = ALL_RULES
        self.block_threshold_pct = block_threshold_pct

    def evaluate(self, transaction: dict) -> Dict[str, Any]:
        """
        Run all rules against the transaction.

        Parameters
        ----------
        transaction : dict
            Raw transaction fields (matching TransactionRequest field names).

        Returns
        -------
        dict with keys:
            rules_flag      : str   — "ALLOW" | "FLAG" | "BLOCK"
            triggered_rules : list  — names of all rules that fired
            block_rules_hit : list  — names of BLOCK-severity rules that fired
            flag_rules_hit  : list  — names of FLAG-severity rules that fired
            total_rules     : int   — total rules evaluated
            rules_failed    : int   — number of rules that fired
            failure_pct     : float — rules_failed / total_rules
        """
        triggered_rules: List[str] = []
        block_rules_hit: List[str] = []
        flag_rules_hit:  List[str] = []

        for rule in self.rules:
            try:
                fired = rule.condition(transaction)
            except Exception:
                fired = False   # never let a bad rule crash the engine

            if fired:
                triggered_rules.append(rule.name)
                if rule.severity == SEVERITY_BLOCK:
                    block_rules_hit.append(rule.name)
                else:
                    flag_rules_hit.append(rule.name)

        total_rules  = len(self.rules)
        rules_failed = len(triggered_rules)
        failure_pct  = rules_failed / total_rules if total_rules > 0 else 0.0

        # ---------------------------------------------------------------
        # Determine final flag
        # ---------------------------------------------------------------
        if block_rules_hit:
            # One or more BLOCK-severity rules fired → hard stop
            rules_flag = "BLOCK"

        elif failure_pct >= self.block_threshold_pct:
            # No single BLOCK rule but enough rules failed → escalate to BLOCK
            rules_flag = "BLOCK"

        elif flag_rules_hit:
            # One or more FLAG-severity rules fired → forward with caution
            rules_flag = "FLAG"

        else:
            # All rules passed
            rules_flag = "ALLOW"

        return {
            "rules_flag":      rules_flag,
            "triggered_rules": triggered_rules,
            "block_rules_hit": block_rules_hit,
            "flag_rules_hit":  flag_rules_hit,
            "total_rules":     total_rules,
            "rules_failed":    rules_failed,
            "failure_pct":     round(failure_pct, 4),
        }


# ---------------------------------------------------------------------------
# Module-level convenience wrapper (singleton)
# ---------------------------------------------------------------------------
_default_engine: RulesEngine | None = None


def _get_engine() -> RulesEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = RulesEngine()
    return _default_engine


def evaluate_rules(transaction: dict) -> dict:
    """
    Module-level convenience function.
    Evaluates all rules and returns the result dict.

    Parameters
    ----------
    transaction : dict — raw transaction fields

    Returns
    -------
    dict — see RulesEngine.evaluate()
    """
    return _get_engine().evaluate(transaction)
