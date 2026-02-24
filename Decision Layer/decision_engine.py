"""
decision_engine.py — Bank Fraud Detection: Decision Layer (Layer 3)
====================================================================
Accepts THREE signals from the ML layer and applies configurable
threshold logic to produce a final transaction verdict:

  1. fraud_score             — XGBoost probability (0.0–1.0)
  2. anomaly_score           — Isolation Forest decision_function score
                               (lower / more negative = more anomalous)
  3. reconstruction_error    — Autoencoder MSE (higher = more anomalous)

Rules Engine flag semantics:
  ┌──────────┬──────────────────────────┬────────────────────────────────┐
  │ Flag     │ Rules Engine condition   │ Action in pipeline             │
  ├──────────┼──────────────────────────┼────────────────────────────────┤
  │ ALLOW    │ All rules passed         │ → ML layer → Decision Engine   │
  │ FLAG     │ One or more rules failed │ → ML layer → Decision Engine   │
  │ BLOCK    │ Failed ≥ threshold % of  │ EARLY EXIT — ML skipped,       │
  │          │ rules                    │ Decision Engine returns DECLINE │
  └──────────┴──────────────────────────┴────────────────────────────────┘

Decision Logic (for ALLOW / FLAG paths that reach the Decision Engine):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ fraud_score ≥ decline_threshold                → DECLINE             │
  │ anomaly boost: IF/AE flags + score in band     → escalate to DECLINE │
  │ rules_flag = FLAG (regardless of score)        → CHALLENGE           │
  │ fraud_score ≥ challenge_threshold              → CHALLENGE           │
  │ Otherwise                                      → APPROVE             │
  └──────────────────────────────────────────────────────────────────────┘

Usage:
    from decision_engine import DecisionEngine

    engine = DecisionEngine()

    # Normal flow (all three ML signals provided)
    verdict = engine.decide(
        fraud_score=0.73,
        rules_flag="FLAG",
        anomaly_score=-0.05,          # from Isolation Forest
        reconstruction_error=0.42     # from Autoencoder
    )

    # Early exit (BLOCK from Rules Engine — ML layer was skipped)
    verdict = engine.early_exit_block(rules_triggered=["velocity", "geo", "device"])
"""

import os
import json
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'decision_config.json')

# ---------------------------------------------------------------------------
# Valid Rules Engine flags
# ---------------------------------------------------------------------------
VALID_RULES_FLAGS = {"ALLOW", "FLAG", "BLOCK"}


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """
    Applies threshold-based logic to produce a final transaction verdict.
    Accepts three ML signals and combines them for a richer decision.

    Two entry points:
      1. decide(fraud_score, rules_flag, anomaly_score, reconstruction_error)
                                     — normal flow (ALLOW / FLAG)
      2. early_exit_block(rules_triggered)   — fast path for BLOCK exits

    Verdict keys returned by both methods:
        decision                 : str   — "APPROVE" | "CHALLENGE" | "DECLINE"
        fraud_score              : float | None
        anomaly_score            : float | None  — Isolation Forest score
        reconstruction_error     : float | None  — Autoencoder MSE
        rules_flag               : str
        is_early_exit            : bool
        confidence               : str   — "HIGH" | "MEDIUM" | "LOW"
        reason                   : str
        triggered_by             : str
        timestamp                : str   — UTC ISO-8601
    """

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config     = self._load_config(config_path)
        self.thresh     = self.config["thresholds"]
        self.overrides  = self.config["rules_overrides"]
        self.descs      = self.config["decisions"]

    # ------------------------------------------------------------------
    # Public API — Normal flow (ALLOW / FLAG)
    # ------------------------------------------------------------------
    def decide(
        self,
        fraud_score: float,
        rules_flag: str = "ALLOW",
        anomaly_score: float = None,
        reconstruction_error: float = None,
    ) -> dict:
        """
        Score a transaction that has passed through all three ML models.

        Parameters
        ----------
        fraud_score : float
            XGBoost output probability (0.0 = safe, 1.0 = definite fraud).
        rules_flag : str
            "ALLOW" | "FLAG" | "BLOCK"
        anomaly_score : float, optional
            Isolation Forest decision_function score.
            Lower (more negative) = more anomalous.
        reconstruction_error : float, optional
            Autoencoder MSE reconstruction error.
            Higher = more anomalous.

        Returns
        -------
        dict — see class docstring.
        """
        rules_flag = rules_flag.upper().strip()
        if rules_flag not in VALID_RULES_FLAGS:
            rules_flag = "ALLOW"

        # Safety net: if BLOCK somehow arrives here, convert to early exit
        if rules_flag == "BLOCK":
            return self.early_exit_block(rules_triggered=["unknown — BLOCK passed to decide()"])

        fraud_score = float(max(0.0, min(1.0, fraud_score)))

        decision, triggered_by = self._apply_logic(
            fraud_score, rules_flag, anomaly_score, reconstruction_error
        )
        confidence = self._get_confidence(fraud_score)
        reason = self._build_reason(
            decision, fraud_score, rules_flag, triggered_by,
            anomaly_score, reconstruction_error
        )

        return {
            "decision":               decision,
            "fraud_score":            round(fraud_score, 6),
            "anomaly_score":          round(anomaly_score, 6) if anomaly_score is not None else None,
            "reconstruction_error":   round(reconstruction_error, 6) if reconstruction_error is not None else None,
            "rules_flag":             rules_flag,
            "is_early_exit":          False,
            "confidence":             confidence,
            "reason":                 reason,
            "triggered_by":           triggered_by,
            "timestamp":              _utc_now(),
        }

    # ------------------------------------------------------------------
    # Public API — Early exit (BLOCK)
    # ------------------------------------------------------------------
    def early_exit_block(self, rules_triggered: list = None) -> dict:
        """
        Produces an immediate DECLINE verdict without requiring an ML score.
        Called when the Rules Engine issues a BLOCK (transaction failed ≥
        block_threshold_pct of all evaluated rules).

        Parameters
        ----------
        rules_triggered : list of str
            Names of rules that were violated (for logging/explanation).

        Returns
        -------
        dict — same schema as decide(), fraud_score is None.
        """
        rules_triggered = rules_triggered or []
        triggered_str   = ", ".join(rules_triggered) if rules_triggered else "threshold exceeded"
        reason = (
            f"{self.descs.get('DECLINE_EARLY_EXIT', 'Declined early.')} "
            f"Rules violated: [{triggered_str}]."
        )

        return {
            "decision":               "DECLINE",
            "fraud_score":            None,
            "anomaly_score":          None,
            "reconstruction_error":   None,
            "rules_flag":             "BLOCK",
            "is_early_exit":          True,
            "confidence":             "HIGH",
            "reason":                 reason,
            "triggered_by":           "RULES_BLOCK_EARLY_EXIT",
            "timestamp":              _utc_now(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _apply_logic(self, fraud_score, rules_flag, anomaly_score=None, reconstruction_error=None):
        """Core decision tree. Returns (decision, triggered_by)."""

        cfg = self.config.get("anomaly_signals", {})
        ae_threshold  = cfg.get("ae_reconstruction_threshold", 0.31)
        if_threshold  = cfg.get("if_anomaly_threshold", 0.0)  # IF: below 0 = anomalous
        boost_enabled = cfg.get("boost_to_decline_when_both_flag", True)

        # Determine if IF and AE both signal anomaly
        if_flagged = (anomaly_score is not None) and (anomaly_score < if_threshold)
        ae_flagged = (reconstruction_error is not None) and (reconstruction_error > ae_threshold)
        both_flagged = if_flagged and ae_flagged

        # 1. Hard decline: XGBoost fraud_score above decline threshold
        if fraud_score >= self.thresh["decline_above"]:
            return "DECLINE", "ML_SCORE"

        # 2. Anomaly boost: both unsupervised models agree AND score is in challenge band
        if boost_enabled and both_flagged and fraud_score >= self.thresh["challenge_above"]:
            return "DECLINE", "ML_SCORE+IF_ANOMALY+AE_RECONSTRUCTION"

        # 3. Rules flagged → escalate to CHALLENGE
        if rules_flag == "FLAG" and self.overrides.get("FLAG_escalates_to_challenge", True):
            if fraud_score >= self.thresh["challenge_above"]:
                return "CHALLENGE", "ML_SCORE+RULES_FLAG"
            return "CHALLENGE", "RULES_FLAG"

        # 4. XGBoost alone in challenge band → CHALLENGE
        if fraud_score >= self.thresh["challenge_above"]:
            triggered = "ML_SCORE"
            if if_flagged:
                triggered += "+IF_ANOMALY"
            if ae_flagged:
                triggered += "+AE_RECONSTRUCTION"
            return "CHALLENGE", triggered

        # 5. Unsupervised signals flag but score is low → lightweight CHALLENGE
        if both_flagged:
            return "CHALLENGE", "IF_ANOMALY+AE_RECONSTRUCTION"

        # 6. All clear
        return "APPROVE", "ML_SCORE"

    def _get_confidence(self, fraud_score: float) -> str:
        bands   = self.config.get("confidence_bands", {})
        h_min   = bands.get("high_certainty_min", 0.85)
        h_max   = bands.get("high_certainty_max", 0.15)
        m_min   = bands.get("medium_certainty_min", 0.65)
        m_max   = bands.get("medium_certainty_max", 0.35)

        if fraud_score >= h_min or fraud_score <= h_max:
            return "HIGH"
        if fraud_score >= m_min or fraud_score <= m_max:
            return "MEDIUM"
        return "LOW"

    def _build_reason(self, decision, fraud_score, rules_flag, triggered_by,
                      anomaly_score=None, reconstruction_error=None) -> str:
        base    = self.descs.get(decision, decision)
        details = []
        if "ML_SCORE" in triggered_by:
            details.append(f"XGBoost fraud score = {fraud_score:.4f}")
        if "IF_ANOMALY" in triggered_by and anomaly_score is not None:
            details.append(f"Isolation Forest anomaly score = {anomaly_score:.4f}")
        if "AE_RECONSTRUCTION" in triggered_by and reconstruction_error is not None:
            details.append(f"Autoencoder reconstruction error = {reconstruction_error:.4f}")
        if "RULES_FLAG" in triggered_by:
            details.append("Rules Engine flagged this transaction")
        return f"{base} ({'; '.join(details)})" if details else base

    @staticmethod
    def _load_config(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[DecisionEngine] Config not found: {path}\n"
                "Ensure decision_config.json is in the Decision Layer folder."
            )
        with open(path, 'r') as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
_default_engine: DecisionEngine | None = None


def _get_engine() -> DecisionEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = DecisionEngine()
    return _default_engine


def make_decision(
    fraud_score: float,
    rules_flag: str = "ALLOW",
    anomaly_score: float = None,
    reconstruction_error: float = None,
) -> dict:
    """Normal flow — transaction reached the ML layer."""
    return _get_engine().decide(fraud_score, rules_flag, anomaly_score, reconstruction_error)


def make_early_exit_decision(rules_triggered: list = None) -> dict:
    """Early exit — Rules Engine issued a BLOCK; ML layer was skipped."""
    return _get_engine().early_exit_block(rules_triggered)
