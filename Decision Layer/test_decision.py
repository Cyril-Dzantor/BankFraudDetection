"""
test_decision.py — Validation Script for the Decision Layer
============================================================
Tests all decision paths including the new multi-signal (XGBoost + IF + AE)
logic and early-exit BLOCK scenarios.

Rules Engine state semantics:
  ALLOW  — all rules passed           → normal flow through ML → Decision Engine
  FLAG   — one rule failed            → normal flow through ML → Decision Engine
  BLOCK  — failed >= threshold % rules → EARLY EXIT, ML skipped → DECLINE immediately

Signal conventions:
  fraud_score          — XGBoost probability (0.0-1.0). Higher = more fraud.
  anomaly_score        — Isolation Forest decision_function. NEGATIVE = anomalous.
  reconstruction_error — Autoencoder MSE. HIGH (> 0.31) = anomalous.

Usage:
    python "Decision Layer\\test_decision.py"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_engine import DecisionEngine

engine = DecisionEngine()

# -----------------------------------------------------------------------------
# SECTION A — Normal flow: clean IF/AE signals (no anomaly flags)
# Format: (fraud_score, rules_flag, anomaly_score, reconstruction_error,
#           expected_decision, expected_is_early_exit, label)
# -----------------------------------------------------------------------------
NORMAL_FLOW_CASES = [
    # --- APPROVE: low score, clean signals ---
    (0.10, "ALLOW", 0.12,  0.05, "APPROVE",   False, "Low score, IF/AE normal → APPROVE"),
    (0.49, "ALLOW", 0.08,  0.10, "APPROVE",   False, "Score just below challenge threshold → APPROVE"),

    # --- CHALLENGE: XGBoost score in band ---
    (0.55, "ALLOW", 0.05,  0.08, "CHALLENGE", False, "Score in challenge band, clean signals → CHALLENGE"),
    (0.72, "ALLOW", 0.10,  0.15, "CHALLENGE", False, "Mid-high score, rules ALLOW, clean signals → CHALLENGE"),

    # --- CHALLENGE: Rules FLAG escalation ---
    (0.25, "FLAG",  0.09,  0.12, "CHALLENGE", False, "Low score but a rule failed → CHALLENGE (FLAG escalates)"),
    (0.62, "FLAG",  0.06,  0.09, "CHALLENGE", False, "Challenge score + rule failed → CHALLENGE"),

    # --- DECLINE: score-based ---
    (0.90, "ALLOW", 0.08,  0.10, "DECLINE",   False, "Score above decline threshold → DECLINE"),
    (0.99, "ALLOW", 0.10,  0.05, "DECLINE",   False, "Near-certain fraud → DECLINE"),
    (0.88, "FLAG",  0.07,  0.08, "DECLINE",   False, "High score + rule failed → DECLINE (score wins)"),
]

# -----------------------------------------------------------------------------
# SECTION B — Anomaly boost: IF + AE signals combined with fraud_score
# Both IF and AE flagging: anomaly_score < 0.0 AND reconstruction_error > 0.31
# -----------------------------------------------------------------------------
ANOMALY_BOOST_CASES = [
    # Score in challenge band + both IF & AE flag → escalate to DECLINE
    (0.65, "ALLOW", -0.08, 0.45, "DECLINE",   False, "Challenge score + both IF & AE flag → DECLINE (anomaly boost)"),
    (0.51, "ALLOW", -0.15, 0.60, "DECLINE",   False, "Just above challenge + both flag → DECLINE (anomaly boost)"),

    # Score below challenge band + both flag → cautious CHALLENGE
    (0.30, "ALLOW", -0.10, 0.40, "CHALLENGE", False, "Low score + both IF & AE flag → CHALLENGE (cautious catch)"),
    (0.10, "ALLOW", -0.20, 0.55, "CHALLENGE", False, "Very low score + both flag strongly → CHALLENGE"),

    # Only one unsupervised model flags → no boost, normal band logic applies
    (0.60, "ALLOW", -0.05, 0.10, "CHALLENGE", False, "Challenge score + only IF flags (AE clean) → CHALLENGE (no boost)"),
    (0.60, "ALLOW",  0.05, 0.45, "CHALLENGE", False, "Challenge score + only AE flags (IF clean) → CHALLENGE (no boost)"),

    # High score always declines regardless of anomaly signals
    (0.90, "ALLOW", -0.12, 0.50, "DECLINE",   False, "High score + both flag → DECLINE (score threshold wins)"),
]

# -----------------------------------------------------------------------------
# SECTION C — Early exit: early_exit_block(rules_triggered)
# -----------------------------------------------------------------------------
EARLY_EXIT_CASES = [
    (
        ["velocity_5m", "geo_mismatch", "new_device", "accounts_per_ip"],
        "DECLINE", True,
        "Failed 4 rules (>= threshold %) → EARLY EXIT DECLINE, ML not called"
    ),
    (
        ["amount_spike", "failed_logins"],
        "DECLINE", True,
        "Failed 2 rules at threshold → EARLY EXIT DECLINE"
    ),
    (
        [],
        "DECLINE", True,
        "BLOCK with no rule list → EARLY EXIT DECLINE"
    ),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def print_verdict(verdict: dict):
    score_str = f"{verdict['fraud_score']:.4f}" if verdict['fraud_score'] is not None else "N/A (early exit)"
    anom_str  = f"{verdict['anomaly_score']:.4f}" if verdict.get('anomaly_score') is not None else "N/A"
    recon_str = f"{verdict['reconstruction_error']:.4f}" if verdict.get('reconstruction_error') is not None else "N/A"
    print(f"    fraud_score          : {score_str}")
    print(f"    anomaly_score (IF)   : {anom_str}")
    print(f"    reconstruction (AE)  : {recon_str}")
    print(f"    rules_flag           : {verdict['rules_flag']}")
    print(f"    decision             : {verdict['decision']}")
    print(f"    is_early_exit        : {verdict['is_early_exit']}")
    print(f"    confidence           : {verdict['confidence']}")
    print(f"    triggered_by         : {verdict['triggered_by']}")
    print(f"    reason               : {verdict['reason']}")


def check(verdict: dict, expected_decision: str, expected_early: bool):
    decision_ok = verdict["decision"]     == expected_decision
    early_ok    = verdict["is_early_exit"] == expected_early
    ok_all      = decision_ok and early_ok
    if ok_all:
        return True, "PASS ✓"
    details = []
    if not decision_ok:
        details.append(f"decision: got '{verdict['decision']}', expected '{expected_decision}'")
    if not early_ok:
        details.append(f"is_early_exit: got {verdict['is_early_exit']}, expected {expected_early}")
    return False, f"FAIL ✗  ({'; '.join(details)})"


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_tests():
    passed = 0
    failed = 0
    total  = len(NORMAL_FLOW_CASES) + len(ANOMALY_BOOST_CASES) + len(EARLY_EXIT_CASES)

    print("\nBank Fraud Detection — Decision Layer Validation")
    print("=" * 70)

    # --- Section A: Normal flow ---
    print("\n[SECTION A] Normal flow — XGBoost score + clean IF/AE signals")
    print("-" * 70)
    for fraud_score, rules_flag, anom, recon, exp_decision, exp_early, label in NORMAL_FLOW_CASES:
        verdict = engine.decide(
            fraud_score=fraud_score, rules_flag=rules_flag,
            anomaly_score=anom, reconstruction_error=recon
        )
        ok, status = check(verdict, exp_decision, exp_early)
        print(f"\n  - {label}")
        print_verdict(verdict)
        print(f"    result               : {status}")
        passed += int(ok)
        failed += int(not ok)

    # --- Section B: Anomaly boost ---
    print(f"\n[SECTION B] Anomaly boost — IF + AE signals combined with fraud_score")
    print("-" * 70)
    for fraud_score, rules_flag, anom, recon, exp_decision, exp_early, label in ANOMALY_BOOST_CASES:
        verdict = engine.decide(
            fraud_score=fraud_score, rules_flag=rules_flag,
            anomaly_score=anom, reconstruction_error=recon
        )
        ok, status = check(verdict, exp_decision, exp_early)
        print(f"\n  - {label}")
        print_verdict(verdict)
        print(f"    result               : {status}")
        passed += int(ok)
        failed += int(not ok)

    # --- Section C: Early exit ---
    print(f"\n[SECTION C] Early-exit tests (BLOCK — ML layer skipped)")
    print("-" * 70)
    for rules_triggered, exp_decision, exp_early, label in EARLY_EXIT_CASES:
        verdict = engine.early_exit_block(rules_triggered=rules_triggered)
        ok, status = check(verdict, exp_decision, exp_early)
        print(f"\n  - {label}")
        print_verdict(verdict)
        print(f"    result               : {status}")
        passed += int(ok)
        failed += int(not ok)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed} passed, {failed} failed out of {total} tests")
    print(f"{'='*70}")

    if failed == 0:
        print("\n  All tests passed. Decision Layer is working correctly.\n")
    else:
        print(f"\n  {failed} test(s) failed. Review logic above.\n")

    return failed == 0


def main():
    success = run_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
