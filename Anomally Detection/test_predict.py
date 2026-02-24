"""
test_predict.py — Validation Script for the ML Inference Pipeline
==================================================================
Runs two hand-crafted transactions through the full scoring pipeline
and asserts that the output fraud scores are within expected ranges.

Usage:
    python "Anomally Detection\test_predict.py"
    (from the project root with venv activated)
"""

import sys
import os

# Allow imports from the Anomally Detection folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import FraudScorer

# ---------------------------------------------------------------------------
# Test Transactions
# ---------------------------------------------------------------------------

# High-risk fraudulent transaction:
#   - Very high amount relative to average
#   - Many recent transactions (velocity)
#   - Multiple failed logins before success
#   - New device + accounts shared across IP
#   - Geo country mismatch
FRAUD_TRANSACTION = {
    "tx_count_last_5m":            15,      # Very high velocity
    "tx_count_last_1h":            40,
    "tx_frequency_ratio":          8.5,     # 8.5x normal frequency
    "amount":                      9800.00,
    "avg_tx_amount_7d":            120.00,
    "amount_to_avg_ratio":         81.67,   # 81x the average amount
    "new_device_flag":             1,
    "device_seen_before":          0,
    "country_mismatch_flag":       1,
    "accounts_per_device":         7,
    "accounts_per_ip_24h":         12,
    "failed_login_count_last_1h":  8,
    "failed_logins_then_success":  1,
    "channel":                     "web",
    "auth_method":                 "password",
    "geo_country":                 "Unknown",  # Unrecognised country
    "currency":                    "USD",
}

# Low-risk legitimate transaction:
#   - Normal amount, consistent with history
#   - Low velocity
#   - Known device and country, no failed logins
LEGIT_TRANSACTION = {
    "tx_count_last_5m":            1,
    "tx_count_last_1h":            2,
    "tx_frequency_ratio":          0.9,
    "amount":                      95.00,
    "avg_tx_amount_7d":            110.00,
    "amount_to_avg_ratio":         0.86,
    "new_device_flag":             0,
    "device_seen_before":          1,
    "country_mismatch_flag":       0,
    "accounts_per_device":         1,
    "accounts_per_ip_24h":         1,
    "failed_login_count_last_1h":  0,
    "failed_logins_then_success":  0,
    "channel":                     "mobile",
    "auth_method":                 "biometric",
    "geo_country":                 "GH",
    "currency":                    "GHS",
}

# ---------------------------------------------------------------------------
# Thresholds for PASS/FAIL
# ---------------------------------------------------------------------------
FRAUD_SCORE_MIN  = 0.70   # Fraud case must score above this
LEGIT_SCORE_MAX  = 0.30   # Legit case must score below this

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def print_result(label: str, tx: dict, result: dict):
    """Pretty-prints a single scoring result."""
    print(f"\n{'='*60}")
    print(f"  Test Case : {label}")
    print(f"{'='*60}")
    print(f"  Fraud Score   : {result['fraud_score']:.4f}  (0=safe, 1=fraud)")
    print(f"  Is Fraud      : {result['is_fraud']}")
    print(f"  Confidence    : {result['confidence']}")
    print(f"  IF Score      : {result['if_score']:.6f}  (lower = more anomalous)")
    print(f"  AE Error      : {result['ae_error']:.6f}  (threshold = {result['ae_threshold']:.6f})")
    print(f"  AE Anomaly    : {result['ae_anomaly']}")
    print()


def run_tests(scorer: FraudScorer):
    results = {}

    # --- Test 1: Fraudulent Transaction ---
    print("\n[TEST 1] Scoring fraudulent transaction...")
    result_fraud = scorer.score(FRAUD_TRANSACTION)
    print_result("FRAUD TRANSACTION", FRAUD_TRANSACTION, result_fraud)

    passed_fraud = result_fraud['fraud_score'] >= FRAUD_SCORE_MIN
    label_fraud = "[PASS]" if passed_fraud else "[FAIL]"
    print(f"  {label_fraud}  fraud_score={result_fraud['fraud_score']:.4f} "
          f"(expected >= {FRAUD_SCORE_MIN})")
    results['fraud'] = passed_fraud

    # --- Test 2: Legitimate Transaction ---
    print("\n[TEST 2] Scoring legitimate transaction...")
    result_legit = scorer.score(LEGIT_TRANSACTION)
    print_result("LEGIT TRANSACTION", LEGIT_TRANSACTION, result_legit)

    passed_legit = result_legit['fraud_score'] <= LEGIT_SCORE_MAX
    label_legit = "[PASS]" if passed_legit else "[FAIL]"
    print(f"  {label_legit}  fraud_score={result_legit['fraud_score']:.4f} "
          f"(expected <= {LEGIT_SCORE_MAX})")
    results['legit'] = passed_legit

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name.upper():<10} : {status}")
    print()

    if all_passed:
        print("  ✅ All tests passed. Inference pipeline is working correctly.")
    else:
        print("  ❌ One or more tests failed. Review the fraud scores above.")
        print("     NOTE: If scores are slightly outside thresholds, consider")
        print("     adjusting the FRAUD_SCORE_MIN / LEGIT_SCORE_MAX constants.")

    return all_passed


def main():
    print("\nBank Fraud Detection — ML Inference Pipeline Validation")
    print("="*60)

    try:
        scorer = FraudScorer()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    success = run_tests(scorer)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
