"""
test_rules.py — Validation Script for the Rules Engine (Layer 1)
================================================================
Tests all rule categories and escalation logic:
  - Individual rule fire / no-fire
  - BLOCK from direct BLOCK-severity rule
  - BLOCK from failure percentage threshold (>= 40% of rules failed)
  - FLAG from FLAG-severity rules only
  - ALLOW when all rules pass
  - Edge cases and boundary conditions

Usage:
    python "Rules Engine\\test_rules.py"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rules_engine import RulesEngine

engine = RulesEngine()

# ---------------------------------------------------------------------------
# Base clean transaction — all values well within safe limits
# ---------------------------------------------------------------------------
BASE_TX = {
    "tx_count_last_5m":          1,
    "tx_count_last_1h":          2,
    "tx_frequency_ratio":        0.9,
    "amount":                    95.00,
    "avg_tx_amount_7d":          110.00,
    "amount_to_avg_ratio":       0.86,
    "new_device_flag":           0,
    "device_seen_before":        1,
    "country_mismatch_flag":     0,
    "accounts_per_device":       1,
    "accounts_per_ip_24h":       1,
    "failed_login_count_last_1h": 0,
    "failed_logins_then_success": 0,
    "channel":                   "mobile",
    "auth_method":               "biometric",
    "geo_country":               "GH",
    "currency":                  "GHS",
}


def tx(**overrides) -> dict:
    """Return a copy of BASE_TX with field overrides applied."""
    t = BASE_TX.copy()
    t.update(overrides)
    return t


# ---------------------------------------------------------------------------
# Test cases
# Format: (label, transaction_dict, expected_flag, expected_rule_in_triggered)
# ---------------------------------------------------------------------------

# --- ALLOW tests ---
ALLOW_CASES = [
    ("Clean transaction — all rules pass",
     tx(),
     "ALLOW", None),

    ("Low-level activity, known device, matching geo",
     tx(tx_count_last_5m=1, tx_count_last_1h=3, amount=50),
     "ALLOW", None),
]

# --- FLAG tests ---
FLAG_CASES = [
    ("High 1-hour velocity (11 tx)",
     tx(tx_count_last_1h=11),
     "FLAG", "velocity_high_1h"),

    ("Frequency spike (ratio = 4.0)",
     tx(tx_frequency_ratio=4.0),
     "FLAG", "velocity_frequency_spike"),

    ("Large amount (>$10,000)",
     tx(amount=15_000),
     "FLAG", "amount_large"),

    ("Amount ratio spike (6× avg)",
     tx(amount_to_avg_ratio=6.0),
     "FLAG", "amount_ratio_spike"),

    ("Country mismatch",
     tx(country_mismatch_flag=1),
     "FLAG", "geo_country_mismatch"),

    ("New device (no failed logins)",
     tx(new_device_flag=1, failed_login_count_last_1h=0),
     "FLAG", "new_device_flag"),

    ("Device shared by 4 accounts (multi-account risk)",
     tx(accounts_per_device=4),
     "FLAG", "device_multi_account"),

    ("IP shared by 6 accounts in 24h",
     tx(accounts_per_ip_24h=6),
     "FLAG", "ip_multi_account"),

    ("3+ brute-force login attempts",
     tx(failed_login_count_last_1h=4),
     "FLAG", "brute_force_login"),

    ("Failed logins then success",
     tx(failed_logins_then_success=1),
     "FLAG", "failed_logins_then_success"),

    ("PIN auth on new device",
     tx(auth_method="pin", new_device_flag=1),
     "FLAG", "weak_auth_new_device"),
]

# --- BLOCK tests (single hard-stop rule) ---
BLOCK_DIRECT_CASES = [
    ("Velocity burst: >5 tx in 5 minutes",
     tx(tx_count_last_5m=6),
     "BLOCK", "velocity_burst_5m"),

    ("Extreme amount ratio (21× avg)",
     tx(amount_to_avg_ratio=21.0),
     "BLOCK", "amount_extreme_ratio"),

    ("New device + 3 failed logins (ATO signal)",
     tx(new_device_flag=1, failed_login_count_last_1h=3),
     "BLOCK", "new_device_with_failed_logins"),

    ("Device farming: 11+ accounts on device",
     tx(accounts_per_device=11),
     "BLOCK", "device_farming"),

    ("Botnet IP: 21 accounts from same IP in 24h",
     tx(accounts_per_ip_24h=21),
     "BLOCK", "ip_botnet"),
]

# --- BLOCK tests (escalation via failure percentage) ---
# With 16 rules total, 40% threshold = 6.4 → need 7 FLAG rules to fire without
# any BLOCK rule to trigger an escalation BLOCK.
BLOCK_THRESHOLD_CASES = [
    ("Multiple FLAG rules accumulate to >= 40% threshold → escalated BLOCK",
     tx(
         tx_count_last_1h=11,          # velocity_high_1h
         tx_frequency_ratio=4.0,        # velocity_frequency_spike
         amount=15_000,                 # amount_large
         amount_to_avg_ratio=6.5,       # amount_ratio_spike
         country_mismatch_flag=1,       # geo_country_mismatch
         accounts_per_device=4,         # device_multi_account
         accounts_per_ip_24h=6,         # ip_multi_account
     ),
     "BLOCK", None),
]

# --- Boundary / edge cases ---
BOUNDARY_CASES = [
    ("tx_count_last_5m exactly at limit (=5) — should NOT trigger",
     tx(tx_count_last_5m=5),
     "ALLOW", None),

    ("tx_count_last_5m just over limit (=6) — should BLOCK",
     tx(tx_count_last_5m=6),
     "BLOCK", "velocity_burst_5m"),

    ("amount_to_avg_ratio exactly 5.0 — should NOT trigger amount_ratio_spike",
     tx(amount_to_avg_ratio=5.0),
     "ALLOW", None),

    ("amount_to_avg_ratio = 5.01 — should FLAG amount_ratio_spike",
     tx(amount_to_avg_ratio=5.01),
     "FLAG", "amount_ratio_spike"),

    ("Missing keys in transaction dict — engine must not crash",
     {"amount": 50},   # minimal dict
     "ALLOW", None),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check(result: dict, expected_flag: str, expected_rule: str | None):
    flag_ok = result["rules_flag"] == expected_flag
    rule_ok = (
        expected_rule is None
        or expected_rule in result["triggered_rules"]
    )
    ok = flag_ok and rule_ok
    if ok:
        return True, "PASS ✓"
    details = []
    if not flag_ok:
        details.append(f"flag: got '{result['rules_flag']}', expected '{expected_flag}'")
    if not rule_ok:
        details.append(f"rule '{expected_rule}' not in triggered_rules {result['triggered_rules']}")
    return False, f"FAIL ✗  ({'; '.join(details)})"


def print_result(result: dict):
    print(f"    rules_flag      : {result['rules_flag']}")
    print(f"    triggered_rules : {result['triggered_rules']}")
    print(f"    block_rules_hit : {result['block_rules_hit']}")
    print(f"    rules_failed    : {result['rules_failed']}/{result['total_rules']}"
          f"  ({result['failure_pct']*100:.1f}%)")


def run_section(title: str, cases: list) -> tuple[int, int]:
    passed = failed = 0
    print(f"\n[{title}]")
    print("-" * 70)
    for label, transaction, expected_flag, expected_rule in cases:
        result = engine.evaluate(transaction)
        ok, status = check(result, expected_flag, expected_rule)
        print(f"\n  - {label}")
        print_result(result)
        print(f"    result          : {status}")
        passed += int(ok)
        failed += int(not ok)
    return passed, failed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_tests():
    print("\nBank Fraud Detection — Rules Engine Validation")
    print("=" * 70)

    total_passed = total_failed = 0

    sections = [
        ("SECTION A — ALLOW: all rules pass",             ALLOW_CASES),
        ("SECTION B — FLAG: individual flag rules",        FLAG_CASES),
        ("SECTION C — BLOCK: direct BLOCK-severity rules", BLOCK_DIRECT_CASES),
        ("SECTION D — BLOCK: failure % threshold",         BLOCK_THRESHOLD_CASES),
        ("SECTION E — Boundary & edge cases",              BOUNDARY_CASES),
    ]

    for title, cases in sections:
        p, f = run_section(title, cases)
        total_passed += p
        total_failed += f

    total = total_passed + total_failed
    print(f"\n{'='*70}")
    print(f"  RESULTS: {total_passed} passed, {total_failed} failed out of {total} tests")
    print(f"{'='*70}")

    if total_failed == 0:
        print("\n  All tests passed. Rules Engine is working correctly.\n")
    else:
        print(f"\n  {total_failed} test(s) failed. Review logic above.\n")

    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
