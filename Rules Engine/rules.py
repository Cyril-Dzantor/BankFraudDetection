"""
rules.py — Bank Fraud Detection: Rules Engine Rule Definitions
==============================================================
Defines every individual rule as a dataclass. Each rule:
  - Has a unique name and a severity level (FLAG or BLOCK)
  - Exposes an evaluate(transaction) method that returns True if the rule fires

Rule severity semantics
-----------------------
  FLAG   — one signal of suspicion; contributes to the failure percentage count
  BLOCK  — a hard stop; immediately escalates to BLOCK regardless of percentage

Failure threshold (from decision_config.json)
---------------------------------------------
  If the number of FAILED rules / total rules evaluated >= block_threshold_pct (0.40)
  AND no single rule alone issued a direct BLOCK,
  the Rules Engine escalates the result to BLOCK automatically.
"""

from dataclasses import dataclass, field
from typing import Callable, List


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------
FLAG  = "FLAG"
BLOCK = "BLOCK"


# ---------------------------------------------------------------------------
# Rule base
# ---------------------------------------------------------------------------
@dataclass
class Rule:
    """
    A single evaluatable rule.

    Attributes
    ----------
    name     : str           — unique rule identifier (also shown in triggered_rules)
    severity : str           — "FLAG" or "BLOCK"
    condition: Callable      — function(transaction: dict) -> bool
                               Returns True if the rule is triggered (fires).
    description: str         — human-readable explanation for logs / audit
    """
    name:        str
    severity:    str
    condition:   Callable[[dict], bool]
    description: str = ""


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
# Each rule receives the full transaction dict.
# Field names must match TransactionRequest exactly (see models.py).
#
# tx_count_last_5m         : float  — transactions in last 5 minutes
# tx_count_last_1h         : float  — transactions in last 1 hour
# tx_frequency_ratio       : float  — current rate vs historical average
# amount                   : float  — transaction amount
# avg_tx_amount_7d         : float  — 7-day average amount
# amount_to_avg_ratio      : float  — amount / avg_tx_amount_7d
# new_device_flag          : int    — 1 if new device
# device_seen_before       : int    — 1 if known device
# country_mismatch_flag    : int    — 1 if geo country ≠ typical country
# accounts_per_device      : float  — accounts linked to this device
# accounts_per_ip_24h      : float  — accounts seen from this IP in 24h
# failed_login_count_last_1h: float — failed logins in last hour
# failed_logins_then_success: int   — 1 if failed logins preceded this tx
# channel                  : str    — "mobile" | "web" | "atm" | "pos"
# auth_method              : str    — "pin" | "biometric" | "otp" | "password"
# geo_country              : str    — e.g. "GH", "NG"
# currency                 : str    — e.g. "GHS", "USD"
# ---------------------------------------------------------------------------

ALL_RULES: List[Rule] = [

    # -----------------------------------------------------------------------
    # VELOCITY RULES
    # -----------------------------------------------------------------------
    Rule(
        name        = "velocity_burst_5m",
        severity    = BLOCK,
        condition   = lambda tx: float(tx.get("tx_count_last_5m", 0)) > 5,
        description = "More than 5 transactions in the last 5 minutes — rapid-fire / bot attack",
    ),

    Rule(
        name        = "velocity_high_1h",
        severity    = FLAG,
        condition   = lambda tx: float(tx.get("tx_count_last_1h", 0)) > 10,
        description = "More than 10 transactions in the last 1 hour — elevated velocity",
    ),

    Rule(
        name        = "velocity_frequency_spike",
        severity    = FLAG,
        condition   = lambda tx: float(tx.get("tx_frequency_ratio", 0)) > 3.0,
        description = "Current transaction frequency is 3x above the user's historical baseline",
    ),

    # -----------------------------------------------------------------------
    # AMOUNT / THRESHOLD RULES
    # -----------------------------------------------------------------------
    Rule(
        name        = "amount_large",
        severity    = FLAG,
        condition   = lambda tx: float(tx.get("amount", 0)) > 10_000,
        description = "Transaction amount exceeds $10,000",
    ),

    Rule(
        name        = "amount_ratio_spike",
        severity    = FLAG,
        condition   = lambda tx: float(tx.get("amount_to_avg_ratio", 0)) > 5.0,
        description = "Amount is more than 5× the user's 7-day average — possible account draining",
    ),

    Rule(
        name        = "amount_extreme_ratio",
        severity    = BLOCK,
        condition   = lambda tx: float(tx.get("amount_to_avg_ratio", 0)) > 20.0,
        description = "Amount is more than 20× the user's 7-day average — extreme outlier",
    ),

    # -----------------------------------------------------------------------
    # GEO / COUNTRY RULES
    # -----------------------------------------------------------------------
    Rule(
        name        = "geo_country_mismatch",
        severity    = FLAG,
        condition   = lambda tx: int(tx.get("country_mismatch_flag", 0)) == 1,
        description = "Transaction country does not match the user's typical country",
    ),

    # -----------------------------------------------------------------------
    # DEVICE RULES
    # -----------------------------------------------------------------------
    Rule(
        name        = "new_device_with_failed_logins",
        severity    = BLOCK,
        condition   = lambda tx: (
            int(tx.get("new_device_flag", 0)) == 1
            and float(tx.get("failed_login_count_last_1h", 0)) > 2
        ),
        description = "New device used after 3+ failed logins — likely ATO via credential stuffing",
    ),

    Rule(
        name        = "new_device_flag",
        severity    = FLAG,
        condition   = lambda tx: int(tx.get("new_device_flag", 0)) == 1,
        description = "Transaction initiated from a device not previously seen for this account",
    ),

    Rule(
        name        = "device_farming",
        severity    = BLOCK,
        condition   = lambda tx: float(tx.get("accounts_per_device", 0)) > 10,
        description = "More than 10 accounts linked to this device — device farming / mule network",
    ),

    Rule(
        name        = "device_multi_account",
        severity    = FLAG,
        condition   = lambda tx: float(tx.get("accounts_per_device", 0)) > 3,
        description = "3+ accounts linked to this device — shared device risk",
    ),

    # -----------------------------------------------------------------------
    # IP / NETWORK RULES
    # -----------------------------------------------------------------------
    Rule(
        name        = "ip_multi_account",
        severity    = FLAG,
        condition   = lambda tx: float(tx.get("accounts_per_ip_24h", 0)) > 5,
        description = "5+ accounts transacting from the same IP in 24 hours — possible proxy or botnet",
    ),

    Rule(
        name        = "ip_botnet",
        severity    = BLOCK,
        condition   = lambda tx: float(tx.get("accounts_per_ip_24h", 0)) > 20,
        description = "20+ accounts from same IP in 24h — strong botnet / proxy indicator",
    ),

    # -----------------------------------------------------------------------
    # AUTHENTICATION RULES
    # -----------------------------------------------------------------------
    Rule(
        name        = "brute_force_login",
        severity    = FLAG,
        condition   = lambda tx: float(tx.get("failed_login_count_last_1h", 0)) > 3,
        description = "More than 3 failed login attempts in the last hour",
    ),

    Rule(
        name        = "failed_logins_then_success",
        severity    = FLAG,
        condition   = lambda tx: int(tx.get("failed_logins_then_success", 0)) == 1,
        description = "Failed login attempts were immediately followed by a successful login",
    ),

    Rule(
        name        = "weak_auth_new_device",
        severity    = FLAG,
        condition   = lambda tx: (
            tx.get("auth_method", "") == "pin"
            and int(tx.get("new_device_flag", 0)) == 1
        ),
        description = "PIN authentication used on a new, previously unseen device",
    ),
]
