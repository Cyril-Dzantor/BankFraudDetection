# ================================
# BANK-GRADE SYNTHETIC FRAUD PIPELINE
# ================================

import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

# ----------------
# CONFIG
# ----------------
NUM_CUSTOMERS = 500
MAX_TXNS_PER_ACCOUNT = 120
START_DATE = datetime(2024, 1, 1)

HOME_LAT, HOME_LON = 5.6037, -0.1870  # Accra

BANKS = ["GCB", "ECOBANK", "ABSA", "MOMO_MTN"]
CHANNELS = ["MOB", "WEB", "ATM", "USSD"]
TXN_CODES = ["TRF", "CW", "PYT"]

SCENARIOS = {
    "NORMAL": 0.85,
    "ATO": 0.05,
    "MULE": 0.04,
    "BOT": 0.03,
    "VISHING": 0.03
}

# ----------------
# UTILS
# ----------------
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * (2 * asin(sqrt(a)))

def daily_limit(kyc):
    return {1:1000, 2:10000, 3:50000}[kyc]

def sample_scenario():
    return random.choices(list(SCENARIOS.keys()), list(SCENARIOS.values()))[0]

# ----------------
# SHARED POOLS (GNN LINKS)
# ----------------
mobile_pool = [f"MOB-{uuid.uuid4().hex}" for _ in range(200)]
device_pool = [f"DEV-{uuid.uuid4().hex}" for _ in range(300)]
bot_ips = [f"192.168.0.{i}" for i in range(1, 6)]

# ----------------
# CUSTOMER + ACCOUNT GENERATION
# ----------------
customers, accounts = [], []

for _ in range(NUM_CUSTOMERS):
    cif = f"CIF-{uuid.uuid4().hex[:10]}"
    risk = random.choices(["LOW", "MEDIUM", "HIGH"], [0.6, 0.25, 0.15])[0]
    kyc = random.choices([1, 2, 3], [0.4, 0.35, 0.25])[0]

    customers.append((cif, risk, kyc))

    for _ in range(random.randint(1, 4)):
        accounts.append({
            "primary_cif_id": cif,
            "account_number": f"ACC-{uuid.uuid4().hex[:12]}",
            "account_status": random.choices(["ACTIVE", "DORMANT"], [0.85, 0.15])[0]
        })

cust_map = {c[0]: {"risk": c[1], "kyc": c[2]} for c in customers}

# ----------------
# TRANSACTION PIPELINE
# ----------------
rows = []

for acc in accounts:
    balance = random.uniform(500, 30000)

    for _ in range(random.randint(20, MAX_TXNS_PER_ACCOUNT)):
        scenario = sample_scenario()
        cif = acc["primary_cif_id"]
        kyc = cust_map[cif]["kyc"]

        txn_date = START_DATE + timedelta(days=random.randint(0, 180))
        limit = daily_limit(kyc)

        amount = min(
            np.random.exponential(300),
            limit * random.uniform(0.7, 0.99 if scenario != "NORMAL" else 0.6)
        )
        amount = round(amount, 2)
        balance = round(max(balance - amount, 0), 2)

        # -------- Identity / Network --------
        sim_swap = 1 if scenario == "ATO" and random.random() < 0.9 else 0
        mobile = random.choice(mobile_pool[:10]) if scenario == "MULE" else random.choice(mobile_pool)
        device = random.choice(device_pool[:5]) if scenario in ["MULE", "BOT"] else random.choice(device_pool)
        ip = random.choice(bot_ips) if scenario == "BOT" else f"102.{random.randint(10,99)}.{random.randint(1,255)}.{random.randint(1,255)}"

        # -------- Geo --------
        if scenario == "ATO":
            lat = HOME_LAT + random.uniform(5, 15)
            lon = HOME_LON + random.uniform(5, 15)
        else:
            lat = HOME_LAT + random.uniform(-0.05, 0.05)
            lon = HOME_LON + random.uniform(-0.05, 0.05)

        dist = round(haversine(HOME_LAT, HOME_LON, lat, lon), 2)

        # -------- Behavior --------
        typing = random.uniform(5, 20) if scenario == "BOT" else \
                 random.uniform(400, 800) if scenario == "VISHING" else \
                 random.uniform(80, 250)

        jitter = random.uniform(0.7, 1.2) if scenario == "VISHING" else random.uniform(0.1, 0.4)
        session = random.randint(1, 5) if scenario == "BOT" else random.randint(30, 600)

        # -------- Auth --------
        auth = "PIN" if scenario == "VISHING" else random.choice(["FACE_ID", "FINGERPRINT"])
        failed = random.randint(3, 10) if scenario in ["ATO", "BOT"] else random.randint(0, 1)

        # -------- Network / App --------
        network = "VPN" if scenario in ["ATO", "BOT"] else random.choice(["WIFI", "4G"])
        app_version = "MODIFIED" if scenario in ["ATO", "BOT"] else random.choice(["LATEST", "OUTDATED"])
        battery = 100 if scenario == "BOT" else random.randint(15, 95)

        rows.append({
            # Core
            "txn_reference": f"TXN-{uuid.uuid4().hex}",
            "primary_cif_id": cif,
            "account_number": acc["account_number"],
            "beneficiary_bank": random.choice(BANKS),
            "amount_ghs": amount,
            "ledger_balance_after": balance,
            "daily_limit_utilized_pct": round(amount / limit, 2),
            "txn_code": random.choice(TXN_CODES),
            "channel_id": random.choice(CHANNELS),
            "value_date": txn_date,
            "account_status": acc["account_status"],
            "risk_rating": cust_map[cif]["risk"],
            "kyc_level": kyc,

            # SIM / Device / Network
            "sim_swap_flag": sim_swap,
            "linked_mobile_hash": mobile,
            "ip_address": ip,
            "geo_location_lat": lat,
            "geo_location_long": lon,
            "distance_from_home_km": dist,
            "network_type": network,
            "device_imei_hash": device,
            "device_battery_pct": battery,
            "app_version": app_version,

            # Behavior
            "mouse_movement_jitter": round(jitter, 2),
            "typing_cadence_ms": round(typing, 2),
            "session_duration_sec": session,
            "auth_method": auth,
            "failed_auth_count": failed,

            # Labels
            "fraud_flag": 0 if scenario == "NORMAL" else 1,
            "fraud_scenario": scenario
        })

# ----------------
# FINAL DATASET
# ----------------
df = pd.DataFrame(rows).sort_values(["account_number", "value_date"])
print(df.shape)
print(df.head())
