import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -----------------------
# CONFIGURATION
# -----------------------
# Simulation Size
N_TRANSACTIONS = 150_000
FRAUD_RATE = 0.04
RANDOM_SEED = 42

# Entities
N_ACCOUNTS = 12_000
N_DEVICES = 15_000
N_IPS = 20_000
N_MERCHANTS = 5_000

# Distributions (Log-Normal parameters for Amount)
FRAUD_AMOUNT_MEAN = 5.8
FRAUD_AMOUNT_SIGMA = 0.7
LEGIT_AMOUNT_MEAN = 4.2
LEGIT_AMOUNT_SIGMA = 0.5

# Probabilities (P(Feature | Fraud))
P_MISMATCH_GIVEN_FRAUD = 0.65
P_NEW_DEVICE_GIVEN_FRAUD = 0.80
P_HIGH_VELOCITY_GIVEN_FRAUD = 0.70

# Noise & Dirtiness
NOISE_FLIP_RATE = 0.08
MISSING_DATA_RATE = 0.015

np.random.seed(RANDOM_SEED)

# Categorical Vocabularies
countries = ["GH", "NG", "KE", "UK", "US", "DE"]
channels = ["card", "web", "mobile", "api"]
auth_methods = ["PIN", "OTP", "BIOMETRIC"]
currencies = ["GHS", "USD", "GBP", "EUR"]

# -----------------------
# 1. BASE ENTITY GENERATION
# -----------------------
print("Generating base entities...")
df = pd.DataFrame({
    "transaction_id": np.arange(N_TRANSACTIONS),
    "account_id": np.random.randint(1, N_ACCOUNTS, N_TRANSACTIONS),
    "device_id_hash": np.random.randint(1, N_DEVICES, N_TRANSACTIONS),
    "ip_address_hash": np.random.randint(1, N_IPS, N_TRANSACTIONS),
    "merchant_id": np.random.randint(1, N_MERCHANTS, N_TRANSACTIONS),
})

# Time Generation (One month window)
start_date = datetime(2026, 1, 1)
df["timestamp"] = [
    start_date + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600))
    for _ in range(N_TRANSACTIONS)
]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# -----------------------
# 2. FRAUD LABEL GENERATION (Inverse Transform Sampling)
# -----------------------
# We generate the label FIRST, so we can condition features on it.
print("Generating fraud labels...")
df["is_fraud"] = (np.random.rand(N_TRANSACTIONS) < FRAUD_RATE).astype(int)
fraud_mask = df["is_fraud"] == 1
n_fraud = fraud_mask.sum()

# -----------------------
# 3. FEATURE GENERATION (Conditioned on Label)
# -----------------------
print("Generating features...")

# --- A. Transaction Amount ---
# Fraudsters target higher values but with high variance
df["amount"] = np.where(
    fraud_mask,
    np.random.lognormal(FRAUD_AMOUNT_MEAN, FRAUD_AMOUNT_SIGMA, N_TRANSACTIONS),
    np.random.lognormal(LEGIT_AMOUNT_MEAN, LEGIT_AMOUNT_SIGMA, N_TRANSACTIONS)
).round(2)

# --- B. Basic Categoricals ---
df["channel"] = np.random.choice(channels, N_TRANSACTIONS)
df["currency"] = np.random.choice(currencies, N_TRANSACTIONS)
df["geo_country"] = np.random.choice(countries, N_TRANSACTIONS)
df["auth_method"] = np.random.choice(auth_methods, N_TRANSACTIONS)

# --- C. Geo-Location Consistency ---
# Calculate "Typical" Country for each user (Mode)
df["typical_country"] = df.groupby("account_id")["geo_country"].transform(lambda x: x.mode().iloc[0])

# Generate Mismatch Flag
# If Legit, mismatch is rare (travel). If Fraud, mismatch is common (proxy).
base_mismatch = (df["geo_country"] != df["typical_country"]).astype(int)
fraud_mismatch = (np.random.rand(n_fraud) < P_MISMATCH_GIVEN_FRAUD).astype(int)
df["country_mismatch_flag"] = base_mismatch
df.loc[fraud_mask, "country_mismatch_flag"] = fraud_mismatch

# --- D. Device Recognition ---
# Logic: Legit users mostly reuse devices. Fraudsters use new devices.
df["device_seen_before"] = df.groupby(["account_id", "device_id_hash"]).cumcount().gt(0).astype(int)
base_new_device = (df["device_seen_before"] == 0).astype(int)
fraud_new_device = (np.random.rand(n_fraud) < P_NEW_DEVICE_GIVEN_FRAUD).astype(int)

df["new_device_flag"] = base_new_device
df.loc[fraud_mask, "new_device_flag"] = fraud_new_device

# --- E. Velocity (Rolling Windows) ---
print("Calculating velocity features...")
# IMPORTANT: Sort by account and time to get correct user history for rolling window
df = df.sort_values(['account_id', 'timestamp'])

# FIXED: Using lowercase 'h', 'min', 'd' for Pandas 2.2+ compatibility
df['tx_count_last_1h'] = df.groupby('account_id').rolling('1h', on='timestamp')['transaction_id'].count().values
df['tx_count_last_5m'] = df.groupby('account_id').rolling('5min', on='timestamp')['transaction_id'].count().values
df['avg_tx_amount_7d'] = df.groupby('account_id').rolling('7d', on='timestamp')['amount'].mean().values

# Restore Time Order
df = df.sort_values('timestamp').reset_index(drop=True)

# Velocity Boost for Fraud (Simulate Attacks/Burst behavior)
boost_1h = np.random.randint(5, 20, n_fraud)
boost_5m = np.random.randint(2, 8, n_fraud)
df.loc[fraud_mask, 'tx_count_last_1h'] += boost_1h
df.loc[fraud_mask, 'tx_count_last_5m'] += boost_5m

# --- F. Auth / Login Behavior ---
# Fraudsters fail logins more often (Brute force / Credential stuffing)
df["failed_login_count_last_1h"] = np.random.poisson(0.4, N_TRANSACTIONS) # Base rate
df.loc[fraud_mask, "failed_login_count_last_1h"] = np.random.poisson(4.0, n_fraud)
df["failed_logins_then_success"] = (df["failed_login_count_last_1h"] >= 3).astype(int)

# --- G. Network / Identity Signals ---
# Re-sort for IP rolling window
df = df.sort_values(['ip_address_hash', 'timestamp'])

# FIXED: Using lowercase '24h'
df['accounts_per_ip_24h'] = df.groupby('ip_address_hash').rolling('24h', on='timestamp')['account_id'].count().values

# Restore Time Order again
df = df.sort_values('timestamp').reset_index(drop=True)

# Count unique accounts per device (Device Farming signal)
df["accounts_per_device"] = df.groupby("device_id_hash")["account_id"].transform("nunique")
# Boost for fraud
df.loc[fraud_mask, "accounts_per_device"] += np.random.randint(3, 10, n_fraud)

# Derived Ratios
df["amount_to_avg_ratio"] = df["amount"] / (df["avg_tx_amount_7d"].fillna(df["amount"]) + 1)
df["tx_frequency_ratio"] = df["tx_count_last_1h"] / 4

# -----------------------
# 4. POST-PROCESSING: REALISM
# -----------------------
print("Applying noise and realism...")
# Flip some bits to break perfect correlations (Noise Injection)
def flip_bits(series, rate):
    mask = np.random.rand(len(series)) < rate
    series.loc[mask] = 1 - series.loc[mask]
    return series

df["new_device_flag"] = flip_bits(df["new_device_flag"], NOISE_FLIP_RATE)
df["country_mismatch_flag"] = flip_bits(df["country_mismatch_flag"], NOISE_FLIP_RATE)

# Inject Missing Data (Dirty Data Simulation)
for col in ["geo_country", "currency"]:
    mask = np.random.rand(N_TRANSACTIONS) < MISSING_DATA_RATE
    df.loc[mask, col] = "UNKNOWN"

# -----------------------
# 5. VALIDATION & SAVE
# -----------------------
print("\n--- Validation ---")
print("Dataset Shape:", df.shape)
print(f"Fraud Rate: {df['is_fraud'].mean():.4f}")
print("\nTop Correlations with is_fraud:")
print(df.select_dtypes(include=np.number).corrwith(df["is_fraud"]).sort_values(ascending=False).head(8))

# Save
output_file = "production_fraud_dataset.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset saved to {output_file}")