#!/bin/bash

# Cognize Fraud Detection - Enterprise API Demonstration Script
# This script demonstrates the 3 key APIs implemented for bank-grade integration.

BASE_URL="http://localhost:8000/api/v1"

echo "------------------------------------------------"
echo "1. REAL-TIME TRANSACTION SCORING API"
echo "Endpoint: POST /transactions/score"
echo "------------------------------------------------"

RESPONSE=$(curl -s -X POST "$BASE_URL/transactions/score" \
  -H "Content-Type: application/json" \
  -d '{
    "tx_count_last_5m": 12,
    "tx_count_last_1h": 45,
    "tx_frequency_ratio": 3.5,
    "amount": 12500.00,
    "avg_tx_amount_7d": 450.00,
    "amount_to_avg_ratio": 27.7,
    "new_device_flag": 1,
    "device_seen_before": 0,
    "country_mismatch_flag": 1,
    "accounts_per_device": 3,
    "accounts_per_ip_24h": 5,
    "failed_login_count_last_1h": 2,
    "failed_logins_then_success": 1,
    "channel": "web",
    "auth_method": "password",
    "geo_country": "GH",
    "currency": "GHS"
  }')

echo "$RESPONSE" | python -m json.tool
ALERT_ID=$(echo "$RESPONSE" | grep -oP '"alert_id":\s*"\K[^"]+')

echo -e "\n------------------------------------------------"
echo "2. EVENT INGESTION API (Behavioral Data)"
echo "Endpoint: POST /events"
echo "------------------------------------------------"

curl -s -X POST "$BASE_URL/events/" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "BENEFICIARY_CREATED",
    "customer_id": "CUST-9921",
    "account_id": "ACC-0012",
    "channel": "mobile",
    "ip_address": "192.168.1.45",
    "device_id": "DEV-IPHONE-15",
    "geo_country": "GH"
  }' | python -m json.tool

echo -e "\n------------------------------------------------"
echo "3. DECISION FEEDBACK API (Ground Truth)"
echo "Endpoint: POST /feedback/decisions"
echo "------------------------------------------------"

curl -s -X POST "$BASE_URL/feedback/decisions" \
  -H "Content-Type: application/json" \
  -d "{
    \"transaction_id\": \"$ALERT_ID\",
    \"alert_id\": \"$ALERT_ID\",
    \"original_decision\": \"DECLINE\",
    \"final_bank_outcome\": \"FRAUD_CONFIRMED\",
    \"analyst_label\": \"Confirmed ATO\",
    \"notes\": \"Customer confirmed they did not authorize this web login/transfer.\"
  }" | python -m json.tool

echo -e "\n------------------------------------------------"
echo "DEMO COMPLETE"
echo "------------------------------------------------"
