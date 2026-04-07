# Cognize Fraud Detection - Enterprise API Demonstration Script (Windows/PowerShell)
# This script demonstrates the 3 key APIs implemented for bank-grade integration.

$baseUrl = "http://localhost:8000/api/v1"

Write-Host "`n------------------------------------------------" -ForegroundColor Cyan
Write-Host "1. REAL-TIME TRANSACTION SCORING API" -ForegroundColor Cyan
Write-Host "Endpoint: POST /transactions/score" -ForegroundColor Cyan
Write-Host "------------------------------------------------"

$scoringPayload = @{
    tx_count_last_5m = 12
    tx_count_last_1h = 45
    tx_frequency_ratio = 3.5
    amount = 12500.00
    avg_tx_amount_7d = 450.00
    amount_to_avg_ratio = 27.7
    new_device_flag = 1
    device_seen_before = 0
    country_mismatch_flag = 1
    accounts_per_device = 3
    accounts_per_ip_24h = 5
    failed_login_count_last_1h = 2
    failed_logins_then_success = 1
    channel = "web"
    auth_method = "password"
    geo_country = "GH"
    currency = "GHS"
}

try {
    $scoringResponse = Invoke-RestMethod -Uri "$baseUrl/transactions/score" -Method Post -Body ($scoringPayload | ConvertTo-Json) -ContentType "application/json"
    $scoringResponse | ConvertTo-Json -Depth 10 | Write-Host
    $alertId = $scoringResponse.alert_id
} catch {
    Write-Host "Scoring API Failed: $_" -ForegroundColor Red
    exit
}

Write-Host "`n------------------------------------------------" -ForegroundColor Green
Write-Host "2. EVENT INGESTION API (Behavioral Data)" -ForegroundColor Green
Write-Host "Endpoint: POST /events" -ForegroundColor Green
Write-Host "------------------------------------------------"

$eventPayload = @{
    event_type = "BENEFICIARY_CREATED"
    customer_id = "CUST-9921"
    account_id = "ACC-0012"
    channel = "mobile"
    ip_address = "192.168.1.45"
    device_id = "DEV-IPHONE-15"
    geo_country = "GH"
}

try {
    $eventResponse = Invoke-RestMethod -Uri "$baseUrl/events/" -Method Post -Body ($eventPayload | ConvertTo-Json) -ContentType "application/json"
    $eventResponse | ConvertTo-Json -Depth 10 | Write-Host
} catch {
    Write-Host "Event Ingestion Failed: $_" -ForegroundColor Red
}

Write-Host "`n------------------------------------------------" -ForegroundColor Yellow
Write-Host "3. DECISION FEEDBACK API (Ground Truth)" -ForegroundColor Yellow
Write-Host "Endpoint: POST /feedback/decisions" -ForegroundColor Yellow
Write-Host "------------------------------------------------"

$feedbackPayload = @{
    transaction_id = $alertId
    alert_id = $alertId
    original_decision = "DECLINE"
    final_bank_outcome = "FRAUD_CONFIRMED"
    analyst_label = "Confirmed ATO"
    notes = "Customer confirmed they did not authorize this web login/transfer."
}

try {
    $feedbackResponse = Invoke-RestMethod -Uri "$baseUrl/feedback/decisions" -Method Post -Body ($feedbackPayload | ConvertTo-Json) -ContentType "application/json"
    $feedbackResponse | ConvertTo-Json -Depth 10 | Write-Host
} catch {
    Write-Host "Decision Feedback Failed: $_" -ForegroundColor Red
}

Write-Host "`n------------------------------------------------" -ForegroundColor Cyan
Write-Host "DEMO COMPLETE" -ForegroundColor Cyan
Write-Host "------------------------------------------------"
