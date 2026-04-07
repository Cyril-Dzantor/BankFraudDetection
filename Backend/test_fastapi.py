from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

payload = {
    "account_number": "ACC-123456789",
    "cif_id": "CIF-987654321",
    "amount": 95.00,
    "channel": "mobile",
    "auth_method": "biometric",
    "geo_country": "GH",
    "currency": "GHS",
    "device_id": "DEV-ABCDEF1234",
    "ip_address": "192.168.1.100"
}

response = client.post("/api/v1/alerts/score", json=payload)
print("STATUS:", response.status_code)
print("BODY:", response.json())
