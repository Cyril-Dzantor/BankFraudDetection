from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class TransactionRequest(BaseModel):
    account_number: str = Field(..., description="Unique account identifier")
    cif_id: str = Field(..., description="Unique customer identifier")
    amount: float = Field(..., description="Transaction amount")
    channel: str = Field(..., description="Transaction channel: 'mobile', 'web', 'atm', 'pos'")
    auth_method: str = Field(..., description="Authentication method: 'pin', 'biometric', 'otp', 'password'")
    geo_country: str = Field(..., description="Country code of the transaction, e.g. 'GH', 'NG'")
    currency: str = Field(..., description="Currency code, e.g. 'GHS', 'USD'")
    device_id: str = Field(..., description="Device identifier (hash)")
    ip_address: str = Field(..., description="IP Address of the client")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

class FraudVerdict(BaseModel):
    decision: str = Field(..., description="'APPROVE', 'CHALLENGE', or 'DECLINE'")
    is_early_exit: bool = Field(..., description="True if Rules Engine issued BLOCK and ML was skipped")
    fraud_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    reconstruction_error: Optional[float] = None
    confidence: str = Field(..., description="'HIGH', 'MEDIUM', or 'LOW'")
    triggered_by: str = Field(..., description="What caused this decision")
    reason: str = Field(..., description="Human-readable explanation of the verdict")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp")
