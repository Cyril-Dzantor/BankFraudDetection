from typing import List
from collections import Counter
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.core.db import get_session
from app.models.account import AccountProfile
from app.models.alert import Alert

router = APIRouter()

@router.get("/", response_model=List[AccountProfile])
def read_accounts(
    session: Session = Depends(get_session),
    offset: int = 0,
    limit: int = 100
):
    accounts = session.exec(select(AccountProfile).offset(offset).limit(limit)).all()
    return accounts

@router.get("/{account_id}/derived")
def get_derived_profile(account_id: str, session: Session = Depends(get_session)):
    """
    Compute a lightweight intelligence profile for a bank account directly 
    from the Alert history. Serves as a bridge until the Post-Auth 
    Intelligence layer is complete and writing to AccountProfile properly.
    """
    alerts = session.exec(
        select(Alert).where(Alert.acctStart == account_id).order_by(Alert.id.desc())
    ).all()

    if not alerts:
        raise HTTPException(
            status_code=404,
            detail=f"No alert history found for account {account_id}. The account may not yet have any flagged transactions."
        )

    # --- Derived Intelligence ---
    scores = [a.score for a in alerts]
    avg_score = round(sum(scores) / len(scores), 1)
    max_score = max(scores)

    # Mode: Most common channel and device
    channels = [a.channel for a in alerts if a.channel]
    devices = [a.device for a in alerts if a.device]
    top_channel = Counter(channels).most_common(1)[0][0] if channels else "Unknown"
    top_device = Counter(devices).most_common(1)[0][0] if devices else "Unknown"

    # Risk level distribution
    risk_counts = Counter([a.riskLevel for a in alerts])

    # Determine overall risk level from highest seen
    overall_risk = "Low"
    if risk_counts.get("Critical", 0) > 0:
        overall_risk = "Critical"
    elif risk_counts.get("High", 0) > 0:
        overall_risk = "High"
    elif risk_counts.get("Medium", 0) > 0:
        overall_risk = "Medium"

    # Recent flags (last 5)
    recent_alerts = [
        {
            "id": a.id,
            "amount": a.amount,
            "channel": a.channel,
            "riskLevel": a.riskLevel,
            "score": a.score,
            "reason": a.reason,
            "time": a.time,
            "location": a.location,
            "device": a.device,
            "transaction_type": a.transaction_type,
            "recipient_name": a.recipient_name,
        }
        for a in alerts[:5]
    ]

    return {
        "account_id": account_id,
        "customer_name": alerts[0].customer if alerts else account_id,
        "total_alerts": len(alerts),
        "avg_fraud_score": avg_score,
        "peak_fraud_score": max_score,
        "overall_risk_level": overall_risk,
        "risk_distribution": dict(risk_counts),
        "top_channel": top_channel,
        "top_device": top_device,
        "data_source": "alert_history",
        "note": "Profile derived from Alert history. Full Post-Auth intelligence pending.",
        "recent_flags": recent_alerts,
    }

@router.get("/{account_id}", response_model=AccountProfile)
def read_account(account_id: str, session: Session = Depends(get_session)):
    account = session.get(AccountProfile, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account profile not found")
    return account
