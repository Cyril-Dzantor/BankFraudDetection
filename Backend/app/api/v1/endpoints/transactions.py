from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel import Session
from app.core.db import get_session
from app.core.manager import manager
from app.models.alert import Alert
from app.schemas.ai import TransactionRequest, FraudVerdict
from app.services.ai_client import ai_client
from app.core.graph import graph_manager
from pydantic import BaseModel, Field
import uuid

router = APIRouter()

class ScoreResponse(FraudVerdict):
    alert_id: str = Field(..., description="The unique identifier generated for this transaction alert in the Cognize ecosystem.")

@router.post("/score", 
             response_model=ScoreResponse, 
             status_code=status.HTTP_201_CREATED,
             summary="Score Real-Time Transaction",
             description="""
Evaluates a financial transaction against the neural scoring engine and rules pipeline.
This is the primary endpoint for bank authorization systems (Core Banking).

Returns a decision (APPROVE/CHALLENGE/DECLINE) along with deep telemetry including:
- **Fraud Probability**: ML-predicted likelihood of illicit activity.
- **Explainability**: Human-readable reasoning behind the verdict.
- **Anomaly Detection**: Behavioral deviation scores from established baselines.
""")
async def score_transaction(request: TransactionRequest, session: Session = Depends(get_session)):
    print(f"\n[BACKEND LOG] --- INCOMING TRANSACTION SCORE REQUEST ---\n{request.model_dump_json(indent=2)}\n------------------------------")
    
    # 1. Call AI Service
    verdict = await ai_client.score_transaction(request)
    
    if not verdict:
        raise HTTPException(status_code=503, detail="AI Service unavailable")
    
    # 2. Map Verdict to Alert
    txn_id = f"TXN-{uuid.uuid4().hex[:5].upper()}"
    
    # Determine risk level from decision
    risk_level = "Medium"
    if verdict.decision == "APPROVE": risk_level = "Low"
    elif verdict.decision == "CHALLENGE": risk_level = "High"
    elif verdict.decision == "DECLINE": risk_level = "Critical"
    
    # Map channel icon
    icon_type = "smartphone"
    if request.channel == "web": icon_type = "globe"
    elif request.channel in ["atm", "pos"]: icon_type = "credit-card"
    
    # Deriving initials from a mock customer for demo
    customer_name = "Digital Transaction"
    if request.geo_country == "GH": customer_name = "Ghanaian Retailer"
    elif request.geo_country == "NG": customer_name = "Nigerian Merchant"
    
    db_alert = Alert(
        id=txn_id,
        branch=f"Regional-{request.geo_country}",
        customer=customer_name,
        acctStart="**** " + str(uuid.uuid4().int)[:4],
        initials="DT",
        amount=f"{request.amount:,.2f}",
        riskLevel=risk_level,
        score=int(verdict.fraud_score * 100) if verdict.fraud_score is not None else (95 if verdict.decision == "DECLINE" else 70 if verdict.decision == "CHALLENGE" else 20),
        channel=request.channel.capitalize(),
        channelIconType=icon_type,
        time="Just Now",
        location=f"{request.geo_country}, Remote",
        device=f"{request.auth_method.capitalize()} Auth"
    )
    
    # 3. Save to DB
    session.add(db_alert)
    session.commit()
    session.refresh(db_alert)
    
    # 3.5 Sync to Graph
    try:
        await graph_manager.sync_transaction(db_alert.dict())
    except Exception as e:
        print(f"Graph Sync Failed: {e}")
    
    # 4. Broadcast
    await manager.broadcast({
        "type": "NEW_ALERT",
        "data": db_alert.dict()
    })
    
    return ScoreResponse(**verdict.dict(), alert_id=db_alert.id)

