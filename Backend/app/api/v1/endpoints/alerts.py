from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlmodel import Session, select, func
from typing import List, Optional
from app.core.db import get_session
from app.core.manager import manager
from app.models.alert import Alert, AlertCreate
from app.schemas.ai import TransactionRequest, FraudVerdict
from app.services.ai_client import ai_client
from app.core.graph import graph_manager
import uuid
import datetime

router = APIRouter()

@router.get("/", response_model=dict)
def read_alerts(
    skip: int = 0,
    limit: int = 100,
    days: Optional[int] = Query(None, description="Filter alerts from the last N days (7, 30, 365)"),
    session: Session = Depends(get_session)
):
    statement = select(Alert)
    if days is not None:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        statement = statement.where(Alert.created_at >= cutoff_str)
    statement = statement.order_by(Alert.id.desc()).offset(skip).limit(limit)
    alerts = session.exec(statement).all()
    count_stmt = select(func.count(Alert.id))
    if days is not None:
        count_stmt = count_stmt.where(Alert.created_at >= cutoff_str)
    total = session.exec(count_stmt).one()
    return {"items": alerts, "total": total}

@router.get("/{alert_id}", response_model=Alert)
def read_alert(alert_id: str, session: Session = Depends(get_session)):
    alert = session.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

@router.post("/", response_model=Alert, status_code=status.HTTP_201_CREATED)
async def create_alert(alert_in: AlertCreate, session: Session = Depends(get_session)):
    db_alert = Alert.from_orm(alert_in)
    db_alert.id = f"ALT-{str(uuid.uuid4())[:8].upper()}"
    session.add(db_alert)
    session.commit()
    session.refresh(db_alert)
    
    # Broadcast the new alert to all connected WebSocket clients
    await manager.broadcast({
        "type": "NEW_ALERT",
        "data": db_alert.dict()
    })
    
    return db_alert

from pydantic import BaseModel
from app.core.audit import audit_manager

class AlertResolveRequest(BaseModel):
    action: str # "APPROVE" or "REJECT"
    actor: str
    
@router.post("/{alert_id}/resolve", response_model=dict)
def resolve_alert(alert_id: str, req: AlertResolveRequest, session: Session = Depends(get_session)):
    alert = session.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    # Hook into Audit Log
    payload = {
        "alert_id": alert.id,
        "customer": alert.customer,
        "amount": alert.amount,
        "action": req.action
    }
    
    audit_manager.log_action(
        session=session,
        actor=req.actor,
        action=f"ALERT_{req.action.upper()}",
        resource=alert_id,
        status="Success",
        risk="Low", # Resolving an alert is an administrative action, usually Low/Med risk
        ip="10.0.0.1", # In a real app, extract from request headers
        payload=payload
    )
    
    session.delete(alert)
    session.commit()
    return {"status": "success", "message": f"Alert {req.action.lower()}d"}
