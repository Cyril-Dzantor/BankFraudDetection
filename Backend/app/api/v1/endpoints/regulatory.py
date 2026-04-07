from fastapi import APIRouter, Depends
from sqlmodel import Session, select, func
from typing import List, Dict
from app.core.db import get_session
from app.models.alert import Alert
from app.models.case import Case
from app.models.audit import AuditLog
from app.api.deps import RoleChecker

router = APIRouter(dependencies=[Depends(RoleChecker(["executive", "analyst"]))])

@router.get("/compliance-stats", response_model=Dict)
def get_compliance_stats(session: Session = Depends(get_session)):
    """
    Returns high-level compliance KPIs for the regulatory portal.
    """
    # STR Submissions (Mocked but scalable)
    total_str = 84 # Suspicious Transaction Reports
    target_str = 92
    
    # Internal Audits
    pending_audits = session.exec(select(func.count(AuditLog.id)).where(AuditLog.status == "PENDING")).one()
    
    # Compliance Health Score (Mock logic: 100 - (pending audits * 2))
    health_score = max(0, 100 - (pending_audits * 2))
    
    return {
        "compliance_health": health_score,
        "str_submissions": f"{total_str} / {target_str}",
        "pending_audits": pending_audits
    }

@router.get("/filings", response_model=List[Dict])
def get_statutory_filings():
    """
    Returns the status of key statutory filings as required by BoG.
    """
    return [
        { "law": 'Anti-Money Laundering Act, 2020', "code": 'Act 1044', "status": 'In Sync', "color": 'text-emerald-500 bg-emerald-100/30' },
        { "law": 'Payment Systems & Services Act', "code": 'Act 987', "status": 'In Sync', "color": 'text-emerald-500 bg-emerald-100/30' },
        { "law": 'Data Protection Directive', "code": 'BoG G4.1', "status": 'Pending Review', "color": 'text-orange-500 bg-orange-100/30' },
        { "law": 'Cybersecurity Regulation', "code": 'CS 2011', "status": 'Delayed Sync', "color": 'text-red-500 bg-red-100/30' }
    ]

@router.get("/str-queue", response_model=dict)
def get_str_queue(skip: int = 0, limit: int = 5, session: Session = Depends(get_session)):
    """
    Returns alerts that are flagged for STR (Suspicious Transaction Report) filing.
    """
    # For the demo, we'll return Critical alerts as potential STR candidates
    total = session.exec(select(func.count(Alert.id)).where(Alert.riskLevel == "Critical")).one()
    statement = select(Alert).where(Alert.riskLevel == "Critical").offset(skip).limit(limit)
    alerts = session.exec(statement).all()
    
    queue = []
    for alert in alerts:
        queue.append({
            "id": f"STR-GH-{alert.id[-4:]}",
            "type": "SAR / AML",
            "risk": "CRITICAL" if alert.amount > 5000 else "HIGH",
            "status": "Pending",
            "color": "text-orange-500"
        })
        
    return {"items": queue, "total": total}
