from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select, func
from typing import List
import uuid
import hashlib
import json
import io
import csv
from app.core.db import get_session
from app.models.audit import AuditLog, AuditLogCreate
from datetime import datetime
from app.api.deps import RoleChecker

router = APIRouter(dependencies=[Depends(RoleChecker(["system_admin", "compliance_lead"]))])

@router.get("/", response_model=dict)
def read_audit_logs(skip: int = 0, limit: int = 15, session: Session = Depends(get_session)):
    total = session.exec(select(func.count(AuditLog.id))).one()
    statement = select(AuditLog).order_by(AuditLog.timestamp.desc()).offset(skip).limit(limit)
    logs = session.exec(statement).all()
    return {"items": logs, "total": total}

@router.get("/export")
def export_audit_logs(session: Session = Depends(get_session)):
    statement = select(AuditLog).order_by(AuditLog.timestamp.desc())
    logs = session.exec(statement).all()
    
    stream = io.StringIO()
    writer = csv.writer(stream)
    
    # CSV Header
    writer.writerow(["ID", "Timestamp", "Actor", "Action", "Resource", "Status", "Risk", "IP", "Payload", "Previous Hash"])
    
    # Write Logs
    for l in logs:
        writer.writerow([l.id, l.timestamp, l.actor, l.action, l.resource, l.status, l.risk, l.ip, l.payload, l.previous_hash])
        
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=audit_compliance_pack.csv"
    return response
