from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from typing import List
from app.core.db import get_session
from app.models.case import Case, CaseCreate, CaseUpdate
import uuid
import datetime

router = APIRouter()

@router.get("/", response_model=dict)
def read_cases(skip: int = 0, limit: int = 100, session: Session = Depends(get_session)):
    total = session.exec(select(func.count(Case.id))).one()
    # Sort by created_at desc for recent first
    cases = session.exec(select(Case).offset(skip).limit(limit)).all()
    # Sort in python for simplicity since SQLite datetime ordering can be tricky
    cases.sort(key=lambda x: x.created_at, reverse=True)
    return {"items": cases, "total": total}

@router.post("/", response_model=Case)
def create_case(case_in: CaseCreate, session: Session = Depends(get_session)):
    # Generate random CAS-YYYY-XXXX ID
    import random
    new_id = f"CAS-2024-{random.randint(1000, 9999)}"
    
    db_case = Case.from_orm(case_in)
    db_case.id = new_id
    
    session.add(db_case)
    session.commit()
    session.refresh(db_case)
    return db_case

@router.patch("/{case_id}", response_model=Case)
def update_case(case_id: str, case_in: CaseUpdate, session: Session = Depends(get_session)):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    old_status = case.status
    old_priority = case.priority
        
    case_data = case_in.dict(exclude_unset=True)
    for key, value in case_data.items():
        setattr(case, key, value)
        
    case.updated_at = datetime.datetime.utcnow()
    case.updated = "Just now"

    from app.core.audit import audit_manager
    payload = {
        "case_id": case.id,
        "old_status": old_status,
        "new_status": case.status,
        "old_priority": old_priority,
        "new_priority": case.priority,
    }
    action = "CASE_RESOLVE" if case.status == "Resolved" else "CASE_MODIFY"
    
    audit_manager.log_action(
        session=session,
        actor="Analyst", # Mocked for now; would extract from token
        action=action,
        resource=case_id,
        status="Success",
        risk="Medium" if action == "CASE_MODIFY" else "Low",
        ip="10.0.0.1",
        payload=payload
    )
    
    session.add(case)
    session.commit()
    session.refresh(case)
    return case
