from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from app.core.db import get_session
from app.models.user_provision import ProvisionedUser, ProvisionedUserCreate, UserLogin
from app.api.deps import RoleChecker, get_current_role
from datetime import datetime

router = APIRouter()

@router.get("/me", response_model=ProvisionedUser)
def get_me(email: str, session: Session = Depends(get_session)):
    user = session.exec(select(ProvisionedUser).where(ProvisionedUser.email == email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/", response_model=List[ProvisionedUser])
def read_users(
    session: Session = Depends(get_session),
    current_role: str = Depends(RoleChecker(["system_admin", "compliance_lead"]))
):
    users = session.exec(select(ProvisionedUser).order_by(ProvisionedUser.created_at.desc())).all()
    return users

@router.post("/", response_model=ProvisionedUser)
def create_user(
    user: ProvisionedUserCreate, 
    session: Session = Depends(get_session),
    current_role: str = Depends(RoleChecker(["system_admin"]))
):
    db_user = ProvisionedUser(**user.model_dump())
    db_user.status = "PENDING"
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user

@router.post("/login", response_model=ProvisionedUser)
def login(login_data: UserLogin, session: Session = Depends(get_session)):
    user = session.exec(
        select(ProvisionedUser)
        .where(ProvisionedUser.email == login_data.email)
        .where(ProvisionedUser.password == login_data.password)
        .where(ProvisionedUser.status == "APPROVED")
    ).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials or account pending approval")
    return user

@router.patch("/approve/{user_id}", response_model=ProvisionedUser)
def approve_user(
    user_id: str, 
    session: Session = Depends(get_session),
    current_role: str = Depends(RoleChecker(["compliance_lead"]))
):
    user = session.get(ProvisionedUser, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.status = "APPROVED"
    user.approved_at = datetime.utcnow()
    # In a real app we'd track who approved it via something like user.checker_id = current_user.id
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
