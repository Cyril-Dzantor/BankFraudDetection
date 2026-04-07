from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from app.core.db import get_session
from app.models.fairness import FairnessSnapshot
from app.api.deps import RoleChecker

router = APIRouter(dependencies=[Depends(RoleChecker(["executive", "scientist"]))])

@router.get("/", response_model=FairnessSnapshot)
def read_fairness_snapshot(session: Session = Depends(get_session)):
    snapshot = session.exec(select(FairnessSnapshot).order_by(FairnessSnapshot.id.desc())).first()
    if not snapshot:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No fairness snapshot available")
    return snapshot
