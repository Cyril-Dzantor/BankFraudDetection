from fastapi import APIRouter, Depends, status
from sqlmodel import Session
from app.core.db import get_session
from app.models.event import Event, EventCreate

router = APIRouter()

@router.post("/", 
             response_model=Event, 
             status_code=status.HTTP_201_CREATED,
             summary="Ingest Behavioral Event",
             description="""
Captures non-financial behavioral markers from banking channels (Mobile, Web, ATM).
Used to build high-fidelity behavioral context and identify potential Account Takeover (ATO) patterns.

**Supported Events**: FAILED_LOGIN, BENEFICIARY_CREATED, DEVICE_CHANGE, PROFILE_UPDATE, etc.
""")
def create_event(event_in: EventCreate, session: Session = Depends(get_session)):
    db_event = Event.from_orm(event_in)
    session.add(db_event)
    session.commit()
    session.refresh(db_event)
    return db_event
