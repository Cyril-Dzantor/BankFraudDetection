from fastapi import APIRouter, Depends, status
from sqlmodel import Session
from app.core.db import get_session
from app.models.feedback import DecisionFeedback, DecisionFeedbackCreate

router = APIRouter()

@router.post("/decisions", 
             response_model=DecisionFeedback, 
             status_code=status.HTTP_201_CREATED,
             summary="Report Decision Feedback",
             description="""
Provides the 'Ground Truth' outcome for a previously scored transaction.
This endpoint is critical for the 'Closed Loop' intelligence cycle, enabling:
- **Model Calibration**: Continuous improvement of precision/recall metrics.
- **Drift Detection**: Identifying shifts in fraud patterns over time.
- **Investigator Attribution**: Linking bank outcomes to platform alerts.
""")
def create_decision_feedback(feedback_in: DecisionFeedbackCreate, session: Session = Depends(get_session)):
    db_feedback = DecisionFeedback.from_orm(feedback_in)
    session.add(db_feedback)
    session.commit()
    session.refresh(db_feedback)
    return db_feedback
