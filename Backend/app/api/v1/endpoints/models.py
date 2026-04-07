from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from typing import List
from app.core.db import get_session
from app.models.riskmodel import RiskModel, ModelMetrics
from app.api.deps import RoleChecker

router = APIRouter()

@router.get("/", response_model=dict)
def read_models(skip: int = 0, limit: int = 10, session: Session = Depends(get_session)):
    total = session.exec(select(func.count(RiskModel.id))).one()
    models = session.exec(select(RiskModel).offset(skip).limit(limit)).all()
    return {"items": models, "total": total}

@router.get("/summary", response_model=dict)
def get_model_summary(session: Session = Depends(get_session)):
    models = session.exec(select(RiskModel)).all()
    if not models:
        return {
            "global_accuracy": "0.0%",
            "fraud_recall": "0.0%",
            "fpr": "0.0%",
            "inference_latency": "0ms"
        }
    
    accuracies = [float(m.accuracy.strip('%')) for m in models]
    recalls = [float(m.recall.strip('%')) for m in models]
    latencies = [int(m.latency.strip('ms')) for m in models]
    
    return {
        "global_accuracy": f"{sum(accuracies) / len(accuracies):.1f}%",
        "fraud_recall": f"{sum(recalls) / len(recalls):.1f}%",
        "inference_latency": f"{int(sum(latencies) / len(latencies))}ms"
    }

@router.get("/{model_id}/metrics", response_model=ModelMetrics)
def read_model_metrics(model_id: str, session: Session = Depends(get_session)):
    metrics = session.exec(
        select(ModelMetrics).where(ModelMetrics.model_id == model_id)
    ).first()
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not found for this model")
    return metrics
