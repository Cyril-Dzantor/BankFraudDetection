from typing import List, Dict, Any, Optional
from sqlmodel import Field, SQLModel, Column, JSON
from datetime import datetime


class RiskModel(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    status: str
    accuracy: str
    precision: str
    recall: str
    latency: str
    last_trained: str
    type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelMetrics(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(foreign_key="riskmodel.id")
    performance_data: List[Dict[str, Any]] = Field(sa_column=Column(JSON))
    drift_data: List[Dict[str, Any]] = Field(sa_column=Column(JSON))
    distribution_shift: List[Dict[str, Any]] = Field(sa_column=Column(JSON))
