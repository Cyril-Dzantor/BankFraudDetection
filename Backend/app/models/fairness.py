from typing import List, Dict, Any, Optional
from sqlmodel import Field, SQLModel, Column, JSON


class FairnessSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    audited_decisions: int
    approval_rate: float
    fairness_score: float
    pending_reviews: int
    gender_data: List[Dict[str, Any]] = Field(sa_column=Column(JSON))
    region_data: List[Dict[str, Any]] = Field(sa_column=Column(JSON))
    age_data: List[Dict[str, Any]] = Field(sa_column=Column(JSON))
