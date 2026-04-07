from typing import Optional, List, Dict, Any
from sqlmodel import Field, SQLModel, Column, JSON
from datetime import datetime

class AccountProfileBase(SQLModel):
    name: str
    initials: Optional[str] = None
    kyc_level: str
    location: str
    account_type: str
    risk_score: float
    risk_level: str
    account_status: str = Field(default="ACTIVE")
    
    # Store complex nested data as JSON
    linked_cases: List[Any] = Field(sa_column=Column(JSON))
    behavior_data: List[Any] = Field(sa_column=Column(JSON))
    feature_importance: List[Any] = Field(sa_column=Column(JSON))

class AccountProfile(AccountProfileBase, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AccountProfileCreate(AccountProfileBase):
    id: str
