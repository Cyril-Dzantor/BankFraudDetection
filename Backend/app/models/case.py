from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime

class CaseBase(SQLModel):
    title: str
    customer_id: Optional[str] = Field(default=None, index=True)
    assignee: str
    status: str = Field(index=True)
    priority: str = Field(index=True)
    value: str
    type: str
    tags: str # Comma-separated tags
    created: str
    updated: str
    notes: Optional[str] = None

class Case(CaseBase, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class CaseCreate(CaseBase):
    pass

class CaseUpdate(SQLModel):
    title: Optional[str] = None
    assignee: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    value: Optional[str] = None
    type: Optional[str] = None
    tags: Optional[str] = None
    updated: Optional[str] = None
    notes: Optional[str] = None
