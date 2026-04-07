from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime


class AuditLogBase(SQLModel):
    timestamp: str
    actor: str
    action: str
    resource: str
    status: str
    risk: str
    ip: str
    payload: Optional[str] = Field(default=None, description="JSON representing state change")
    previous_hash: Optional[str] = Field(default=None, description="Hash of the previous log for immutability")

class AuditLog(AuditLogBase, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AuditLogCreate(AuditLogBase):
    id: Optional[str] = None
