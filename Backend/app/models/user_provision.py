from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime
import uuid


class ProvisionedUserBase(SQLModel):
    full_name: str
    email: str
    employee_id: str
    department: str
    role: str
    status: str = "PENDING"
    password: Optional[str] = None


class ProvisionedUser(ProvisionedUserBase, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    checker_id: Optional[str] = None
    approved_at: Optional[datetime] = None


class ProvisionedUserCreate(ProvisionedUserBase):
    pass

class UserLogin(SQLModel):
    email: str
    password: str
