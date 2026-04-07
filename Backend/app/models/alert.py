from typing import Optional
from sqlmodel import Field, SQLModel
import datetime

class AlertBase(SQLModel):
    branch: str
    customer: str
    customer_id: Optional[str] = Field(default=None, index=True)
    acctStart: str
    initials: str
    amount: str
    riskLevel: str = Field(index=True)
    score: int
    channel: str
    channelIconType: str
    time: str
    location: str
    device: str = Field(index=True)
    reason: str = Field(default="", description="Human-readable explanation of the verdict")
    triggered_rules: str = Field(default="", description="Comma-separated list of rules that fired")
    
    # Enrichment Details
    transaction_type: Optional[str] = Field(None, description="e.g. P2P Transfer")
    recipient_account: Optional[str] = Field(None, description="Destination account")
    recipient_name: Optional[str] = Field(None, description="Name of recipient")
    transaction_notes: Optional[str] = Field(None, description="Memo/notes")

class Alert(AlertBase, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    created_at: Optional[str] = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat(), index=True)

class AlertCreate(AlertBase):
    pass

class AlertUpdate(SQLModel):
    branch: Optional[str] = None
    customer: Optional[str] = None
    riskLevel: Optional[str] = None
    score: Optional[int] = None
    reason: Optional[str] = None
    triggered_rules: Optional[str] = None
