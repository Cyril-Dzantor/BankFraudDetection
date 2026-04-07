from pydantic import BaseModel
from typing import Optional

class AlertBase(BaseModel):
    id: str
    branch: str
    customer: str
    acctStart: str
    initials: str
    amount: str
    riskLevel: str
    score: int
    channel: str
    channelIconType: str
    time: str
    location: str
    device: str

class Alert(AlertBase):
    pass
