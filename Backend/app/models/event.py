from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class EventBase(SQLModel):
    event_type: str = Field(..., description="The category of the behavioral event (e.g., FAILED_LOGIN, BENEFICIARY_CREATED).")
    customer_id: str = Field(..., description="The unique customer identifier associated with this event.")
    account_id: str = Field(..., description="The bank account ID involved in the event.")
    channel: str = Field(..., description="The digital or physical channel where the event originated.")
    ip_address: Optional[str] = Field(None, description="The IP address of the client device.")
    device_id: Optional[str] = Field(None, description="A unique fingerprint or identifier for the device used.")
    geo_country: Optional[str] = Field(None, description="The 2-letter ISO country code where the event was localized.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Event(EventBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

class EventCreate(EventBase):
    pass
