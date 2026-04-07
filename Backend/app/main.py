from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel
from .api.v1.api import api_router
from .core.config import settings
from .core.db import engine
from .core.seed import seed_db

# Import models to ensure they are registered with SQLModel
from .models.alert import Alert 
from .models.case import Case
from .models.account import AccountProfile
from .models.audit import AuditLog
from .models.riskmodel import RiskModel, ModelMetrics
from .models.user_provision import ProvisionedUser
from .models.fairness import FairnessSnapshot
from .models.event import Event
from .models.feedback import DecisionFeedback

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    seed_db()

app = FastAPI(
    title="FraudSense AI",
    description="""
FraudSense AI is a bank-grade fraud detection and prevention platform. 
It provides real-time transaction scoring, behavioral event ingestion, and closed-loop decision feedback.

### Key Capabilities:
* **Real-time Scoring**: High-throughput ML-driven transaction verdicts.
* **Event Ingestion**: Advanced behavioral signal tracking for customer context.
* **Decision Feedback**: Ground-truth labels for model retraining.
* **Operational APIs**: Full access to alerts, cases, and audit logs.
""",
    version="2.1.0 (Enterprise)",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    contact={
        "name": "FraudSense Engineering",
        "url": "http://localhost:3000/support",
    }
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# Set all CORS enabled origins
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
if settings.BACKEND_CORS_ORIGINS:
    for origin in settings.BACKEND_CORS_ORIGINS:
        if origin not in origins:
            origins.append(str(origin))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "FraudSense AI API - Operational"}
