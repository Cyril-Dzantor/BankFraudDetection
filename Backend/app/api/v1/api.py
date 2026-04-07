from fastapi import APIRouter
from .endpoints import alerts, health, signals, network, dashboard, cases, accounts, audit, models, users, fairness, regulatory, transactions, events, feedback

api_router = APIRouter()
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(signals.router, prefix="/signals", tags=["signals"])
api_router.include_router(network.router, prefix="/network", tags=["network"])
api_router.include_router(cases.router, prefix="/cases", tags=["cases"])
api_router.include_router(accounts.router, prefix="/accounts", tags=["accounts"])
api_router.include_router(audit.router, prefix="/audit", tags=["audit"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(fairness.router, prefix="/fairness", tags=["fairness"])
api_router.include_router(regulatory.router, prefix="/regulatory", tags=["regulatory"])
api_router.include_router(transactions.router, prefix="/transactions", tags=["transactions"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
