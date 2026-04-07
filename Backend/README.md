# The AI Foundry - Fraud Intelligence Backend

This is the enterprise-grade FastAPI backend for **The AI Foundry** Fraud Detection and Prevention System.

## 🚀 Overview

The AI Foundry is a bank-grade platform designed for real-time transaction scoring, behavioral event ingestion, and closed-loop decision feedback. It leverages machine learning and sophisticated rules engines to detect illicit activity at scale.

### Key Capabilities
*   **Real-time Scoring**: High-throughput ML-driven transaction verdicts.
*   **Event Ingestion**: Advanced behavioral signal tracking for customer context.
*   **Decision Feedback**: Ground-truth labels for model retraining.
*   **Operational APIs**: Full access to alerts, cases, and audit logs.

## 📈 Current Status: Phase 4 (ML Integration)

The AI Foundry is currently in **Phase 4: ML Integration**. 
-   **Phase 1-2 (Foundation & Signals)**: Completed. Alerts and cases are fully stateful and real-time.
-   **Phase 4 (Current)**: The system is now integrated with the Custom Neural Scoring Engine for real-time risk assessment via the `/transactions/score` endpoint.
-   **Next Steps**: Refining the Network Graph (Phase 3) for deep link analysis.

## 🏗️ Project Structure

```text
app/
├── api/             # API Route Definitions
│   └── v1/          # Version 1 Endpoints (Alerts, Cases, Audit, etc.)
├── core/            # Configuration, Security, DB Setup, and Seeding
├── models/          # Database Models (SQLModel)
├── schemas/         # Pydantic Validation Schemas
└── services/        # External Service Clients (AI Engine, etc.)
```

## 📊 Data Management

The AI Foundry uses a hybrid approach for data handling to support both functional operations and high-fidelity demonstrations.

### Persistent Data (Database)
The system uses **SQLModel** with a **SQLite** backend (mapped to `fraud_detection.db`). Core entities are fully stateful:
-   **Alerts & Cases**: Core workflow items.
-   **Audit Logs**: Comprehensive trace of system and user actions.
-   **Risk Models**: Metadata for active ML deployments.
-   **User Provisioning**: System access and identity management.

*Note: The database is automatically seeded with initial mock data from `app/core/seed.py` on first startup.*

### Demonstration Logic (Mock Data)
To ensure a rich experience in demo environments, certain analytics are programmatically generated or mapped:
-   **Metrics**: The **Precision Rate** KPI is currently static (98.4%).
-   **Geographical Mapping**: Map markers map region names in the DB to hardcoded GPS coordinates in `dashboard.py`.
-   **Trend Analysis**: Volume trends and fraud distribution charts use derived logic based on current database counts.

## 📡 API Reference

The interactive Swagger documentation is available at [/docs](http://localhost:8000/docs) when the server is running.

### Version 1 Endpoints
-   `GET /api/v1/dashboard/stats`: KPI summaries and chart data.
-   `GET /api/v1/alerts/`: Transaction alert queue.
-   `POST /api/v1/transactions/score`: Primary entry point for real-time scoring.
-   `GET /api/v1/cases/`: Investigation case management.
-   `GET /api/v1/audit/`: System-wide audit logs.

## 🛠️ Getting Started

1.  **Setup Virtual Environment**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
2.  **Install Dependencies**:
    ```powershell
    python -m pip install -r requirements.txt
    ```
3.  **Run the Server**:
    ```powershell
    uvicorn app.main:app --reload
    ```
