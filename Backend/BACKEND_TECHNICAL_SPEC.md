# Cognize Fraud Intelligence Hub: Technical Specification

## 1. System Architecture Overview

The Cognize Fraud Intelligence Hub is built as a high-performance, asynchronous micro-service designed to handle real-time financial transaction scoring and behavioral analysis.

### Tech Stack
- **Framework**: FastAPI (Python 3.10+)
- **Database**: SQLite (Demo/Sandbox) / SQLModel (ORM)
- **ML Layer**: Custom Neural Scoring Engine (integrated via `ai_client`)
- **Real-time**: WebSockets for live dashboard synchronization
- **Graph Engine**: In-memory relational graph for link analysis

---

## 2. API Ecosystem (The 7-API Cycle)

The platform implements an industry-standard feedback loop across 7 key interfaces:

1.  **Event Ingestion (`/events`)**: Asynchronous capture of non-financial signals (Logins, Device registration).
2.  **Account Sync**: Integration with Core Banking KYC/Profile data.
3.  **Real-time Scoring (`/transactions/score`)**: The critical authorization path for funds approval.
4.  **Alert API (`/alerts`)**: External SIEM/SOAR hook for threat detection.
5.  **Case Management (`/cases`)**: Workflow engine for fraud investigators.
6.  **Decision Feedback (`/feedback/decisions`)**: The "Closed Loop" ground truth used for ML retraining.
7.  **Webhooks**: Outbound callbacks for automated block-list actions.

---

## 3. Data Schema & Persistence

### Core Models
- **Alert**: Primary record of a flagged transaction or behavioral anomaly.
- **Case**: An investigation container for one or more related alerts.
- **Event**: Log of non-financial behavioral actions.
- **DecisionFeedback**: Ground-truth outcome provided by bank investigators.
- **AuditLog**: Immutable trace of all system and user actions for BoG compliance.

### Database Operations
Persistence is managed via **SQLModel**, enabling seamless transitions from SQLite to PostgreSQL for production environments. All tables are automatically initialized on system startup via `SQLModel.metadata.create_all`.

---

## 4. Security & Access Control

### Role-Based Access Control (RBAC)
Access is restricted via the `RoleChecker` dependency.
- **Executive**: Full vision across portfolio KPIs and compliance.
- **Scientist**: Access to model telemetry, fairness metrics, and registry.
- **Analyst**: Focused operational view of alerts and case lineage.

### Data Privacy
Sensitive customer identifiers (PII) are masked using regional branch prefixes and truncated account references (e.g., `**** 1234`) to comply with Data Protection Directives.

---

## 5. Deployment & Integration

### Scaling
The backend is designed for stateless horizontal scaling using **Uvicorn** and **Gunicorn**. 

### Interactive Documentation
A live, interactive API explorer is available at:
`http://localhost:8000/api/v1/docs`

---
**Confidential: For Internal Bank Use Only**
