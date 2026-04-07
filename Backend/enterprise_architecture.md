# Enterprise Fraud Detection Architecture

The following diagram illustrates the bank-grade API ecosystem implemented in the Cognize Fraud Detection Platform.

```mermaid
graph TD
    subgraph "External Bank Systems"
        Events["Event Ingestion API<br/>(Logins, Device, Profile)"]
        Profiles["Account Profile API<br/>(KYC, Risk Segments)"]
        Inference["Transaction Scoring API<br/>(Real-time Verdicts)"]
        Feedback["Decision Feedback API<br/>(Ground Truth labels)"]
        Webhooks["Webhooks / Callbacks<br/>(Automated Actions)"]
    end

    subgraph "Cognize Intelligence Hub"
        Engine["ML Scoring Engine"]
        Context["Behavioral Context Store"]
        Alerts_API["Alert API"]
        Cases_API["Case Management API"]
    end

    %% Flow: Ingestion & Profiles
    Events --> Context
    Profiles --> Context

    %% Flow: Scoring
    Inference --> Engine
    Context --> Engine
    Engine --> Inference
    Engine --> Alerts_API

    %% Flow: Operations
    Alerts_API --> Cases_API
    Cases_API --> Analysts["Fraud Analysts / Investigators"]
    Analysts --> Feedback
    Feedback --> Engine["Model Retraining & Drift Monitor"]

    %% Flow: Outbound
    Alerts_API --> Webhooks
    Cases_API --> Webhooks

    style Inference fill:#f9f,stroke:#333,stroke-width:4px
    style Feedback fill:#bbf,stroke:#333,stroke-width:2px
    style Engine fill:#dfd,stroke:#333,stroke-width:2px
```

## The 7-API Cycle

1.  **Event Ingestion**: Platforms receive non-financial signals (logins, beneficiary adds) to build context.
2.  **Account Profile**: Syncs KYC and historical risk segments from the bank's core.
3.  **Transaction Scoring**: The critical real-time lookup for transaction authorization.
4.  **Alert API**: Programmatic access to detected threats for downstream SIEM/SOAR systems.
5.  **Case Management**: Investigator workflow for deep-dive forensics and SAR filing.
6.  **Decision Feedback**: The most vital link for "Closed Loop" ML—learning from real-world outcomes.
7.  **Webhooks**: Enables the bank to act immediately on critical alerts without polling.
