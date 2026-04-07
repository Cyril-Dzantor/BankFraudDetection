from sqlmodel import Session
from ..core.config import settings
from ..core.db import engine
from ..models.alert import Alert
from ..models.case import Case
from ..models.account import AccountProfile
from ..models.audit import AuditLog
from ..models.riskmodel import RiskModel, ModelMetrics
from ..models.user_provision import ProvisionedUser
from ..models.fairness import FairnessSnapshot
import uuid

# Mock Data
MOCK_ALERTS = [
    {
        "id": "TXN-99281", "branch": "Accra Central", "customer": "Kwame Mensah", "customer_id": "ACC-09283-GH", "acctStart": "Savings • 2018", "initials": "KM", "amount": "GH₵ 12,400.00", "riskLevel": "High", "score": 82, "channel": "Mobile App", "channelIconType": "smartphone", "time": "2 mins ago", "location": "Accra, GH", "device": "iPhone 13 Pro", "reason": "High Velocity Inbound", "triggered_rules": "velocity_limit_exceeded,anomaly_detected", "created_at": "2026-03-08T10:45:22"
    },
    {
        "id": "TXN-99282", "branch": "Kumasi Hub", "customer": "Acme Widgets Ltd", "customer_id": "ACC-09341-GH", "acctStart": "Corporate • 2015", "initials": "AW", "amount": "GH₵ 450,000.00", "riskLevel": "Critical", "score": 94, "channel": "Web Portal", "channelIconType": "globe", "time": "15 mins ago", "location": "Kumasi, GH", "device": "macOS Desktop", "reason": "Suspicious UBO Transfer", "triggered_rules": "large_amount_threshold,high_risk_destination", "created_at": "2026-03-08T10:32:15"
    }
]

MOCK_CASES = [
    {
        "id": "CAS-9921-GH", "title": "High Velocity Inbound Chain", "customer_id": "ACC-09283-GH", "assignee": "Kwame Mensah", "status": "In Investigation", "priority": "High", "value": "GH₵ 12,400.00", "type": "Retail", "tags": "Retail,Velocity,Urgent", "created": "Oct 22, 2023", "updated": "Oct 24, 2023", "notes": "Customer flagged for multiple rapid incoming transfers from unverified wallets."
    },
    {
        "id": "CAS-1004-GH", "title": "Suspicious UBO Transfer Correlation", "customer_id": "ACC-09341-GH", "assignee": "Me", "status": "In Investigation", "priority": "Critical", "value": "GH₵ 450,000.00", "type": "Corporate", "tags": "Corporate,KYC,HighValue", "created": "Nov 01, 2023", "updated": "Nov 02, 2023", "notes": "Corporate entity attempting large transfer to a newly flagged offshore account."
    }
]

MOCK_ACCOUNTS = [
    {
        "id": "ACC-09283-GH",
        "name": "Kwame Mensah",
        "kyc_level": "KYC Level 3",
        "location": "Accra, Ghana",
        "account_type": "Personal Savings Account",
        "risk_score": 0.82,
        "risk_level": "High Risk",
        "account_status": "UNDER_INVESTIGATION",
        "linked_cases": [
            { "id": "CAS-9921-GH", "type": "High Velocity Inbound", "date": "22 Oct, 2023", "status": "IN INVESTIGATION", "statusColor": "bg-orange-100 text-orange-700" },
            { "id": "CAS-8402-GH", "type": "New Device Link", "date": "15 Sep, 2023", "status": "RESOLVED - SAFE", "statusColor": "bg-emerald-100 text-emerald-700" },
        ],
        "behavior_data": [
            { "date": "OCT 18", "value": 30 }, { "date": "OCT 19", "value": 28 },
            { "date": "OCT 20", "value": 45 }, { "date": "OCT 21", "value": 65 },
            { "date": "OCT 22", "value": 40 }, { "date": "OCT 23", "value": 35 },
            { "date": "TODAY", "value": 85 },
        ],
        "feature_importance": [
            { "subject": "Geo-Velocity", "A": 120, "B": 110, "fullMark": 150 },
            { "subject": "Amt. Delta", "A": 98, "B": 130, "fullMark": 150 },
            { "subject": "IP Reputation", "A": 86, "B": 130, "fullMark": 150 },
            { "subject": "Login Time", "A": 99, "B": 100, "fullMark": 150 },
            { "subject": "Device Age", "A": 85, "B": 90, "fullMark": 150 },
            { "subject": "UBO Link", "A": 65, "B": 85, "fullMark": 150 },
        ]
    },
    {
        "id": "ACC-09341-GH",
        "name": "Acme Widgets Ltd",
        "kyc_level": "Corporate Enhanced",
        "location": "Kumasi, Ghana",
        "account_type": "Corporate Operating",
        "risk_score": 0.94,
        "risk_level": "Critical",
        "account_status": "FROZEN",
        "linked_cases": [
            { "id": "CAS-1004-GH", "type": "Suspicious UBO Transfer", "date": "01 Nov, 2023", "status": "ESCALATED", "statusColor": "bg-red-100 text-red-700" },
        ],
        "behavior_data": [
            { "date": "OCT 18", "value": 10 }, { "date": "OCT 19", "value": 12 },
            { "date": "OCT 20", "value": 15 }, { "date": "OCT 21", "value": 20 },
            { "date": "OCT 22", "value": 90 }, { "date": "OCT 23", "value": 110 },
            { "date": "TODAY", "value": 150 },
        ],
        "feature_importance": [
            { "subject": "Geo-Velocity", "A": 140, "B": 80, "fullMark": 150 },
            { "subject": "Amt. Delta", "A": 145, "B": 90, "fullMark": 150 },
            { "subject": "IP Reputation", "A": 130, "B": 100, "fullMark": 150 },
            { "subject": "Login Time", "A": 110, "B": 100, "fullMark": 150 },
            { "subject": "Device Age", "A": 120, "B": 110, "fullMark": 150 },
            { "subject": "UBO Link", "A": 145, "B": 70, "fullMark": 150 },
        ]
    },
    {
        "id": "ACC-08812-GH",
        "name": "Abena Osei",
        "kyc_level": "KYC Level 2",
        "location": "Takoradi, Ghana",
        "account_type": "Student Account",
        "risk_score": 0.15,
        "risk_level": "Low Risk",
        "account_status": "ACTIVE",
        "linked_cases": [],
        "behavior_data": [
            { "date": "OCT 18", "value": 12 }, { "date": "OCT 19", "value": 14 },
            { "date": "OCT 20", "value": 13 }, { "date": "OCT 21", "value": 15 },
            { "date": "OCT 22", "value": 14 }, { "date": "OCT 23", "value": 12 },
            { "date": "TODAY", "value": 15 },
        ],
        "feature_importance": [
            { "subject": "Geo-Velocity", "A": 40, "B": 100, "fullMark": 150 },
            { "subject": "Amt. Delta", "A": 30, "B": 100, "fullMark": 150 },
            { "subject": "IP Reputation", "A": 20, "B": 100, "fullMark": 150 },
            { "subject": "Login Time", "A": 80, "B": 100, "fullMark": 150 },
            { "subject": "Device Age", "A": 50, "B": 100, "fullMark": 150 },
            { "subject": "UBO Link", "A": 10, "B": 100, "fullMark": 150 },
        ]
    },
    {
        "id": "ACC-09115-GH",
        "name": "Samuel Tetteh",
        "kyc_level": "KYC Level 3",
        "location": "Tema, Ghana",
        "account_type": "Premium Current Account",
        "risk_score": 0.45,
        "risk_level": "Medium Risk",
        "account_status": "ACTIVE",
        "linked_cases": [
            { "id": "CAS-7732-GH", "type": "Unusual Location Access", "date": "05 Oct, 2023", "status": "RESOLVED - SAFE", "statusColor": "bg-emerald-100 text-emerald-700" },
        ],
        "behavior_data": [
            { "date": "OCT 18", "value": 45 }, { "date": "OCT 19", "value": 40 },
            { "date": "OCT 20", "value": 50 }, { "date": "OCT 21", "value": 55 },
            { "date": "OCT 22", "value": 45 }, { "date": "OCT 23", "value": 60 },
            { "date": "TODAY", "value": 70 },
        ],
        "feature_importance": [
            { "subject": "Geo-Velocity", "A": 85, "B": 90, "fullMark": 150 },
            { "subject": "Amt. Delta", "A": 75, "B": 100, "fullMark": 150 },
            { "subject": "IP Reputation", "A": 65, "B": 110, "fullMark": 150 },
            { "subject": "Login Time", "A": 120, "B": 100, "fullMark": 150 },
            { "subject": "Device Age", "A": 45, "B": 100, "fullMark": 150 },
            { "subject": "UBO Link", "A": 35, "B": 95, "fullMark": 150 },
        ]
    }
]

MOCK_AUDIT_LOGS = [
    { "id": "LOG-88291", "timestamp": "2026-03-08 10:45:22", "actor": "system_admin_01", "action": "POLICY_UPDATE", "resource": "RuleEngine.Thresholds", "status": "Success", "risk": "Low", "ip": "10.2.44.11" },
    { "id": "LOG-88290", "timestamp": "2026-03-08 10:42:15", "actor": "k.mensah@bank.com.gh", "action": "DATA_EXPORT", "resource": "Customer.RiskProfiles.CSV", "status": "Success", "risk": "Medium", "ip": "192.168.1.105" },
    { "id": "LOG-88289", "timestamp": "2026-03-08 10:35:01", "actor": "unknown_external", "action": "API_AUTH_FAILURE", "resource": "Gateway.OAuth2", "status": "Failed", "risk": "High", "ip": "45.22.19.88" },
    { "id": "LOG-88288", "timestamp": "2026-03-08 10:15:33", "actor": "a.asante@bank.com.gh", "action": "CASE_RESOLUTION", "resource": "Case.CAS-2023-8989", "status": "Success", "risk": "Low", "ip": "192.168.1.112" },
    { "id": "LOG-88287", "timestamp": "2026-03-08 09:55:10", "actor": "db_service_account", "action": "BULK_READ", "resource": "Table.Transactions_Q3", "status": "Success", "risk": "Medium", "ip": "10.2.44.05" },
    { "id": "LOG-88286", "timestamp": "2026-03-08 09:40:05", "actor": "e.yeboah@bank.com.gh", "action": "PRIVILEGE_ESCALATION_ATTEMPT", "resource": "Role.SuperAdmin", "status": "Blocked", "risk": "Critical", "ip": "192.168.2.55" },
]

MOCK_RISK_MODELS = [
    { "id": "MDL-XGB-001", "name": "XGBoost Classifier", "status": "Healthy", "accuracy": "98.7%", "precision": "97.2%", "recall": "96.2%", "latency": "42ms", "last_trained": "2 days ago", "type": "Supervised" },
    { "id": "MDL-IF-001", "name": "Isolation Forest", "status": "Healthy", "accuracy": "95.4%", "precision": "92.5%", "recall": "91.8%", "latency": "28ms", "last_trained": "12 hours ago", "type": "Anomaly Detection" },
    { "id": "MDL-AE-001", "name": "Lightweight Autoencoder", "status": "Healthy", "accuracy": "94.1%", "precision": "90.8%", "recall": "89.5%", "latency": "15ms", "last_trained": "5 days ago", "type": "Unsupervised" },
]

MOCK_MODEL_METRICS = {
    "MDL-XGB-001": {
        "performance_data": [
            { "time": "00:00", "xgb": 98.7, "iso": 95.4, "ae": 94.1 },
            { "time": "04:00", "xgb": 98.8, "iso": 95.6, "ae": 94.5 },
            { "time": "08:00", "xgb": 98.5, "iso": 95.2, "ae": 94.0 },
            { "time": "12:00", "xgb": 98.6, "iso": 94.8, "ae": 93.5 },
            { "time": "16:00", "xgb": 98.7, "iso": 95.7, "ae": 94.2 },
            { "time": "20:00", "xgb": 98.9, "iso": 95.9, "ae": 94.8 },
            { "time": "24:00", "xgb": 99.0, "iso": 96.0, "ae": 95.1 },
        ],
        "drift_data": [
            { "day": "Mon", "score": 0.02 }, { "day": "Tue", "score": 0.03 },
            { "day": "Wed", "score": 0.05 }, { "day": "Thu", "score": 0.12 },
            { "day": "Fri", "score": 0.15 }, { "day": "Sat", "score": 0.08 },
            { "day": "Sun", "score": 0.04 },
        ],
        "distribution_shift": [
            { "feature": "Income", "train": 0.45, "current": 0.62 },
            { "feature": "Age", "train": 0.38, "current": 0.39 },
            { "feature": "Geo", "train": 0.55, "current": 0.58 },
            { "feature": "Velocity", "train": 0.22, "current": 0.45 },
            { "feature": "Amt", "train": 0.61, "current": 0.59 },
        ],
    },
    "MDL-IF-001": {
        "performance_data": [
            { "time": "00:00", "xgb": 98.7, "iso": 95.4, "ae": 94.1 },
            { "time": "04:00", "xgb": 98.8, "iso": 95.6, "ae": 94.5 },
            { "time": "08:00", "xgb": 98.5, "iso": 95.2, "ae": 94.0 },
            { "time": "12:00", "xgb": 98.6, "iso": 94.8, "ae": 93.5 },
            { "time": "16:00", "xgb": 98.7, "iso": 95.7, "ae": 94.2 },
            { "time": "20:00", "xgb": 98.9, "iso": 95.9, "ae": 94.8 },
            { "time": "24:00", "xgb": 99.0, "iso": 96.0, "ae": 95.1 },
        ],
        "drift_data": [
            { "day": "Mon", "score": 0.02 }, { "day": "Tue", "score": 0.03 },
            { "day": "Wed", "score": 0.05 }, { "day": "Thu", "score": 0.12 },
            { "day": "Fri", "score": 0.15 }, { "day": "Sat", "score": 0.08 },
            { "day": "Sun", "score": 0.04 },
        ],
        "distribution_shift": [
            { "feature": "Income", "train": 0.45, "current": 0.62 },
            { "feature": "Age", "train": 0.38, "current": 0.39 },
            { "feature": "Geo", "train": 0.55, "current": 0.58 },
            { "feature": "Velocity", "train": 0.22, "current": 0.45 },
            { "feature": "Amt", "train": 0.61, "current": 0.59 },
        ],
    },
    "MDL-AE-001": {
        "performance_data": [
            { "time": "00:00", "xgb": 98.7, "iso": 95.4, "ae": 94.1 },
            { "time": "04:00", "xgb": 98.8, "iso": 95.6, "ae": 94.5 },
            { "time": "08:00", "xgb": 98.5, "iso": 95.2, "ae": 94.0 },
            { "time": "12:00", "xgb": 98.6, "iso": 94.8, "ae": 93.5 },
            { "time": "16:00", "xgb": 98.7, "iso": 95.7, "ae": 94.2 },
            { "time": "20:00", "xgb": 98.9, "iso": 95.9, "ae": 94.8 },
            { "time": "24:00", "xgb": 99.0, "iso": 96.0, "ae": 95.1 },
        ],
        "drift_data": [
            { "day": "Mon", "score": 0.02 }, { "day": "Tue", "score": 0.03 },
            { "day": "Wed", "score": 0.05 }, { "day": "Thu", "score": 0.12 },
            { "day": "Fri", "score": 0.15 }, { "day": "Sat", "score": 0.08 },
            { "day": "Sun", "score": 0.04 },
        ],
        "distribution_shift": [
            { "feature": "Income", "train": 0.45, "current": 0.62 },
            { "feature": "Age", "train": 0.38, "current": 0.39 },
            { "feature": "Geo", "train": 0.55, "current": 0.58 },
            { "feature": "Velocity", "train": 0.22, "current": 0.45 },
            { "feature": "Amt", "train": 0.61, "current": 0.59 },
        ],
    }
}

MOCK_USERS = [
    { "id": "USR-001-ADMIN", "full_name": "Felix Yeboah", "email": "f.yeboah@bank.com.gh", "employee_id": "EMP-0022", "department": "IT Security", "role": "system_admin", "status": "APPROVED", "password": "fraud-sense-2026" },
    { "id": "USR-002-CHECKER", "full_name": "Ama Asante", "email": "a.asante@bank.com.gh", "employee_id": "EMP-0017", "department": "Compliance", "role": "compliance_lead", "status": "APPROVED", "password": "fraud-sense-2026" },
    { "id": "USR-003-SENIOR", "full_name": "Kwame Mensah", "email": "k.mensah@bank.com.gh", "employee_id": "EMP-0012", "department": "Fraud Intelligence", "role": "senior_analyst", "status": "APPROVED", "password": "fraud-sense-2026" },
    { "id": "USR-004-JUNIOR", "full_name": "Mia Boateng", "email": "m.boateng@bank.com.gh", "employee_id": "EMP-0025", "department": "Fraud Intelligence", "role": "junior_analyst", "status": "APPROVED", "password": "fraud-sense-2026" },
]

def seed_db():
    # ── Pre-seed schema migration ────────────────────────────────────────────────
    # Some columns were added to models after the initial table creation.
    # We patch the live schema with raw SQL before any ORM query.
    import sqlite3 as _sqlite3
    _db_path = settings.SQLALCHEMY_DATABASE_URI.replace("sqlite:///", "").replace("sqlite:////", "/")
    _migrations = [
        ("alert", "customer_id", "TEXT"),
        ("alert", "account_status", "TEXT"),
        ("alert", "reason", "TEXT DEFAULT ''"),
        ("alert", "triggered_rules", "TEXT DEFAULT ''"),
        ("alert", "transaction_type", "TEXT"),
        ("alert", "recipient_account", "TEXT"),
        ("alert", "recipient_name", "TEXT"),
        ("alert", "transaction_notes", "TEXT"),
        ("case", "customer_id", "TEXT"),
        ("case", "notes", "TEXT"),
        ("accountprofile", "initials", "TEXT"),
        ("accountprofile", "account_status", "TEXT"),
        ("accountprofile", "linked_cases", "TEXT"),
        ("accountprofile", "behavior_data", "TEXT"),
        ("accountprofile", "feature_importance", "TEXT"),
    ]
    try:
        _conn = _sqlite3.connect(_db_path)
        for _tbl, _col, _typ in _migrations:
            try:
                _conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN {_col} {_typ}")
            except _sqlite3.OperationalError:
                pass  # already exists
        _conn.commit()
        _conn.close()
    except Exception as _e:
        print(f"[seed migration] {_e}")
    # ────────────────────────────────────────────────────────────────────────────

    with Session(engine) as session:
        for alert_data in MOCK_ALERTS:
            session.merge(Alert(**alert_data))
        
        for case_data in MOCK_CASES:
            session.merge(Case(**case_data))

        for account_data in MOCK_ACCOUNTS:
            # Add initials for the UI if not present
            if "initials" not in account_data:
                names = account_data["name"].split()
                account_data["initials"] = "".join([n[0] for n in names])
            session.merge(AccountProfile(**account_data))

        for log_data in MOCK_AUDIT_LOGS:
            existing_log = session.get(AuditLog, log_data["id"])
            if not existing_log:
                session.add(AuditLog(**log_data))

        for model_data in MOCK_RISK_MODELS:
            existing_model = session.get(RiskModel, model_data["id"])
            if not existing_model:
                session.add(RiskModel(**model_data))
                if model_data["id"] in MOCK_MODEL_METRICS:
                    metrics = MOCK_MODEL_METRICS[model_data["id"]]
                    session.add(ModelMetrics(model_id=model_data["id"], **metrics))

        for user_data in MOCK_USERS:
            session.merge(ProvisionedUser(**user_data))

        from sqlmodel import select
        existing_fairness = session.exec(select(FairnessSnapshot)).first()
        if not existing_fairness:
            session.add(FairnessSnapshot(
                audited_decisions=1240000,
                approval_rate=88.4,
                fairness_score=98.2,
                pending_reviews=45,
                gender_data=[
                    { "group": "Male", "approval_rate": 89.1, "block_rate": 10.9 },
                    { "group": "Female", "approval_rate": 87.8, "block_rate": 12.2 },
                    { "group": "Non-Binary", "approval_rate": 86.5, "block_rate": 13.5 },
                ],
                age_data=[
                    { "group": "18-25", "approval_rate": 82.0, "block_rate": 18.0 },
                    { "group": "26-35", "approval_rate": 90.1, "block_rate": 9.9 },
                    { "group": "36-50", "approval_rate": 91.4, "block_rate": 8.6 },
                    { "group": "51-65", "approval_rate": 89.2, "block_rate": 10.8 },
                    { "group": "65+", "approval_rate": 85.3, "block_rate": 14.7 },
                ],
                region_data=[
                    { "region": "Greater Accra", "block_deviation": 1.2, "status": "ok" },
                    { "region": "Ashanti", "block_deviation": 2.8, "status": "warning" },
                    { "region": "Western", "block_deviation": 1.5, "status": "ok" },
                    { "region": "Central", "block_deviation": 0.9, "status": "ok" },
                    { "region": "Eastern", "block_deviation": 3.2, "status": "warning" },
                    { "region": "Volta", "block_deviation": 1.8, "status": "ok" },
                    { "region": "Northern", "block_deviation": 6.7, "status": "critical" },
                    { "region": "Upper East", "block_deviation": 5.4, "status": "critical" },
                    { "region": "Upper West", "block_deviation": 4.9, "status": "warning" },
                    { "region": "Brong-Ahafo", "block_deviation": 2.1, "status": "warning" },
                    { "region": "Savannah", "block_deviation": 7.2, "status": "critical" },
                    { "region": "Bono East", "block_deviation": 3.4, "status": "warning" },
                    { "region": "Ahafo", "block_deviation": 1.6, "status": "ok" },
                    { "region": "Oti", "block_deviation": 2.5, "status": "warning" },
                    { "region": "North East", "block_deviation": 5.9, "status": "critical" },
                    { "region": "Western North", "block_deviation": 1.1, "status": "ok" },
                ],
            ))

        session.commit()

if __name__ == "__main__":
    seed_db()
