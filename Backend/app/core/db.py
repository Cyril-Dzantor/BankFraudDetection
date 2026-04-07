import sqlite3
import os
from sqlmodel import create_engine, Session
from .config import settings
from sqlalchemy import event

# ── Raw SQLite migration – runs BEFORE the ORM engine is created ──────────────
# This is critical: SQLAlchemy compiles queries at engine-init time.
# If the DB file exists but is missing new columns, the ORM will crash
# on the first query. We patch the schema with raw sqlite3 first.
_DB_PATH = settings.SQLALCHEMY_DATABASE_URI.replace("sqlite:///", "").replace("sqlite:////", "/")

_MIGRATIONS = [
    # Alert enrichments
    ("alert", "customer_id",         "TEXT"),
    ("alert", "account_status",      "TEXT"),
    ("alert", "reason",              "TEXT DEFAULT ''"),
    ("alert", "triggered_rules",     "TEXT DEFAULT ''"),
    ("alert", "transaction_type",    "TEXT"),
    ("alert", "recipient_account",   "TEXT"),
    ("alert", "recipient_name",      "TEXT"),
    ("alert", "transaction_notes",   "TEXT"),
    # Case enrichments
    ("case",  "customer_id",         "TEXT"),
    ("case",  "notes",               "TEXT"),
    # Account profile enrichments
    ("accountprofile", "initials",          "TEXT"),
    ("accountprofile", "account_status",    "TEXT"),
    ("accountprofile", "linked_cases",      "TEXT"),
    ("accountprofile", "behavior_data",     "TEXT"),
    ("accountprofile", "feature_importance","TEXT"),
]

def _run_migrations():
    """Add missing columns to an existing SQLite database before the ORM starts."""
    if not os.path.exists(_DB_PATH):
        return  # Fresh DB — create_all will build the correct schema from scratch

    try:
        conn = sqlite3.connect(_DB_PATH)
        cur  = conn.cursor()
        for table, column, col_type in _MIGRATIONS:
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                print(f"[migration] Added column '{column}' to '{table}'")
            except sqlite3.OperationalError:
                pass  # Column already exists — safe to ignore
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[migration] Warning: {exc}")

_run_migrations()
# ─────────────────────────────────────────────────────────────────────────────

engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, echo=False)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

def get_session():
    with Session(engine) as session:
        yield session
