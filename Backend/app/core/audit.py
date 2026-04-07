import hashlib
import json
import uuid
from sqlmodel import Session, select
from datetime import datetime
from app.models.audit import AuditLog

class AuditManager:
    @staticmethod
    def log_action(session: Session, actor: str, action: str, resource: str, status: str, risk: str, ip: str, payload: dict = None):
        log_id = f"LOG-{str(uuid.uuid4())[:8].upper()}"
        timestamp = datetime.utcnow().isoformat()
        
        # Determine previous hash
        last_log_stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(1)
        last_log = session.exec(last_log_stmt).first()
        
        previous_hash = None
        if last_log:
            last_log_rep = f"{last_log.id}|{last_log.timestamp}|{last_log.actor}|{last_log.action}|{last_log.resource}|{last_log.status}|{last_log.payload}|{last_log.previous_hash}"
            previous_hash = hashlib.sha256(last_log_rep.encode()).hexdigest()

        payload_str = json.dumps(payload) if payload else None

        db_log = AuditLog(
            id=log_id,
            timestamp=timestamp,
            actor=actor,
            action=action,
            resource=resource,
            status=status,
            risk=risk,
            ip=ip,
            payload=payload_str,
            previous_hash=previous_hash
        )
        
        session.add(db_log)
        session.commit()
        session.refresh(db_log)
        return db_log

audit_manager = AuditManager()
