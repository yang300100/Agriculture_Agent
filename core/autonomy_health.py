"""哥哥，维护成功与异常分开持久化，不把进程心跳当成维护健康。"""
import os
from datetime import datetime

from core.database.engine import get_session
from core.database.models import SchedulerHeartbeat


def record_maintenance_result(success: bool, detail: str = '') -> None:
    now = datetime.now()
    row_id = 'maintenance_success' if success else 'maintenance_failure'
    with get_session() as session:
        row = session.get(SchedulerHeartbeat, row_id)
        if row is None:
            row = SchedulerHeartbeat(id=row_id)
            session.add(row)
        row.status = 'success' if success else 'failure'
        row.detail = str(detail)[:2000]
        row.last_seen_at = now
        row.process_id = os.getpid()
        session.commit()


def maintenance_status() -> dict:
    with get_session() as session:
        success = session.get(SchedulerHeartbeat, 'maintenance_success')
        failure = session.get(SchedulerHeartbeat, 'maintenance_failure')
        success_at = success.last_seen_at if success else None
        failure_at = failure.last_seen_at if failure else None
        age = (datetime.now() - success_at).total_seconds() if success_at else None
        healthy = bool(age is not None and 0 <= age <= 60 and (failure_at is None or failure_at <= success_at))
        return {
            'last_maintenance_success': success_at.isoformat() if success_at else None,
            'last_maintenance_failure': failure_at.isoformat() if failure_at else None,
            'maintenance_error': failure.detail if failure else None,
            'maintenance_healthy': healthy,
        }
