"""哥哥，维护健康只在隔离数据库和替身处理器里核验。"""
from datetime import datetime, timedelta


def test_maintenance_health_tracks_success_and_newer_failure():
    from core.database.engine import init_db, get_session
    from core.database.models import SchedulerHeartbeat
    from core.autonomy_health import record_maintenance_result, maintenance_status
    init_db()
    with get_session() as session:
        session.query(SchedulerHeartbeat).filter(SchedulerHeartbeat.id.in_(['maintenance_success', 'maintenance_failure'])).delete(synchronize_session=False)
        session.commit()
    initial = maintenance_status()
    assert initial['last_maintenance_success'] is None and not initial['maintenance_healthy']
    record_maintenance_result(True, '成功')
    assert maintenance_status()['maintenance_healthy'] is True
    record_maintenance_result(False, '数据库离线')
    failed = maintenance_status()
    assert not failed['maintenance_healthy'] and failed['maintenance_error'] == '数据库离线'
    record_maintenance_result(True, '恢复')
    assert maintenance_status()['maintenance_healthy'] is True
    with get_session() as session:
        success = session.get(SchedulerHeartbeat, 'maintenance_success')
        success.last_seen_at = datetime.now() - timedelta(minutes=2)
        failure = session.get(SchedulerHeartbeat, 'maintenance_failure')
        failure.last_seen_at = success.last_seen_at - timedelta(seconds=1)
        session.commit()
    assert not maintenance_status()['maintenance_healthy']


def test_maintenance_job_records_pipeline_outcome(monkeypatch):
    from core import scheduler_service, autonomy_runtime, autonomy_health
    outcomes = []
    monkeypatch.setattr(autonomy_health, 'record_maintenance_result', lambda success, detail='': outcomes.append((success, detail)))
    monkeypatch.setattr(autonomy_runtime, 'process_due_jobs', lambda: [])
    monkeypatch.setattr(autonomy_runtime, 'dispatch_alerts', lambda: [])
    scheduler_service._autonomy_maintenance()
    assert outcomes[-1][0] is True
    def broken(): raise RuntimeError('管道故障')
    monkeypatch.setattr(autonomy_runtime, 'process_due_jobs', broken)
    scheduler_service._autonomy_maintenance()
    assert outcomes[-1][0] is False and '管道故障' in outcomes[-1][1]
