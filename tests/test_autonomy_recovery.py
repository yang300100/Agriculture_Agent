"""哥哥，重启和并发不能把待核对的作业误当作已完成。"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from core import autonomy_runtime as runtime
from core.autonomy_store import claim_jobs, claim_alerts, enqueue, finish_job


def test_待安排任务可指定时间且用户不能改别人的任务():
    username = 'recovery_' + uuid4().hex
    job = runtime.schedule_action(username, '1', {'device_id': 'pump', 'action': 'irrigate'})
    due = datetime.now() + timedelta(minutes=5)
    with pytest.raises(LookupError):
        runtime.arrange_job('other', job['id'], due)
    with pytest.raises(ValueError):
        runtime.arrange_job(username, job['id'], datetime.now() - timedelta(seconds=5))
    saved = runtime.arrange_job(username, job['id'], due)
    assert saved['status'] == 'pending' and saved['due_at'] == due.isoformat()
    assert saved['payload']['action']['execution_id'] == job['payload']['action']['execution_id']
    with pytest.raises(ValueError):
        runtime.arrange_job(username, job['id'], due)


def test_并发领取只有一方成功且旧租约不能覆盖恢复后的结果():
    username = 'recovery_' + uuid4().hex
    now = datetime.now()
    enqueue(username, '1', 'effect_check', {}, due_at=now)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: claim_jobs(now, 1, username), range(2)))
    assert sum(map(len, claims)) == 1
    old = next(rows[0] for rows in claims if rows)
    newer = claim_jobs(now + timedelta(minutes=3), 1, username)[0]
    finish_job(newer, {'success': True}, now + timedelta(minutes=3))
    finish_job(old, {'success': False, 'terminal': True}, now + timedelta(minutes=4))
    assert runtime.list_jobs(username)[0]['status'] == 'completed'


def test_下发中断后的计划只能待核验不能在重启后重发():
    username = 'recovery_' + uuid4().hex
    now = datetime.now()
    runtime.schedule_action(username, '1', {'device_id': 'pump'}, due_at=now)
    assert len(claim_jobs(now, 1, username)) == 1
    runtime.process_due_jobs(username=username, now=now + timedelta(minutes=3),
        handlers={'planned_action': lambda job: pytest.fail('中断后的作业不得重新下发')})
    assert runtime.list_jobs(username)[0]['status'] == 'unverified'
    assert len(runtime.list_alerts(username)) == 1


def test_依赖仍在运行时延后而非消耗执行重试(monkeypatch):
    username = 'recovery_' + uuid4().hex
    now = datetime.now()
    runtime.schedule_action(username, '1', {'device_id': 'fert', 'dependency_runs': [
        {'device_id': 'pump', 'execution_id': 'run-first'}]}, due_at=now)
    monkeypatch.setattr(runtime, '_check_dependencies', lambda job: {
        'success': False, 'defer_until': (now + timedelta(seconds=30)).isoformat(),
        'error': '前置作业尚在运行'}, raising=False)
    monkeypatch.setattr(runtime, '_run_planned_action', lambda job: pytest.fail('依赖未完成不能执行'))
    runtime.process_due_jobs(username=username, now=now)
    saved = runtime.list_jobs(username)[0]
    assert saved['status'] == 'pending' and saved['attempts'] == 0


def test_通知通道返回成功HTTP但明确拒绝时仍是失败(monkeypatch):
    monkeypatch.setenv('AUTONOMY_ALERT_WEBHOOK', 'http://127.0.0.1:1234/alert')
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self, *args): return json.dumps({'success': False, 'error': '额度不足'}).encode()
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda *a, **kw: Response())
    alert = runtime.raise_alert('recovery_' + uuid4().hex, '1', '灌溉设备异常')
    result = runtime._send_alert(alert)
    assert result['success'] is False and '拒绝' in result['error']


def test_依赖完成必须有同地块新鲜且匹配编号的完成记录(monkeypatch):
    import asyncio
    from contextlib import contextmanager
    from core import device_registry_factory as factory
    state = {'_quality': 'good', '_observed_at': datetime.now().isoformat(),
             'status': 'standby', 'run_history': {'old-run': {'reason': 'completed'}}}
    class Registry:
        async def discover_all(self): return []
        async def read_state(self, device_id): return state
    @contextmanager
    def session(username):
        loop = asyncio.new_event_loop()
        try: yield Registry(), loop
        finally: loop.close()
    monkeypatch.setattr(factory, 'RegistrySession', session)
    configs = [{'device_id': 'pump', 'plot_id': '1'}]
    monkeypatch.setattr(factory, 'load_custom_devices', lambda username: configs)
    job = {'username': 'alice', 'region': '1', 'created_at': datetime.now().isoformat(),
           'payload': {'action': {'dependency_runs': [{'device_id': 'pump', 'execution_id': 'first'}]}}}
    assert runtime._check_dependencies(job)['status'] == 'unverified'
    state['run_history']['first'] = {'reason': 'completed'}
    assert runtime._check_dependencies(job) is None
    configs[0]['plot_id'] = '2'
    assert runtime._check_dependencies(job)['status'] == 'unverified'
    configs[0]['plot_id'] = '1'
    state['_observed_at'] = (datetime.now() - timedelta(hours=1)).isoformat()
    assert runtime._check_dependencies(job)['status'] == 'unverified'


def test_排期接口校验归属与未来时间且不会立即下发():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api_routes import register_routes
    username = 'recovery_' + uuid4().hex
    job = runtime.schedule_action(username, '1', {'action': 'irrigate', 'device_id': 'pump'})
    due = (datetime.now() + timedelta(minutes=10)).isoformat()
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        endpoint = f"/api/autonomous/jobs/{job['id']}/schedule"
        assert client.post(endpoint + '?username=other', json={'due_at': due}).status_code == 404
        assert client.post(endpoint + f'?username={username}', json={'due_at': '错误时间'}).status_code == 422
        assert client.post(endpoint + f'?username={username}', json={'due_at': due, 'device_id': 'other'}).status_code == 422
        response = client.post(endpoint + f'?username={username}', json={'due_at': due})
        assert response.status_code == 200 and response.json()['job']['status'] == 'pending'
        assert runtime.list_jobs(username)[0]['attempts'] == 0


def test_另一进程已设通知退避时旧候选不能立即抢发():
    from sqlalchemy import event
    from core.database.engine import _engine
    from core.database.models import AutonomyAlert
    username = 'recovery_' + uuid4().hex
    alert = runtime.raise_alert(username, '1', '设备离线')
    now = datetime.now() + timedelta(seconds=1)
    intercepted = []
    def concurrent_retry(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith('UPDATE autonomy_alerts') and not intercepted:
            intercepted.append(True)
            with _engine.begin() as competing:
                competing.execute(AutonomyAlert.__table__.update().where(AutonomyAlert.id == alert['id']).values(
                    next_attempt_at=now + timedelta(minutes=5)))
    event.listen(_engine, 'before_cursor_execute', concurrent_retry)
    try:
        assert claim_alerts(now, username=username) == []
    finally:
        event.remove(_engine, 'before_cursor_execute', concurrent_retry)
    assert intercepted


def test_排期规则到期时条件已不满足则不再启动(monkeypatch):
    from core.device_registry_factory import load_custom_devices, save_custom_devices
    from core.device_rule_engine import RuleEngine
    from core.autonomous_farm_manager import AutonomousFarmManager
    monkeypatch.setattr(AutonomousFarmManager, '_collect_weather', lambda *a, **kw: (None, []))
    username = 'recovery_' + uuid4().hex
    monkeypatch.setenv('AUTONOMY_LEVEL', 'high')
    save_custom_devices(username, [{'device_id': 'pump', 'name': '麦田水泵',
        'driver': 'simulator', 'plot_id': '1', 'capabilities': ['irrigate'],
        'initial_state': {'power': True, 'status': 'standby'}}])
    rule_id = RuleEngine(username).add_rule({'name': '干旱灌溉', 'enabled': True,
        'trigger': {'conditions': [{'type': 'sensor', 'field': 'soil_moisture', 'op': '<', 'value': 0}]},
        'action': {'device_id': 'pump', 'capability': 'irrigate', 'command': 'start', 'params': {'duration': 1}},
        'execution_mode': 'auto', 'constraints': {}})
    now = datetime.now()
    job = runtime.schedule_action(username, '1', {'action': 'irrigate', 'device_id': 'pump',
        'params': {'duration': 1}, 'rule_id': rule_id}, due_at=now)
    runtime.process_due_jobs(username=username, now=now)
    assert next(row for row in runtime.list_jobs(username) if row['id'] == job['id'])['status'] == 'failed'
    assert load_custom_devices(username)[0]['initial_state']['status'] == 'standby'
