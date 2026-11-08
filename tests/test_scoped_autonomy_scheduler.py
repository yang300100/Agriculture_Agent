"""哥哥，调度与区域观测不接触正式设备。"""
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from app import scheduler_jobs
from core import scheduler_service


def test_scheduler_installs_runtime_maintenance_jobs():
    scheduler = scheduler_service.create_scheduler()
    jobs = {j.id: j for j in scheduler.get_jobs()}
    assert jobs['autonomy_heartbeat'].trigger.interval.total_seconds() == 20
    assert jobs['autonomy_maintenance'].trigger.interval.total_seconds() == 5


def test_rule_context_rejects_stale_sensors_and_uses_plot_weather(monkeypatch):
    from core import device_registry_factory
    monkeypatch.setattr(device_registry_factory, 'load_custom_devices', lambda _: [
        {'device_id': 'fresh', 'plot_id': '1'}, {'device_id': 'stale', 'plot_id': '1'}])
    monkeypatch.setattr(scheduler_jobs, '_weather_by_plot', lambda _: {'1': {'temperature': 24}} , raising=False)
    now = datetime.now()
    class Registry:
        async def discover_all(self): return [SimpleNamespace(device_id='fresh'), SimpleNamespace(device_id='stale')]
        async def read_state(self, device_id):
            stamp = now if device_id == 'fresh' else now - timedelta(hours=1)
            return {'_quality': 'good', '_observed_at': stamp.isoformat(), 'soil_moisture': 12}
    loop = asyncio.new_event_loop()
    try: context = scheduler_jobs._collect_device_rule_context('alice', Registry(), loop)
    finally: loop.close()
    assert context['device_states']['fresh']['soil_moisture'] == 12
    assert 'soil_moisture' not in context['device_states']['stale']
    assert context['weather_by_plot']['1']['temperature'] == 24

def test_disease_job_is_scoped_and_clears_old_results(tmp_path, monkeypatch):
    import json
    from core import plot_manager, disease_risk
    monkeypatch.setattr(scheduler_jobs, 'DEFAULT_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(scheduler_jobs, '_disease_usernames', lambda: ['alice', 'bob'], raising=False)
    monkeypatch.setattr(scheduler_jobs, '_weather_by_plot', lambda username: {'1': {'temperature': 25, 'humidity': 90}} if username == 'alice' else {'2': {'temperature': 17, 'humidity': 50}}, raising=False)
    monkeypatch.setattr(plot_manager, 'PlotManager', lambda username: SimpleNamespace(list_plots=lambda: [{'plot_id': '1' if username == 'alice' else '2', 'crop': '番茄'}]))
    monkeypatch.setattr(disease_risk, 'assess_disease_risk', lambda crop, **kw: [{'risk': '高'}] if kw['humidity'] == 90 else [])
    old = tmp_path / 'bob' / 'disease_risks' / '2.json'; old.parent.mkdir(parents=True)
    old.write_text('{"risks":[{"risk":"旧"}]}', encoding='utf8')
    scheduler_jobs.check_disease_job()
    alice = json.loads((tmp_path / 'alice' / 'disease_risks' / '1.json').read_text(encoding='utf8'))
    bob = json.loads(old.read_text(encoding='utf8'))
    assert alice['plot_id'] == '1' and alice['risks'] == [{'risk': '高'}]
    assert bob['plot_id'] == '2' and bob['risks'] == [] and bob['updated']

def test_status_and_alert_endpoints_reflect_runtime_and_scope(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api_routes import register_routes
    from core import autonomy_runtime
    monkeypatch.setenv('ENABLE_SCHEDULER', 'true')
    monkeypatch.setattr(autonomy_runtime, 'runtime_status', lambda: {'running': False, 'heartbeat_status': 'stale', 'last_heartbeat': 'old'})
    monkeypatch.setattr(autonomy_runtime, 'list_alerts', lambda username: [{'username': username, 'id': 3}])
    monkeypatch.setattr(autonomy_runtime, 'list_jobs', lambda username: [{'username': username, 'id': 4}])
    monkeypatch.setattr(autonomy_runtime, 'acknowledge_alert', lambda username, ident: username == 'alice' and ident == 3)
    app = FastAPI(); register_routes(app)
    with TestClient(app) as client:
        status = client.get('/api/autonomous/status').json()
        assert status['configured'] and not status['running'] and not status['enabled'] and status['heartbeat_status'] == 'stale'
        assert client.get('/api/autonomous/alerts?username=alice').json()[0]['username'] == 'alice'
        assert client.get('/api/autonomous/jobs?username=bob').json()[0]['username'] == 'bob'
        assert client.post('/api/autonomous/alerts/3/ack?username=bob').status_code == 404
        assert client.post('/api/autonomous/alerts/3/ack?username=alice').status_code == 200

def test_camera_actions_do_not_use_other_plot_or_stale_sensors(monkeypatch):
    from core import device_rule_engine, device_executor
    called = []
    monkeypatch.setattr(scheduler_jobs, '_collect_device_rule_context', lambda *a: {'device_configs': [
        {'device_id':'cam', 'plot_id':'1'}, {'device_id':'pump2', 'plot_id':'2'}], 'device_states':{}, 'weather_by_plot':{}})
    monkeypatch.setattr(device_rule_engine, 'RuleEngine', lambda **kw: SimpleNamespace(find_matching_rules=lambda context: []))
    monkeypatch.setattr(device_executor, 'DeviceExecutor', lambda *a, **kw: SimpleNamespace(execute_sync=lambda *a, **kw: called.append(a)))
    from app.agent import config
    monkeypatch.setattr(config, 'get_autonomy_level', lambda _: 'high')
    class Registry:
        async def discover_all(self): return [SimpleNamespace(device_id='pump2', capabilities=[SimpleNamespace(value='irrigate')])]
    loop = asyncio.new_event_loop()
    try: scheduler_jobs._process_camera_actions([{'action':'irrigate', 'urgency':'immediate'}], 'alice', Registry(), loop, 'cam')
    finally: loop.close()
    assert called == []

def test_weather_by_plot_excludes_expired_observations(monkeypatch):
    from core import plot_manager, weather_service
    monkeypatch.setattr(plot_manager, 'PlotManager', lambda username: SimpleNamespace(list_plots=lambda: [
        {'plot_id':'1','lat':30,'lon':120}, {'plot_id':'2','lat':31,'lon':121}]))
    class Weather:
        def get_grid_weather(self, lon, lat):
            stamp = datetime.now() if lon == 120 else datetime.now() - timedelta(days=2)
            return {'temp': 20, 'humidity': 80, 'obsTime': stamp.isoformat()}
    monkeypatch.setattr(weather_service, 'WeatherService', Weather)
    result = scheduler_jobs._weather_by_plot('alice')
    assert set(result) == {'1'}

def test_scoped_risks_ignore_other_users_and_expired(tmp_path, monkeypatch):
    import json
    from app import api_routes
    from core import plot_manager
    monkeypatch.setattr(api_routes, 'DEFAULT_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(plot_manager, 'PlotManager', lambda username: SimpleNamespace(list_plots=lambda: [{'plot_id':'1'}]))
    for username, risk, age in [('alice','高',0), ('bob','中',0), ('carol','高',2)]:
        directory = tmp_path / username / 'disease_risks'; directory.mkdir(parents=True)
        (directory / '1.json').write_text(json.dumps({'plot_id':'1', 'updated':(datetime.now()-timedelta(days=age)).isoformat(), 'risks':[{'risk':risk}]}), encoding='utf8')
    assert api_routes._scoped_disease_risks('alice') == [{'risk':'高','plot_id':'1'}]
    assert api_routes._scoped_disease_risks('bob') == [{'risk':'中','plot_id':'1'}]
    assert api_routes._scoped_disease_risks('carol') == []


def test_authenticated_alert_requests_cannot_impersonate_another_user(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api_routes import register_routes
    from core.auth_token import create_token
    from core import autonomy_runtime
    secret = 's' * 40
    monkeypatch.setenv('REQUIRE_AUTH', 'true'); monkeypatch.setenv('AUTH_SECRET_KEY', secret)
    monkeypatch.setattr(autonomy_runtime, 'list_alerts', lambda username: [{'username':username}])
    app=FastAPI(); register_routes(app)
    with TestClient(app) as client:
        headers={'Authorization':f'Bearer {create_token("alice", secret)}'}
        assert client.get('/api/autonomous/alerts?username=bob', headers=headers).status_code == 403
        assert client.post('/api/autonomous/alerts/1/ack?username=bob', headers=headers).status_code == 403
        assert client.get('/api/autonomous/alerts', headers=headers).json() == [{'username':'alice'}]

def test_disease_job_marks_missing_weather_unavailable(tmp_path, monkeypatch):
    import json
    from core import plot_manager
    monkeypatch.setattr(scheduler_jobs, 'DEFAULT_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(scheduler_jobs, '_disease_usernames', lambda: ['alice'])
    monkeypatch.setattr(scheduler_jobs, '_weather_by_plot', lambda username: {})
    monkeypatch.setattr(plot_manager, 'PlotManager', lambda username: SimpleNamespace(list_plots=lambda: [{'plot_id':'1','crop':'番茄'}]))
    scheduler_jobs.check_disease_job()
    data=json.loads((tmp_path/'alice'/'disease_risks'/'1.json').read_text(encoding='utf8'))
    assert data['quality'] == 'unavailable' and data['risks'] == []
