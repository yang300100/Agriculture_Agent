"""哥哥，设备停机和作物维护效果必须分别取得证据。"""

from datetime import datetime, timedelta

from core.autonomy_feedback import evaluate_effect


def _inputs():
    now = datetime.now()
    payload = {'execution_id': 'run-a', 'action': {'device_id': 'pump'},
               'baseline': {'timestamp': (now - timedelta(minutes=5)).isoformat(),
                            'sensor_readings': {'soil.soil_moisture': 20}},
               'feedback': {'sensor_id': 'soil', 'field': 'soil_moisture', 'min_delta': 3}}
    state = {'_quality': 'good', '_observed_at': now.isoformat(), 'status': 'standby',
             'run_history': {'run-a': {'reason': 'completed', 'ended_at': (now - timedelta(seconds=1)).isoformat()}}}
    sensors = {'soil': {'soil_moisture': 24, '_quality': 'good', '_observed_at': now.isoformat()}}
    return payload, state, sensors, now


def test_同一运行完成且湿度提升达到判据才确认维护效果():
    payload, state, sensors, now = _inputs()
    assert evaluate_effect(payload, state, sensors, now=now)['success'] is True
    sensors['soil']['soil_moisture'] = 21
    result = evaluate_effect(payload, state, sensors, now=now)
    assert result['success'] is False and result['terminal'] is True
    assert result['comparison']['before'] == 20
    assert result['comparison']['after'] == 21


def test_陈旧观测和前一轮结束不能作为本轮维护成功():
    payload, state, sensors, now = _inputs()
    sensors['soil']['_observed_at'] = (now - timedelta(hours=1)).isoformat()
    assert evaluate_effect(payload, state, sensors, now=now)['status'] == 'unverified'
    payload, state, sensors, now = _inputs()
    state['run_history'] = {'older-run': {'reason': 'completed'}}
    assert evaluate_effect(payload, state, sensors, now=now)['success'] is False


def test_没有配置判据或者无开始前基线时明确未验证():
    payload, state, sensors, now = _inputs()
    payload['feedback'] = {}
    assert evaluate_effect(payload, state, sensors, now=now)['status'] == 'unverified'
    payload, state, sensors, now = _inputs()
    payload['baseline']['sensor_readings'] = {}
    assert evaluate_effect(payload, state, sensors, now=now)['status'] == 'unverified'


def test_降温反馈按降低方向判断且不接受操作前的旧数据():
    payload, state, sensors, now = _inputs()
    payload['feedback'].update(field='temperature', direction='decrease', min_delta=2)
    payload['baseline']['sensor_readings'] = {'soil.temperature': 30}
    sensors['soil']['temperature'] = 27
    assert evaluate_effect(payload, state, sensors, now=now)['success'] is True
    sensors['soil']['_observed_at'] = (now - timedelta(minutes=6)).isoformat()
    assert evaluate_effect(payload, state, sensors, now=now)['success'] is False


def test_设备结束前的缓存读数不能确认维护后的效果():
    payload, state, sensors, now = _inputs()
    sensors['soil']['_observed_at'] = (now - timedelta(seconds=2)).isoformat()
    assert evaluate_effect(payload, state, sensors, now=now)['status'] == 'unverified'
    state['run_history']['run-a'].pop('ended_at')
    assert evaluate_effect(payload, state, sensors, now=now)['status'] == 'unverified'


def test_无编号或陈旧设备状态不得触发自动停止(monkeypatch):
    import asyncio
    from contextlib import contextmanager
    from core import device_executor, device_registry_factory as factory
    from core.autonomy_feedback import run_effect_check
    state = {'_quality': 'good', '_observed_at': datetime.now().isoformat(), 'status': 'running'}
    class Registry:
        async def discover_all(self): return []
        async def read_state(self, device_id): return state
    @contextmanager
    def session(username):
        loop = asyncio.new_event_loop()
        try: yield Registry(), loop
        finally: loop.close()
    monkeypatch.setattr(factory, 'RegistrySession', session)
    monkeypatch.setattr(factory, 'load_custom_devices', lambda username: [{'device_id': 'pump', 'plot_id': '1'}])
    class Executor:
        def __init__(self, *a, **kw): pass
        def execute_sync(self, *a, **kw): raise AssertionError('不能用缺失或过期的编号误停设备')
    monkeypatch.setattr(device_executor, 'DeviceExecutor', Executor)
    job = {'username': 'alice', 'region': '1', 'payload': {'action': {'device_id': 'pump'}}}
    assert run_effect_check(job)['status'] == 'unverified'
    state['run_id'] = 'run-a'
    job['payload']['execution_id'] = 'run-a'
    state['_observed_at'] = (datetime.now() - timedelta(hours=1)).isoformat()
    assert run_effect_check(job)['status'] == 'unverified'


def test_坏复查配置不能吞掉已成功下发的结果(monkeypatch):
    from core import device_registry_factory as factory
    from core.autonomy_runtime import queue_effect_check
    monkeypatch.setattr(factory, 'load_custom_devices', lambda username: [
        {'device_id': 'pump', 'connection': {'feedback': {'settle_seconds': '错误'}}}])
    job = queue_effect_check('feedback_bad_config', '1', {'device_id': 'pump', 'params': {'duration': 1}},
        {'success': True, 'execution_id': 'run-config'}, {})
    assert job['kind'] == 'effect_check'
    assert evaluate_effect(job['payload'], {}, {})['status'] == 'unverified'
