import asyncio
import pytest
from core.device_safety_policy import SafetyPolicyService, REJECTED, AUTO_EXECUTE
from core.device_rule_engine import RuleEngine
from core.device_executor import DeviceExecutor
from devices.base import DeviceCommand, DeviceResult

@pytest.mark.parametrize('params', [{}, {'duration': 0}, {'duration': 1, 'target_temp': 999}, {'duration': 1, 'concentration_percent': 101}])
def test_start_rejects_missing_duration_and_unsafe_dimensions(params):
    assert SafetyPolicyService(policies=[]).evaluate('x', 'heat', params).decision == REJECTED

def test_parameter_range_can_restrict_and_automatic_high_risk_requires_range():
    service = SafetyPolicyService(policies=[{'name':'范围', 'scope_type':'global', 'limits':{'parameter_ranges':{'target_temp': [10, 30]}}, 'violation_action':'reject'}])
    assert service.evaluate('x', 'heat', {'duration': 1, 'target_temp': 35}).decision == REJECTED
    assert SafetyPolicyService(policies=[]).evaluate('x', 'heat', {'duration': 1, 'target_temp': 25}, context={'automatic': True}).decision == REJECTED

def test_context_selects_plot_weather():
    engine = RuleEngine('test_user')
    context = {'device_configs':[{'device_id':'x','plot_id':2}], 'device_states':{}, 'weather':{'rain':99}, 'weather_by_plot': {'2': {'rain': 1}}}
    assert engine.context_for_device('x', context)['weather'] == {'rain':1}

def test_ambiguous_result_is_not_retried(monkeypatch, tmp_path):
    import core.device_executor as module
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Registry:
        calls = 0
        async def execute(self, device_id, command):
            self.calls += 1
            return DeviceResult(False, device_id, command.command, error_code='ACK_UNCONFIRMED')
    registry = Registry()
    result = asyncio.run(DeviceExecutor(registry).execute('x', DeviceCommand('start', {'duration':1}), skip_policy=True))
    assert result['attempts'] == 1
    assert registry.calls == 1

def test_model_action_inherits_rule_constraints(monkeypatch, tmp_path):
    import core.device_executor as module
    from core.device_rule_engine import RuleEngine
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(RuleEngine, '_load_rules', lambda self: setattr(self, 'rules', [{'enabled':True, 'action':{'device_id':'x'}, 'constraints':{'max_duration_per_use': 5}, 'execution_mode':'auto'}]))
    class Registry:
        async def execute(self, device_id, command):
            raise AssertionError('超限动作不能下发')
    result = asyncio.run(DeviceExecutor(Registry()).execute('x', DeviceCommand('start', {'duration':10}), trigger='autonomous', capability='irrigate'))
    assert result['success'] is False
    assert result['decision'] in ('need_confirm', 'rejected')

def test_same_device_concurrent_calls_do_not_overlap(monkeypatch, tmp_path):
    import core.device_executor as module
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Registry:
        active = 0
        overlap = False
        async def execute(self, device_id, command):
            self.active += 1
            self.overlap |= self.active > 1
            await asyncio.sleep(0.03)
            self.active -= 1
            return DeviceResult(True, device_id, command.command)
    registry = Registry()
    async def run():
        executor = DeviceExecutor(registry)
        return await asyncio.gather(*(executor.execute('x', DeviceCommand('start', {'duration':1}), skip_policy=True) for _ in range(2)))
    asyncio.run(run())
    assert not registry.overlap

def test_automatic_rejects_real_device_without_contract(monkeypatch, tmp_path):
    import core.device_executor as module
    import core.device_registry_factory as factory
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(factory, 'load_custom_devices', lambda username: [{'device_id':'real', 'driver':'mqtt', 'capabilities':['irrigate'], 'connection':{}}])
    class Registry:
        async def execute(self, device_id, command):
            raise AssertionError('契约不足不能下发')
    result = asyncio.run(DeviceExecutor(Registry()).execute('real', DeviceCommand('start', {'duration':1}), trigger='autonomous'))
    assert result['decision'] == 'rejected'

def test_climate_does_not_assume_unknown_opposite_state_is_idle(monkeypatch, tmp_path):
    import core.device_executor as module
    import core.device_registry_factory as factory
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(factory, 'load_custom_devices', lambda username: [
        {'device_id':'heater', 'driver':'simulator', 'plot_id':1, 'zone_id':'a', 'capabilities':['heat']},
        {'device_id':'cooler', 'driver':'simulator', 'plot_id':1, 'zone_id':'a', 'capabilities':['cool']},
    ])
    class Registry:
        async def read_state(self, device_id): return {'error':'offline'}
        async def execute(self, device_id, command): raise AssertionError('未知状态不能下发')
    result = asyncio.run(DeviceExecutor(Registry()).execute('heater', DeviceCommand('start', {'duration':1}), trigger='autonomous'))
    assert result['success'] is False
    assert result['attempts'] == 0

def test_unconfirmed_pending_result_cannot_be_reconfirmed(monkeypatch, tmp_path):
    import core.device_executor as module
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Registry:
        async def execute(self, device_id, command):
            return DeviceResult(False, device_id, command.command, error_code='ACK_UNCONFIRMED')
    executor = DeviceExecutor(Registry(), username="uncertain_test")
    action_id = executor.add_pending({'device_id':'x', 'command':'start', 'params':{'duration':1}, 'capability':'irrigate'})
    first = executor.confirm_pending(action_id)
    assert first['action_status'] == 'uncertain', first
    assert not executor.confirm_pending(action_id)['success']


def test_stale_climate_status_is_not_idle(monkeypatch, tmp_path):
    import core.device_executor as module
    import core.device_registry_factory as factory
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(factory, 'load_custom_devices', lambda username: [
        {'device_id':'heater', 'driver':'simulator', 'plot_id':1, 'capabilities':['heat']},
        {'device_id':'cooler', 'driver':'simulator', 'plot_id':1, 'capabilities':['cool']},
    ])
    class Registry:
        async def read_state(self, device_id): return {'status':'standby', '_quality':'good', '_observed_at':'2000-01-01T00:00:00'}
        async def execute(self, device_id, command): return DeviceResult(True, device_id, command.command)
    result = asyncio.run(DeviceExecutor(Registry(), username='stale_climate').execute('heater', DeviceCommand('start', {'duration':1}), trigger='autonomous'))
    assert result['success'] is False
    assert result['attempts'] == 0

@pytest.mark.parametrize('params', [{'duration': True}, {'duration':0, 'duration_minutes':1}, {'duration':1, 'duration_minutes':2}])
def test_start_rejects_boolean_or_conflicting_durations(params):
    assert SafetyPolicyService(policies=[]).evaluate('x', 'irrigate', params).decision == REJECTED

def test_ambiguous_result_blocks_later_automatic_start(monkeypatch, tmp_path):
    import core.device_executor as module
    import core.device_registry_factory as factory
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(factory, 'load_custom_devices', lambda username: [{'device_id':'x', 'driver':'simulator', 'capabilities':['irrigate']}])
    class Registry:
        calls = 0
        async def execute(self, device_id, command):
            self.calls += 1
            return DeviceResult(False, device_id, command.command, error_code='ACK_UNCONFIRMED')
    registry = Registry()
    async def run():
        one = DeviceExecutor(registry, username='ambiguous_test')
        await one.execute('x', DeviceCommand('start', {'duration':1}), trigger='autonomous')
        two = DeviceExecutor(registry, username='ambiguous_test')
        return await two.execute('x', DeviceCommand('start', {'duration':1}), trigger='autonomous')
    second = asyncio.run(run())
    assert registry.calls == 1
    assert second['attempts'] == 0
@pytest.mark.parametrize('key,value', [('target_lux',1000), ('brightness_percent',50), ('power_percent',50), ('speed_percent',50), ('target_soil_moisture',50)])
def test_automatic_targets_require_explicit_agronomic_ranges(key, value):
    result = SafetyPolicyService(policies=[]).evaluate('x', 'light', {'duration':1, key:value}, context={'automatic': True})
    assert result.decision == REJECTED
