import asyncio
from datetime import datetime, timedelta

import pytest
from core.device_executor import DeviceExecutor
from devices.base import DeviceCommand, DeviceResult, DeviceCapability
from devices.simulator_driver import SimulatorDriver


@pytest.mark.parametrize('state,run_id', [
    ({'status':'running','run_id':'new','_quality':'good','_observed_at':datetime.now().isoformat()}, 'old'),
    ({'status':'running','run_id':'old','_quality':'good','_observed_at':(datetime.now()-timedelta(hours=1)).isoformat()}, 'old'),
    ({'status':'standby','run_id':'old','_quality':'good','_observed_at':datetime.now().isoformat()}, 'old'),
    ({'status':'running','run_id':'old','_quality':'good','_observed_at':datetime.now().isoformat()}, None),
])
def test_effect_stop_rechecks_current_run(monkeypatch, tmp_path, state, run_id):
    import core.device_executor as module
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Registry:
        calls = 0
        async def read_state(self, device_id): return state
        async def execute(self, device_id, command):
            self.calls += 1
            return DeviceResult(True, device_id, command.command)
    registry = Registry()
    result = asyncio.run(DeviceExecutor(registry, 'guard').execute('pump', DeviceCommand('stop', execution_id=run_id), trigger='effect_check'))
    assert result['success'] is False
    assert result['attempts'] == 0
    assert registry.calls == 0


def test_simulator_rejects_stop_for_old_run():
    async def run():
        sim = SimulatorDriver(simulated_latency_ms=0)
        sim.add_virtual_device('pump','pump',[DeviceCapability.IRRIGATE],[], '')
        await sim.connect()
        await sim.execute('pump', DeviceCommand('start', {'duration':1}, execution_id='new'))
        result = await sim.execute('pump', DeviceCommand('stop', execution_id='old'))
        state = await sim.read_state('pump')
        assert not result.success
        assert state['status'] == 'running' and state['run_id'] == 'new'
    asyncio.run(run())

def test_effect_stop_allows_matching_fresh_running_state(monkeypatch, tmp_path):
    import core.device_executor as module
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Registry:
        async def read_state(self, device_id):
            return {'status':'running','run_id':'same','_quality':'good','_observed_at':datetime.now().isoformat()}
        async def execute(self, device_id, command):
            return DeviceResult(True, device_id, command.command)
    result = asyncio.run(DeviceExecutor(Registry(), 'guard').execute('pump', DeviceCommand('stop', execution_id='same'), trigger='effect_check'))
    assert result['success'] is True

@pytest.mark.parametrize('legacy', [False, True])
def test_generic_stop_via_executor_stops_running_simulator_with_any_action_id(monkeypatch, tmp_path, legacy):
    import core.device_executor as module
    from devices.registry import DeviceDriverRegistry
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    async def run():
        sim = SimulatorDriver(simulated_latency_ms=0)
        state = {'status':'running','power':True} if legacy else {'status':'powered_off','power':False}
        sim.add_virtual_device('pump','pump',[DeviceCapability.IRRIGATE],[], '', initial_state=state)
        await sim.connect()
        registry = DeviceDriverRegistry()
        registry.register('simulator', sim)
        await registry.discover_all()
        if not legacy:
            assert (await registry.execute('pump', DeviceCommand('start', {'duration':1}, execution_id='active'))).success
        result = await DeviceExecutor(registry, 'generic').execute('pump', DeviceCommand('stop', execution_id='random-action-id'), trigger='rule')
        assert result['success'] is True, result
        assert (await registry.read_state('pump'))['status'] == 'standby'
    asyncio.run(run())
