import asyncio
import hashlib
import json
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.device_executor import DeviceExecutor
from app.routes.device_actions import register_device_action_routes


def setup_executor(monkeypatch, tmp_path, username, state, execution_id='run-1'):
    import core.device_executor as module
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Registry:
        async def read_state(self, device_id):
            return state
    executor = DeviceExecutor(Registry(), username=username)
    marker = executor._pending_path() + '.uncertain.' + hashlib.sha256(b'pump').hexdigest()
    with open(marker, 'w', encoding='utf-8') as handle:
        json.dump({'device_id':'pump', 'execution_id':execution_id}, handle)
    return executor, marker


def fresh(**items):
    return {'_observed_at': datetime.now().isoformat(), '_quality':'good', 'status':'standby', **items}


def test_reconcile_rejects_stale_even_with_matching_history(monkeypatch, tmp_path):
    state = fresh(_observed_at=(datetime.now()-timedelta(hours=1)).isoformat(), run_history={'run-1': {'reason':'completed'}})
    executor, marker = setup_executor(monkeypatch, tmp_path, 'alice', state)
    result = asyncio.run(executor.reconcile_uncertain('pump'))
    assert not result['success']
    assert __import__('os').path.exists(marker)


@pytest.mark.parametrize('state', [
    fresh(run_history={'run-1': {'reason':'completed'}}),
    fresh(_stop_ack={'acknowledged':True, 'execution_id':'run-1'}),
])
def test_reconcile_accepts_fresh_linked_evidence(monkeypatch, tmp_path, state):
    executor, marker = setup_executor(monkeypatch, tmp_path, 'alice', state)
    action_id = executor.add_pending({'device_id':'pump','params':{'duration':1}})
    with executor._pending_transaction():
        executor.pending_actions[0].update(status='uncertain', execution_id='run-1')
        executor._save_pending()
    result = asyncio.run(executor.reconcile_uncertain('pump'))
    assert result['success'] and result['execution_id'] == 'run-1'
    assert not __import__('os').path.exists(marker)
    assert not executor.confirm_pending(action_id)['success']
    assert executor.pending_actions[0]['status'] == 'reconciled'


def test_reconcile_rejects_unlinked_idle_and_other_users(monkeypatch, tmp_path):
    alice, marker = setup_executor(monkeypatch, tmp_path, 'alice', fresh())
    bob = DeviceExecutor(alice.registry, username='bob')
    assert not asyncio.run(bob.reconcile_uncertain('pump'))['success']
    assert not asyncio.run(alice.reconcile_uncertain('pump'))['success']
    assert __import__('os').path.exists(marker)


def test_reconcile_api_returns_reason(monkeypatch, tmp_path):
    import app.routes.device_actions as routes
    executor, _ = setup_executor(monkeypatch, tmp_path, 'alice', fresh())
    class Session:
        def __init__(self, username): pass
        def __enter__(self):
            class Registry:
                async def discover_all(self): return []
                async def read_state(self, device_id): return fresh()
            return Registry(), asyncio.new_event_loop()
        def __exit__(self, *args): pass
    monkeypatch.setattr(routes, 'RegistrySession', Session)
    app = FastAPI()
    register_device_action_routes(app)
    response = TestClient(app).post('/api/devices/pump/reconcile?username=alice')
    assert response.status_code == 409
    assert response.json()['detail']

def test_matching_stop_ack_allows_reconcile_after_fresh_read(monkeypatch, tmp_path):
    from devices.base import DeviceCommand, DeviceResult
    executor, marker = setup_executor(monkeypatch, tmp_path, 'alice', fresh())
    async def stop(device_id, command):
        return DeviceResult(True, device_id, command.command, execution_id=command.execution_id, acknowledged=True)
    executor.registry.execute = stop
    command = DeviceCommand('stop', execution_id='run-1')
    result = asyncio.run(executor.execute('pump', command))
    assert result['success']
    assert asyncio.run(executor.reconcile_uncertain('pump'))['success']


def test_unmatched_stop_ack_cannot_resolve_original(monkeypatch, tmp_path):
    from devices.base import DeviceCommand, DeviceResult
    executor, marker = setup_executor(monkeypatch, tmp_path, 'alice', fresh())
    async def stop(device_id, command):
        return DeviceResult(True, device_id, command.command, execution_id='other', acknowledged=True)
    executor.registry.execute = stop
    result = asyncio.run(executor.execute('pump', DeviceCommand('stop', execution_id='run-1')))
    assert result['success']
    assert not asyncio.run(executor.reconcile_uncertain('pump'))['success']

def test_reconcile_links_crash_marker_to_pending_action(monkeypatch, tmp_path):
    executor, marker = setup_executor(monkeypatch, tmp_path, 'alice', fresh(run_history={'run-1': {'reason':'completed'}}))
    action_id = executor.add_pending({'device_id':'pump','params':{'duration':1}})
    with executor._pending_transaction():
        executor.pending_actions[0]['status'] = 'uncertain'
        executor._save_pending()
    with open(marker, 'w', encoding='utf-8') as handle:
        json.dump({'device_id':'pump','execution_id':'run-1','pending_action_id':action_id}, handle)
    assert asyncio.run(executor.reconcile_uncertain('pump'))['success']
    assert not executor.confirm_pending(action_id)['success']
    assert executor.pending_actions[0]['status'] == 'reconciled'

def test_pending_success_is_recorded_before_unknown_marker_is_removed(monkeypatch, tmp_path):
    from devices.base import DeviceResult
    import core.device_executor as module
    monkeypatch.setattr(module, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Registry:
        async def execute(self, device_id, command):
            return DeviceResult(True, device_id, command.command, execution_id=command.execution_id, acknowledged=True)
    executor = DeviceExecutor(Registry(), username='pending_order')
    action_id = executor.add_pending({'device_id':'pump','command':'start','params':{'duration':1},'capability':'irrigate'})
    original_save = executor._save_pending
    observed = []
    def save():
        action = executor.pending_actions[0]
        marker = executor._pending_path() + '.uncertain.' + hashlib.sha256(b'pump').hexdigest()
        if action.get('status') == 'executed':
            observed.append(__import__('os').path.exists(marker))
        original_save()
    monkeypatch.setattr(executor, '_save_pending', save)
    assert executor.confirm_pending(action_id)['success']
    assert observed == [True]
