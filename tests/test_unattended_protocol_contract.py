import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from devices.base import DeviceCommand, DeviceResult


def contract(driver='http'):
    return {'driver': driver, 'connection': {'autonomy': {
        'local_timed_stop': {'duration_param': 'duration', 'unit': 'minutes'},
        'link_loss_stop': {'mode': 'firmware_watchdog', 'timeout_seconds': 30},
        'execution_id': {'field': 'execution_id', 'deduplicates': True},
        'ack': {'execution_id_field': 'execution_id', 'success_field': 'success', **({'topic': 'device/ack'} if driver == 'mqtt' else {})},
    }}}


def test_contract_requires_structured_capabilities_and_supported_driver():
    from core.device_execution_contract import unattended_contract_error
    assert unattended_contract_error({'driver': 'simulator'}, 'start', {'duration': 1}) is None
    assert unattended_contract_error(contract(), 'start', {'duration': 1}) is None
    assert unattended_contract_error(contract('mqtt'), 'start', {'duration': 1}) is None
    assert unattended_contract_error(contract('modbus'), 'start', {'duration': 1})
    bad = contract(); bad['connection']['autonomy']['local_timed_stop'] = True
    assert unattended_contract_error(bad, 'start', {'duration': 1})
    assert unattended_contract_error({'driver': 'http'}, 'stop', {}) is None


def test_result_defaults_to_unacknowledged():
    result = DeviceResult(True, 'd', 'start')
    assert result.execution_id is None and not result.acknowledged and not result.retry_safe


def test_http_execution_id_requires_matching_ack_and_decodes_capture():
    from devices.http_driver import HTTPDriver
    d = HTTPDriver(); d.register_device('d', 'd', [], base_url='http://invalid')
    response = SimpleNamespace(status_code=200, text='x', json=lambda: {'success': True, 'execution_id': 'wrong', 'image_base64': 'YWJj'})
    d._async_request = AsyncMock(return_value=response)
    result = asyncio.run(d.execute('d', DeviceCommand('capture', execution_id='id')))
    assert not result.success
    response.json = lambda: {'success': True, 'execution_id': 'id', 'image_base64': 'YWJj'}
    result = asyncio.run(d.execute('d', DeviceCommand('capture', execution_id='id')))
    assert result.success and result.acknowledged and result.raw_response['image_bytes'] == b'abc'
    assert d._async_request.call_args.kwargs['json']['execution_id'] == 'id'


def test_http_observation_time_survives_failure():
    from devices.http_driver import HTTPDriver
    d = HTTPDriver(); d.register_device('d', 'd', [], base_url='http://invalid')
    response = SimpleNamespace(status_code=200, json=lambda: {'temperature': 0})
    d._async_request = AsyncMock(side_effect=[response, RuntimeError('offline')])
    a = asyncio.run(d.read_state('d')); b = asyncio.run(d.read_state('d'))
    assert a['_quality'] == 'good' and a['_observed_at']
    assert b['_quality'] == 'error' and b['_observed_at'] == a['_observed_at']

def test_mqtt_publish_does_not_claim_success_without_ack(monkeypatch):
    import devices.mqtt_driver as m
    monkeypatch.setattr(m, 'HAS_PAHO', True, raising=False)
    d = m.MQTTDriver(); d.register_device('d', 'd', [])
    d._connected = True
    d._client = SimpleNamespace(publish=lambda *a, **k: SimpleNamespace(rc=0, wait_for_publish=lambda **k: None))
    result = asyncio.run(d.execute('d', DeviceCommand('start', execution_id='id', timeout_ms=10)))
    assert not result.success and result.error_code == 'ACK_UNCONFIRMED'
    assert asyncio.run(d.read_state('d'))['status'] != 'running'


def test_modbus_fractional_duration_and_zero_observation(monkeypatch):
    import devices.modbus_driver as m
    monkeypatch.setattr(m, 'HAS_PYMODBUS', True, raising=False)
    d = m.ModbusDriver(); d.register_device('d', 'd', [])
    writes = []
    class Client:
        def write_registers(self, address, values, **kwargs):
            writes.append(values); return SimpleNamespace(isError=lambda: False)
        def read_holding_registers(self, address, count, **kwargs):
            return SimpleNamespace(isError=lambda: False, registers=[0]*15)
    d._connected = True; d._client = Client()
    result = asyncio.run(d.execute('d', DeviceCommand('start', {'duration': .5})))
    assert result.success and writes[0][1] == 30
    state = asyncio.run(d.read_state('d'))
    assert state['temperature'] == 0 and state['_quality'] == 'good'

def test_coap_id_ack_and_observation(monkeypatch):
    import devices.coap_driver as m
    monkeypatch.setattr(m, 'HAS_AIOCOAP', True)
    d = m.CoAPDriver(); d.register_device('d', 'd', [], base_uri='coap://invalid')
    response = SimpleNamespace(code=SimpleNamespace(is_successful=lambda: True), payload=b'{"success":true,"execution_id":"wrong"}')
    d._request = AsyncMock(return_value=response)
    result = asyncio.run(d.execute('d', DeviceCommand('start', execution_id='id')))
    assert not result.success and result.error_code == 'ACK_UNCONFIRMED'
    assert d._request.call_args.args[2]['execution_id'] == 'id'
    response.payload = b'{"temperature":0}'
    state = asyncio.run(d.read_state('d'))
    assert state['_quality'] == 'good' and state['_observed_at']


def test_opcua_rejects_unmapped_duration(monkeypatch):
    import devices.opcua_driver as m
    monkeypatch.setattr(m, 'HAS_ASYNCUA', True)
    d = m.OPCUADriver(); d.register_device('d', 'd', [], endpoint='opc.tcp://invalid', command_nodes={'start': 'node'})
    result = asyncio.run(d.execute('d', DeviceCommand('start', {'duration': 1})))
    assert not result.success and result.error_code == 'INVALID_CONFIG'

def test_mqtt_matched_ack_and_receive_timestamp(monkeypatch):
    import devices.mqtt_driver as m
    monkeypatch.setattr(m, 'HAS_PAHO', True, raising=False)
    d = m.MQTTDriver(); d.register_device('d', 'd', [], ack_topic='d/ack')
    d._connected = True
    def publish(topic, payload, qos):
        msg = SimpleNamespace(topic='d/ack', payload=json.dumps({'device_id':'d', 'execution_id':'id', 'success':True}).encode())
        d._on_message(None, None, msg)
        return SimpleNamespace(rc=0, wait_for_publish=lambda **kw: None)
    d._client = SimpleNamespace(publish=publish)
    result = asyncio.run(d.execute('d', DeviceCommand('start', execution_id='id', timeout_ms=100)))
    assert result.success and result.acknowledged
    d._on_message(None, None, SimpleNamespace(topic='devices/d/state', payload=b'{"temperature":0}'))
    a = asyncio.run(d.read_state('d')); b = asyncio.run(d.read_state('d'))
    assert a['_quality'] == 'good' and a['_observed_at'] == b['_observed_at']


def test_simulator_observation_and_execution_confirmation():
    from devices.simulator_driver import SimulatorDriver
    d = SimulatorDriver(simulated_latency_ms=0)
    asyncio.run(d.connect()); d.add_virtual_device('d', 'd', [])
    result = asyncio.run(d.execute('d', DeviceCommand('start', {'duration': 1}, execution_id='id')))
    state = asyncio.run(d.read_state('d'))
    assert result.acknowledged and result.retry_safe and result.execution_id == 'id'
    assert state['_quality'] == 'good' and state['_observed_at']

def test_modbus_bus_serializes_shared_client():
    import threading
    import time
    from devices.modbus_driver import ModbusDriver
    d = ModbusDriver(); d.register_device('a', 'a', []); d.register_device('b', 'b', [], slave_id=2)
    active = 0; maximum = 0; guard = threading.Lock()
    class Client:
        def write_registers(self, address, values, **kwargs):
            nonlocal active, maximum
            with guard:
                active += 1; maximum = max(maximum, active)
            time.sleep(.04)
            with guard:
                active -= 1
            return SimpleNamespace(isError=lambda: False)
    d._connected = True; d._client = Client()
    async def run():
        await asyncio.gather(d.execute('a', DeviceCommand('stop')), d.execute('b', DeviceCommand('stop')))
    asyncio.run(run())
    assert maximum == 1

def test_mqtt_offline_cache_is_error_not_new_observation():
    from devices.mqtt_driver import MQTTDriver
    d = MQTTDriver(); d.register_device('d', 'd', [])
    d._on_message(None, None, SimpleNamespace(topic='devices/d/state', payload=b'{"temperature":0}'))
    old = d._state_cache['d']['_observed_at']
    state = asyncio.run(d.read_state('d'))
    assert state['_quality'] == 'error' and state['_observed_at'] == old
