"""对话入口、持久卡片和设备确认使用同一条真实业务链路。"""

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_routes import register_routes
from core.device_registry_factory import load_custom_devices, save_custom_devices


def test_对话返回设备候选卡且确认前不启动并可保存恢复(monkeypatch):
    from app import chat_service
    from app.agent.agents.device_agent import DeviceAgent
    username = 'chat_cards_' + uuid4().hex
    save_custom_devices(username, [{'device_id': 'pump', 'name': '麦田灌溉泵', 'driver': 'simulator',
        'capabilities': ['irrigate'], 'initial_state': {'power': True, 'status': 'standby'}}])
    monkeypatch.setenv('AUTONOMY_LEVEL', 'high')
    monkeypatch.setattr(DeviceAgent, '_parse_device_intent', lambda *args: {
        'action': 'irrigate', 'device_hint': 'pump', 'params': {'duration': 1}})
    class Agent:
        def invoke(self, state):
            state.user_question = state.messages[-1].content
            return DeviceAgent().invoke(state)
    monkeypatch.setattr(chat_service, 'get_chat_agent', lambda: Agent())
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        response = client.post(f'/api/chat?username={username}', json={
            'user_question': '请给pump浇水1分钟', 'interactive_cards': True}).json()
        assert not response['error_code']
        pending_id = response['pending_action']['id']
        assert response['cards'] == [{'type': 'device', 'device_id': 'pump', 'pending_action_id': pending_id}]
        assert load_custom_devices(username)[0]['initial_state']['status'] == 'standby'
        saved = client.post(f'/api/chat/sessions?username={username}', json={'messages': [
            {'role': 'assistant', 'content': response['final_answer'], 'cards': response['cards']}]}).json()
        loaded = client.get(f"/api/chat/sessions/{saved['id']}?username={username}").json()
        assert loaded['messages'][0]['cards'] == response['cards']
        confirmed = client.post(f'/api/actions/{pending_id}/confirm?username={username}').json()
        assert confirmed['success']
        assert load_custom_devices(username)[0]['initial_state']['status'] == 'running'
        duplicate = client.post(f'/api/actions/{pending_id}/confirm?username={username}')
        assert duplicate.status_code == 409


def test_预约聊天返回实际任务卡但不返回即时硬件控制卡(tmp_path, monkeypatch):
    from app import chat_service
    from app.agent.nodes import reminder
    username = 'chat_cards_' + uuid4().hex
    monkeypatch.setattr(reminder, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Agent:
        def invoke(self, state):
            state.user_question = state.messages[-1].content
            state.intent_type = 'reminder_setup'
            return reminder.reminder_management_node(state)
    monkeypatch.setattr(chat_service, 'get_chat_agent', lambda: Agent())
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        data = client.post(f'/api/chat?username={username}', json={
            'user_question': '明天下午3点提醒给小麦浇水10分钟', 'interactive_cards': True}).json()
        assert not data['error_code']
        assert len(data['cards']) == 1 and data['cards'][0]['type'] == 'task'
        task = client.get(f"/api/chat/cards/task/{data['cards'][0]['task_id']}?username={username}").json()
        assert task['status'] == '待办' and task['end_date']


def test_重复用户名参数不能绕过卡片归属校验(monkeypatch):
    from core.auth_token import create_token
    secret = 'chat-cards-test-authentication-secret'
    monkeypatch.setenv('REQUIRE_AUTH', 'true')
    monkeypatch.setenv('AUTH_SECRET_KEY', secret)
    attacker = 'attacker_' + uuid4().hex
    victim = 'victim_' + uuid4().hex
    save_custom_devices(victim, [{'device_id': 'private-pump', 'name': '其他账号设备',
        'driver': 'simulator', 'capabilities': ['irrigate']}])
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        headers = {'Authorization': 'Bearer ' + create_token(attacker, secret)}
        query = f'?username={attacker}&username={victim}'
        response = client.get('/api/chat/cards/device/private-pump' + query, headers=headers)
        assert response.status_code == 400
        response = client.post('/api/chat/cards/device/private-pump/prepare' + query,
            headers=headers, json={'command': 'stop', 'params': {}, 'request_id': 'not-authorized'})
        assert response.status_code == 400


def test_循环预约卡显示真实提醒且完成本次任务不会停止循环(tmp_path, monkeypatch):
    from app import chat_service
    from app.agent.nodes import reminder
    from core.planting_tracker import PlantingTracker
    from core.database.repository.reminders import ReminderRepository
    username = 'chat_cards_' + uuid4().hex
    monkeypatch.setattr(reminder, 'DEFAULT_DATA_DIR', str(tmp_path))
    class Agent:
        def invoke(self, state):
            state.user_question = state.messages[-1].content
            state.intent_type = 'reminder_setup'
            return reminder.reminder_management_node(state)
    monkeypatch.setattr(chat_service, 'get_chat_agent', lambda: Agent())
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        data = client.post(f'/api/chat?username={username}', json={
            'user_question': '每天下午3点提醒给小麦浇水10分钟', 'interactive_cards': True}).json()
        assert not data['error_code']
        task_id = data['cards'][0]['task_id']
        task = client.get(f'/api/chat/cards/task/{task_id}?username={username}').json()
        assert task['reminder_frequency'] == '每天'
        assert 'T15:00:' in task['reminder_time']
        tracker = PlantingTracker(str(tmp_path / username))
        assert tracker.update_task_status(task_id, '已完成', 100)
        assert len(ReminderRepository().find_by(user_id=tracker._uid, task_id=int(task_id), status='active')) == 1
