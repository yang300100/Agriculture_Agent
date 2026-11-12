"""预约消息只生成预约，不追加即时设备路由的内部提示。"""

from uuid import uuid4

from app.agent.agents.orchestrator import AgentOrchestrator
from app.agent.agents.device_agent import DeviceAgent
from app.agent.state import AgentState
from app.agent.nodes import reminder


def test_明确预约不会再调用即时设备代理(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(reminder, 'DEFAULT_DATA_DIR', str(tmp_path))
    def device_call(self, state):
        calls.append(state.user_question)
        state.final_answer = '即时设备提示'
        return state
    monkeypatch.setattr(DeviceAgent, 'invoke', device_call)
    state = AgentState(username='cards_route_' + uuid4().hex, interactive_cards=True,
        intent_type='reminder_setup', user_question='明天下午3点提醒我用麦田灌溉阀给小麦浇水10分钟')
    AgentOrchestrator().dispatch(state)
    assert len(state.reminders) == 1
    assert not calls
    assert 'device_control' not in state.final_answer


def test_复合回答使用用户能理解的中文标题():
    answer = AgentOrchestrator()._merge_answers({
        'weather_query': '当前天气数据不可用。', 'device_control': '设备信息请查看卡片。'}, 'weather_query')
    assert 'device_control' not in answer and '设备操作' in answer


def test_预约之外明确要求现在停止时仍保留设备操作意图():
    intents = AgentOrchestrator()._detect_secondary('现在停止麦田灌溉阀，并在明天下午3点提醒我浇水', 'reminder_setup')
    assert 'device_control' in intents
