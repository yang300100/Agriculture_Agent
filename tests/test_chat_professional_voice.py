"""验证智能对话使用专业身份，保留业务事实和历史记录。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.agents import device_agent
from app.agent.nodes import llm_response
from app.agent.state import AgentState
from devices.base import DeviceCapability, DeviceInfo


@pytest.mark.parametrize('question,parsed,device_count,expected', [
    ('明天浇水，现在先别执行', None, 0, '任务列表'),
    ('帮我操作设备', None, 0, '设备'),
    ('查看设备状态', {'action': 'status', 'params': {}}, 0, '没有'),
    ('浇水一分钟', {'action': 'irrigate', 'params': {'duration': 1}}, 2, '指定'),
])
def test_设备对话的澄清和状态回复不使用亲属称谓(monkeypatch, question, parsed, device_count, expected):
    devices = [DeviceInfo(f'pump-{i}', f'{i}号水泵', 'simulator', [DeviceCapability.IRRIGATE]) for i in range(device_count)]
    monkeypatch.setattr(device_agent, '_discover_devices', lambda username: devices)
    agent = device_agent.DeviceAgent()
    monkeypatch.setattr(agent, '_parse_device_intent', lambda *args: parsed)
    state = agent.invoke(AgentState(username='voice_test', user_question=question))
    assert expected in state.final_answer
    assert not any(word in state.final_answer for word in ('哥哥', '妹妹', 'Yuki', '～'))


@pytest.mark.parametrize('entry', ['llm_response_node', 'llm_expert_answer', 'clarification_node'])
def test_模型入口固定专业身份且旧角色历史不会被当作身份设定(monkeypatch, entry):
    captured = []
    class Model:
        def __init__(self, **kwargs): pass
        def invoke(self, messages):
            captured.append(messages)
            return AIMessage(content='我是青禾智能农场的农业助手。')
    monkeypatch.setattr(llm_response, 'ChatOpenAI', Model)
    monkeypatch.setattr(llm_response, 'extract_facts_from_conversation', lambda *args, **kwargs: {})
    previous = AIMessage(content='哥哥，我是你的妹妹，帮你管理小麦。')
    state = AgentState(user_question='你是谁', intent_type='identity', need_clarification=True,
        messages=[previous, HumanMessage(content='你是谁')], short_term_facts={'crop': '小麦'})
    result = getattr(llm_response, entry)(state)
    messages = captured[0]
    assert isinstance(messages[0], SystemMessage)
    assert '专业农业助手' in messages[0].content
    assert '不扮演亲属或虚拟人物' in messages[0].content
    assert '历史' in messages[0].content
    assert result.short_term_facts['crop'] == '小麦'
    assert result.messages[0].content == previous.content


def test_流式回复同样携带专业身份约束():
    captured, events = [], []
    class Model:
        def stream(self, messages):
            captured.extend(messages)
            yield SimpleNamespace(content='农业助手')
    token = llm_response._stream_token_callback.set(events.append)
    try:
        result = llm_response._stream_invoke(Model(), [HumanMessage(content='介绍一下你自己')])
    finally:
        llm_response._stream_token_callback.reset(token)
    assert isinstance(captured[0], SystemMessage)
    assert '不扮演亲属或虚拟人物' in captured[0].content
    assert result.content == '农业助手'
    assert events == [{'type': 'token', 'token': '农业助手'}]
