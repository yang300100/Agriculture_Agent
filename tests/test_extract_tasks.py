"""任务抽取失败路径回归测试。"""

from app.agent.nodes import extract_tasks


def test_llm超时返回空建议而不是引用节点状态(monkeypatch):
    class FailingLLM:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            raise TimeoutError("模拟超时")

    monkeypatch.setattr(extract_tasks, "ChatOpenAI", FailingLLM)

    result = extract_tasks.extract_suggestions_from_answer(
        "请在今天完成一次灌溉。", "番茄", "应该怎么做？"
    )

    assert result == []
