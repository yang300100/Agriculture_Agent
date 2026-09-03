"""对话资源复用测试。"""

from app import chat_service


def test_对话agent在进程内只初始化一次(monkeypatch):
    calls = {"rag": 0, "faiss": 0, "build": 0}

    class FakeRag:
        def __init__(self):
            calls["rag"] += 1

    class FakeFaiss:
        is_available = False

        def __init__(self):
            calls["faiss"] += 1

    def fake_build(rag, faiss):
        calls["build"] += 1
        return object()

    import app.agent.graph as graph
    import knowledge.faiss_agriculture_rag as faiss_module
    import knowledge.simple_agriculture_rag as simple_module

    monkeypatch.setattr(graph, "build_agricultural_policy_agent", fake_build)
    monkeypatch.setattr(simple_module, "SimpleAgricultureRAG", FakeRag)
    monkeypatch.setattr(faiss_module, "FAISSAgricultureRAG", FakeFaiss)
    chat_service.clear_chat_agent_cache()

    assert chat_service.get_chat_agent() is chat_service.get_chat_agent()
    assert calls == {"rag": 1, "faiss": 1, "build": 1}
    chat_service.clear_chat_agent_cache()
