"""政策节点资源复用测试。"""

from app.agent.nodes import policy as policy_node


def test_政策查询复用知识库实例(monkeypatch):
    import knowledge.faiss_agriculture_rag as faiss_module
    import knowledge.simple_agriculture_rag as simple_module

    created = {"faiss": 0, "simple": 0}

    class FakeFaiss:
        is_available = True

        def __init__(self):
            created["faiss"] += 1

    class FakeSimple:
        def __init__(self):
            created["simple"] += 1

    monkeypatch.setattr(faiss_module, "FAISSAgricultureRAG", FakeFaiss)
    monkeypatch.setattr(simple_module, "SimpleAgricultureRAG", FakeSimple)
    policy_node._get_policy_rag_resources.cache_clear()
    try:
        first = policy_node._get_policy_rag_resources()
        second = policy_node._get_policy_rag_resources()
    finally:
        policy_node._get_policy_rag_resources.cache_clear()

    assert first is second
    assert created == {"faiss": 1, "simple": 1}

