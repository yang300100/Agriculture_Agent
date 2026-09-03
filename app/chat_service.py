"""对话 Agent 的进程级资源缓存。"""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_chat_agent():
    """复用知识库与编译后的对话图，避免每个请求重复初始化。"""
    from app.agent.graph import build_agricultural_policy_agent
    from knowledge.faiss_agriculture_rag import FAISSAgricultureRAG
    from knowledge.simple_agriculture_rag import SimpleAgricultureRAG

    rag = SimpleAgricultureRAG()
    faiss_candidate = FAISSAgricultureRAG()
    faiss = faiss_candidate if faiss_candidate.is_available else None
    return build_agricultural_policy_agent(rag, faiss)


def clear_chat_agent_cache():
    """供测试或知识库热更新后显式清理缓存。"""
    get_chat_agent.cache_clear()
