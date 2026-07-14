"""RAG 检索节点：FAISS 向量检索 + 关键词匹配双通道检索"""

import logging
from typing import Optional

from knowledge.simple_agriculture_rag import SimpleAgricultureRAG
from knowledge.faiss_agriculture_rag import FAISSAgricultureRAG

from ..state import AgentState
from ..config import RAG_TOP_K

logger = logging.getLogger(__name__)


SYMPTOM_KEYWORDS = [
    "发黄", "变黄", "枯黄", "黄化", "黄斑",
    "斑点", "病斑", "黑斑", "褐斑", "白斑",
    "枯萎", "萎蔫", "枯死", "干枯",
    "腐烂", "烂根", "烂叶", "烂果",
    "白粉", "霉层", "霉斑", "霜霉",
    "锈斑", "锈粉", "黑穗", "黑粉",
    "花叶", "皱缩", "畸形", "卷叶",
    "虫眼", "虫孔", "蛀入", "啃食",
    "蚜虫", "螟虫", "青虫", "蛾",
]


def _extract_symptoms(query: str) -> str:
    """从用户问题中提取症状关键词"""
    found = [w for w in SYMPTOM_KEYWORDS if w in query]
    return " ".join(found[:3]) if found else ""


def _normalize_faiss_result(result: dict) -> dict:
    """将 FAISS 检索结果统一为 rag_retrieval 下游期望的格式"""
    meta = result.get("metadata", {})
    return {
        "page_content": result.get("content", ""),
        "source": meta.get("crop", meta.get("source", "未知来源")),
    }


def rag_retrieval_node(state: AgentState,
                       rag_system: SimpleAgricultureRAG,
                       faiss_rag: Optional[FAISSAgricultureRAG] = None) -> AgentState:
    state.progress_message = "正在检索农业知识库..."
    """RAG 检索节点 - FAISS 向量检索为主，关键词匹配为 fallback"""

    # need_rag 检查放在最前面，不需要检索时直接返回
    if not state.need_rag:
        state.retrieved_docs = []
        return state

    queries = []

    if state.user_question:
        queries.append(state.user_question)

    if state.image_analysis_result:
        crop_type = state.image_analysis_result.get("crop_type", "")
        for issue in state.image_analysis_result.get("detected_issues", []):
            issue_name = issue.get("name", "")
            if crop_type and issue_name:
                queries.append(f"{crop_type}{issue_name}防治方法")

    # 注入节气上下文
    if state.intent_type in ("crop_selection", "planting_schedule", "planting_method"):
        try:
            from core.lunar_calendar import get_farming_context_for_query
            ctx = get_farming_context_for_query(state.user_question or "")
            if ctx:
                queries[0] = f"{ctx} {queries[0]}" if queries else f"{ctx}"
        except Exception:
            pass

    # 病虫害意图：提取症状+作物，构造精准查询
    if state.intent_type == "disease_prevention" and queries:
        crop = state.short_term_facts.get("crop", "")
        symptom_words = _extract_symptoms(state.user_question or "")
        if crop and symptom_words:
            queries.insert(0, f"{crop} {symptom_words} 病害 防治方法")
        elif crop:
            queries.insert(0, f"{crop} 常见病害 常见虫害 防治")

    all_results = []
    for query in queries[:2]:
        # 第一通道：FAISS 向量检索
        if faiss_rag and faiss_rag.is_available:
            try:
                faiss_results = faiss_rag.search(query, k=RAG_TOP_K)
                for r in faiss_results:
                    doc = _normalize_faiss_result(r)
                    if doc not in all_results:
                        all_results.append(doc)
                if faiss_results:
                    logger.info("FAISS 检索命中 %d 条", len(faiss_results))
            except Exception as e:
                logger.warning("FAISS 检索出错: %s", e)

        # 第二通道：关键词匹配补充
        if len(all_results) < RAG_TOP_K:
            try:
                remaining = RAG_TOP_K - len(all_results)
                simple_results = rag_system.search(query, k=remaining)
                for result in simple_results:
                    doc = {"page_content": result["content"],
                           "source": result["metadata"].get("crop", "未知作物")}
                    if doc not in all_results:
                        all_results.append(doc)
                if simple_results:
                    logger.info("关键词检索补充 %d 条", len(simple_results))
            except Exception as e:
                logger.warning("关键词检索出错: %s", e)

    state.retrieved_docs = all_results if all_results else []
    return state
