"""LangGraph 工作流构建 — 多 Agent 调度"""

from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes.parse_input import parse_user_input
from .nodes.classify_intent import classify_intent
from .nodes.rag_retrieval import rag_retrieval_node
from .nodes.llm_response import general_response_node, llm_expert_answer, clarification_node
from .nodes.extract_tasks import extract_and_create_tasks_node
from .nodes.update_memory import update_long_memory

from knowledge.simple_agriculture_rag import SimpleAgricultureRAG
from knowledge.faiss_agriculture_rag import FAISSAgricultureRAG

# 多 Agent 调度中心
from .agents.orchestrator import AgentOrchestrator

# 全局单例
_orchestrator: AgentOrchestrator = None


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def _agent_dispatch_node(state: AgentState) -> AgentState:
    """Agent 调度节点：由 orchestrator 根据意图执行对应 Agent 逻辑"""
    orch = _get_orchestrator()
    # dispatch 内部会调用 agent.invoke(state)，直接修改 state
    orch.dispatch(state)
    return state


def build_agricultural_policy_agent(rag_system: SimpleAgricultureRAG,
                                     faiss_rag: FAISSAgricultureRAG = None):
    """构建多 Agent 协同工作流"""
    workflow = StateGraph(AgentState)
    orch = _get_orchestrator()

    # 添加节点（精简：只保留入口、分类、RAG、通用回答、调度、提取、记忆、澄清）
    workflow.add_node("parse_input", parse_user_input)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("agent_dispatch", _agent_dispatch_node)
    workflow.add_node("general_response", general_response_node)
    workflow.add_node("extract_tasks", extract_and_create_tasks_node)
    workflow.add_node("update_long_memory", update_long_memory)
    workflow.add_node("clarify", clarification_node)
    workflow.add_node("rag_retrieval", lambda s: rag_retrieval_node(s, rag_system, faiss_rag))
    workflow.add_node("generate_answer", llm_expert_answer)

    # 入口
    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "classify_intent")

    # 意图分类 → Agent 调度
    def route_after_classify(state: AgentState) -> str:
        if state.need_clarification:
            return "clarify"
        return "agent_dispatch"

    workflow.add_conditional_edges(
        source="classify_intent",
        path=route_after_classify,
        path_map={"agent_dispatch": "agent_dispatch", "clarify": "clarify"}
    )

    # Agent 调度后的路由
    def route_after_agent(state: AgentState) -> str:
        intent = state.intent_type or ""
        answered = bool(state.final_answer)

        # Agent 未产生回答 → 走 RAG + 通用 LLM 补全
        if not answered and not state.retrieved_docs:
            return "rag_retrieval"

        # 需要提取任务的意图
        task_intents = ("crop_selection", "planting_schedule", "planting_method",
                        "disease_prevention", "harvest_planning", "image_analysis",
                        "device_control", "crop_monitoring")
        if answered and intent in task_intents:
            return "extract_tasks"

        return "update_long_memory"

    workflow.add_conditional_edges(
        source="agent_dispatch",
        path=route_after_agent,
        path_map={
            "rag_retrieval": "rag_retrieval",
            "extract_tasks": "extract_tasks",
            "update_long_memory": "update_long_memory",
        }
    )

    # 剩余边
    workflow.add_edge("rag_retrieval", "general_response")
    workflow.add_edge("general_response", "extract_tasks")
    workflow.add_edge("clarify", "update_long_memory")
    workflow.add_edge("extract_tasks", "update_long_memory")
    workflow.add_edge("update_long_memory", END)

    return workflow.compile()
