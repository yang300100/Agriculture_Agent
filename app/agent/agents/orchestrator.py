"""Agent 调度中心 — 路由 + 复合意图并行 + Agent 间互调 + 回答合并"""

import logging
import concurrent.futures
from typing import Dict, List, Optional, Tuple
from ..state import AgentState
from .base import BaseAgent
from .planting_agent import PlantingAgent
from .disease_agent import DiseaseAgent
from .weather_agent import WeatherAgent
from .finance_agent import FinanceAgent
from .farming_agent import FarmingAgent
from .device_agent import DeviceAgent
from .crop_monitor_agent import CropMonitorAgent

logger = logging.getLogger(__name__)

CROSS_DOMAIN_KEYWORDS = {
    "disease_prevention": ["病虫害", "叶子发黄", "长斑", "烂根", "虫害", "病害", "打药", "喷药", "防治"],
    "weather_query": ["天气", "下雨", "刮风", "气温", "霜冻", "适合喷药"],
    "finance_query": ["花了多少", "赚了多少", "成本", "收入", "价格", "行情", "记账", "多少钱"],
    "progress_tracking": ["进度", "阶段", "该做什么", "生长情况"],
    "reminder_setup": ["提醒", "别忘了", "记得"],
    "device_control": ["浇水", "灌溉", "施肥", "通风", "补光", "加热", "降温", "开关", "启动", "停止"],
    "crop_monitoring": ["监控作物", "拍照分析", "查看长势", "摄像", "巡检", "监测"],
}


class AgentOrchestrator:
    """多 Agent 调度中心"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self._register(PlantingAgent())
        self._register(DiseaseAgent())
        self._register(WeatherAgent())
        self._register(FinanceAgent())
        self._register(FarmingAgent())
        self._register(DeviceAgent())
        self._register(CropMonitorAgent())
        # 注入调度中心引用
        for agent in self.agents.values():
            agent._orchestrator = self

    def _register(self, agent: BaseAgent):
        self.agents[agent.name] = agent

    # ── 主入口 ─────────────────────────────────────

    def dispatch(self, state: AgentState) -> str:
        question = state.user_question or ""
        intent = state.intent_type or ""

        if state.need_clarification:
            return "clarify"

        skip_intents = ("greeting", "thanks", "farewell", "identity", "function", "unclear")
        if intent in skip_intents:
            return "rag_retrieval"

        secondary = self._detect_secondary(question, intent)
        targets = [intent] + secondary if secondary else [intent]
        logger.info("调度: targets=%s", targets)

        if len(targets) == 1:
            return self._run_single(state, targets[0])
        else:
            return self._run_parallel(state, targets)

    # ── Agent 间互调 ──────────────────────────────

    def interop_call(self, intent: str, state: AgentState) -> Optional[str]:
        """Agent 间互调：一个 Agent 向另一个 Agent 请求信息片段"""
        agent = self._find_agent(intent)
        if not agent:
            return None
        try:
            # 复制 state 避免污染
            import copy
            s_copy = copy.deepcopy(state)
            agent.invoke(s_copy)
            return s_copy.final_answer
        except Exception as e:
            logger.warning("Agent 间互调失败 %s→%s: %s", intent, agent.name, e)
            return None

    # ── 内部方法 ────────────────────────────────────

    def _run_single(self, state: AgentState, intent: str) -> str:
        agent = self._find_agent(intent)
        if agent:
            agent.invoke(state)
        return self._next_node(intent)

    def _run_parallel(self, state: AgentState, intents: List[str]) -> str:
        results: Dict[str, str] = {}
        primary_intent = intents[0]

        primary_agent = self._find_agent(primary_intent)
        if primary_agent:
            primary_agent.invoke(state)
            results[primary_intent] = state.final_answer or ""

        if len(intents) > 1:
            import copy
            def _run_secondary(intent: str) -> Tuple[str, str]:
                agent = self._find_agent(intent)
                if not agent:
                    return (intent, "")
                s_copy = copy.deepcopy(state)
                try:
                    agent.invoke(s_copy)
                    return (intent, s_copy.final_answer or "")
                except Exception as e:
                    logger.warning("副 Agent %s 失败: %s", intent, e)
                    return (intent, "")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_run_secondary, i): i for i in intents[1:]}
                for f in concurrent.futures.as_completed(futures):
                    intent, answer = f.result()
                    if answer:
                        results[intent] = answer

        state.final_answer = self._merge_answers(results, primary_intent)
        return self._next_node(primary_intent)

    def _merge_answers(self, results: Dict[str, str], primary: str) -> str:
        if len(results) <= 1:
            return results.get(primary, "")

        section_names = {
            "crop_selection": "🌱 种植建议", "planting_schedule": "📅 种植时间",
            "planting_method": "🌾 种植方法", "disease_prevention": "🩺 病虫害分析",
            "weather_query": "🌤 气象信息", "finance_query": "💰 财务信息",
            "progress_tracking": "📋 进度跟踪",
            "reminder_setup": "⏰ 提醒", "field_management": "📍 地块信息",
            "harvest_planning": "🌾 收获规划", "image_analysis": "🔍 图片分析",
        }

        sections = []
        for intent, answer in results.items():
            if not answer:
                continue
            if intent == primary:
                sections.insert(0, answer)
            else:
                name = section_names.get(intent, intent)
                sections.append(f"\n---\n### {name}\n{answer}")
        return "\n".join(sections)

    def _detect_secondary(self, question: str, primary: str) -> List[str]:
        found = []
        for intent, keywords in CROSS_DOMAIN_KEYWORDS.items():
            if intent == primary:
                continue
            if any(kw in question for kw in keywords):
                found.append(intent)
        return found[:2]

    def _find_agent(self, intent: str) -> Optional[BaseAgent]:
        for agent in self.agents.values():
            if agent.can_handle(intent):
                return agent
        return None

    def _next_node(self, intent: str) -> str:
        task_intents = ("crop_selection", "planting_schedule", "planting_method",
                       "disease_prevention", "harvest_planning", "image_analysis")
        return "extract_tasks" if intent in task_intents else "update_long_memory"

    def list_agents(self) -> List[Dict]:
        return [{"name": a.name, "description": a.description, "intents": a.intent_types}
                for a in self.agents.values()]
