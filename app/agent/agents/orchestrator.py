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

# 否定/完成/状态前缀 — 当关键词紧邻这些词时，不应触发操作意图
_NEGATION_PATTERNS = [
    r'(未|没|不|已|已经|刚|刚刚|之前|上次|昨天|今天|前几天|之前已经)\s*{kw}',
    r'{kw}\s*(过|了|完|好|完毕|结束|完成)',
]

# 任务创建语境 — 用户说"添加/创建/设置...任务/提醒"时，不应触发 device_control
# 即使句中包含"浇水""施肥"等设备关键词，用户意图是创建任务而非执行设备
_TASK_CREATION_PATTERNS = [
    r'(添加|创建|新建|设置|帮我建|帮我加|帮忙建|帮忙加)\s*.*?\s*(任务|提醒|待办)',
]


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
            logger.info("调度: 需要澄清，跳过 Agent 匹配")
            return "clarify"

        skip_intents = ("greeting", "thanks", "farewell", "identity", "function", "unclear")
        if intent in skip_intents:
            logger.info("调度: intent=%s 属于跳过类型，直接走 RAG 检索", intent)
            return "rag_retrieval"

        secondary = self._detect_secondary(question, intent)
        targets = [intent] + secondary if secondary else [intent]
        logger.info("调度: targets=%s", targets)

        if len(targets) == 1:
            return self._run_single(state, targets[0])
        else:
            logger.info("调度: 复合意图 → 并行执行 %d 个 Agent", len(targets))
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
            logger.info("Agent 执行: %s → intent=%s", agent.name, intent)
            agent.invoke(state)
            answer = state.final_answer or ""
            logger.info(
                "Agent 完成: %s final_answer=%s answer_len=%d",
                agent.name,
                bool(answer),
                len(answer),
            )
        else:
            logger.warning("未找到可处理 intent=%s 的 Agent，回退 RAG 检索", intent)
        next_node = self._next_node(intent)
        logger.info("下一节点: %s (intent=%s)", next_node, intent)
        return next_node

    def _run_parallel(self, state: AgentState, intents: List[str]) -> str:
        results: Dict[str, str] = {}
        primary_intent = intents[0]

        primary_agent = self._find_agent(primary_intent)
        if primary_agent:
            logger.info("Agent 执行(主): %s → intent=%s", primary_agent.name, primary_intent)
            primary_agent.invoke(state)
            results[primary_intent] = state.final_answer or ""
            answer = state.final_answer or ""
            logger.info("Agent 完成(主): %s answer_len=%d", primary_agent.name, len(answer))

        if len(intents) > 1:
            logger.info("并行执行 %d 个副 Agent: %s", len(intents) - 1, intents[1:])
            import copy
            secondary_states = []  # 收集副 Agent 的 state 用于回并 messages
            def _run_secondary(intent: str) -> Tuple[str, str]:
                agent = self._find_agent(intent)
                if not agent:
                    logger.warning("副 Agent 未找到: intent=%s", intent)
                    return (intent, "")
                s_copy = copy.deepcopy(state)
                try:
                    logger.info("Agent 执行(副): %s → intent=%s", agent.name, intent)
                    agent.invoke(s_copy)
                    ans = s_copy.final_answer or ""
                    logger.info(
                        "Agent 完成(副): %s intent=%s answer_len=%d",
                        agent.name,
                        intent,
                        len(ans),
                    )
                    secondary_states.append(s_copy)
                    return (intent, ans)
                except Exception as e:
                    logger.warning("副 Agent %s 失败: %s", intent, e)
                    return (intent, "")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_run_secondary, i): i for i in intents[1:]}
                for f in concurrent.futures.as_completed(futures):
                    intent, answer = f.result()
                    if answer:
                        results[intent] = answer
                # 添加超时：等待剩余未完成的任务（最多120秒）
                concurrent.futures.wait(futures, timeout=120)

            # 将副 Agent 的 messages 回并到主 state（保留 merge 后的完整对话）
            for s in secondary_states:
                if s.messages and state.messages:
                    existing_ids = set(id(m) for m in state.messages)
                    for msg in s.messages:
                        if id(msg) not in existing_ids:
                            state.messages.append(msg)

        state.final_answer = self._merge_answers(results, primary_intent)
        logger.info("回答合并完成: %d 个 Agent 参与", len(results))
        return self._next_node(primary_intent)

    def _merge_answers(self, results: Dict[str, str], primary: str) -> str:
        if len(results) <= 1:
            return results.get(primary, "")

        section_names = {
            "crop_selection": "🌱 种植建议", "planting_schedule": "📅 种植时间",
            "planting_method": "🌾 种植方法", "disease_prevention": "🩺 病虫害分析",
            "weather_query": "🌤 气象信息", "finance_query": "💰 财务信息",
            "policy_query": "📜 政策信息", "progress_tracking": "📋 进度跟踪",
            "reminder_setup": "⏰ 提醒", "field_management": "📍 地块信息",
            "harvest_planning": "🌾 收获规划", "image_analysis": "🔍 图片分析",
        }

        primary_answer = results.get(primary, "")
        # 检查 primary answer 是否已包含 weather 相关信息
        has_weather = any(kw in (primary_answer or "") for kw in ["天气", "气温", "降水", "湿度", "风力", "🌤", "气象"])

        sections = []
        for intent, answer in results.items():
            if not answer:
                continue
            if intent == primary:
                sections.insert(0, answer)
            else:
                # 如果 primary 已包含天气，跳过 weather_query 副 section
                if intent == "weather_query" and has_weather:
                    logger.info("合并回答: primary 已包含天气信息，跳过 weather_query 副 section")
                    continue
                name = section_names.get(intent, intent)
                sections.append(f"\n---\n### {name}\n{answer}")
        return "\n".join(sections)

    def _detect_secondary(self, question: str, primary: str) -> List[str]:
        """检测次级意图，排除否定/完成/任务创建语境中的误匹配"""
        import re
        # 任务创建语境检测："添加浇水任务"不应触发 device_control
        is_task_creation = any(
            re.search(p, question) for p in _TASK_CREATION_PATTERNS
        )
        found = []
        for intent, keywords in CROSS_DOMAIN_KEYWORDS.items():
            if intent == primary:
                continue
            # 任务创建语境下，跳过 device_control（用户想创建任务而非执行设备）
            if is_task_creation and intent == "device_control":
                logger.info("次级意图过滤(任务创建语境): 跳过 device_control")
                continue
            matched = []
            for kw in keywords:
                if kw not in question:
                    continue
                # 检查是否处于否定/完成语境中
                negated = False
                for pattern in _NEGATION_PATTERNS:
                    if re.search(pattern.format(kw=re.escape(kw)), question):
                        negated = True
                        break
                if negated:
                    logger.info("次级意图过滤(否定语境): intent=%s keyword=%s", intent, kw)
                    continue
                matched.append(kw)
            if matched:
                logger.info("检测到次级意图: intent=%s (命中关键词: %s)", intent, matched)
                found.append(intent)
        if found:
            logger.info("次级意图汇总: %s (主意图=%s)", found[:2], primary)
        return found[:2]

    def _find_agent(self, intent: str) -> Optional[BaseAgent]:
        for agent in self.agents.values():
            if agent.can_handle(intent):
                logger.info("Agent 匹配: intent=%s → %s (%s)", intent, agent.name, agent.description)
                return agent
        logger.warning("Agent 匹配失败: 没有 Agent 能处理 intent=%s", intent)
        return None

    def _next_node(self, intent: str) -> str:
        task_intents = ("crop_selection", "planting_schedule", "planting_method",
                       "disease_prevention", "harvest_planning", "image_analysis")
        return "extract_tasks" if intent in task_intents else "update_long_memory"

    def list_agents(self) -> List[Dict]:
        return [{"name": a.name, "description": a.description, "intents": a.intent_types}
                for a in self.agents.values()]
