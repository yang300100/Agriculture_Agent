"""病虫害诊断 Agent — 图片分析、症状匹配、防治方案，自动联动气象 Agent"""

import logging
from .base import BaseAgent
from ..state import AgentState

logger = logging.getLogger(__name__)

# 需要喷药防治的病害 → 自动问天气
SPRAY_DISEASES = ["赤霉病", "锈病", "白粉病", "晚疫病", "早疫病", "叶斑病", "菌核病",
                  "稻瘟病", "纹枯病", "大斑病", "黑穗病", "霜霉病", "炭疽病"]
SPRAY_KEYWORDS = ["喷施", "喷雾", "喷药", "打药", "喷洒", "杀菌剂", "杀虫剂"]


class DiseaseAgent(BaseAgent):
    name = "disease"
    description = "植保专家，负责病虫害图片诊断、症状识别、防治方案推荐，自动联动气象判断施药时机"
    system_prompt = """你是一位植物保护专家，专精农作物病虫害诊断与防治。
你能通过图片或症状描述识别病虫害类型，评估严重程度，
推荐科学的防治方案（生物/化学/农业防治），
并结合气象数据预判病害发生风险，判断当前是否适合喷药，给出最佳施药窗口。"""
    intent_types = ["disease_prevention", "image_analysis"]

    def invoke(self, state: AgentState) -> AgentState:
        from ..nodes.image_analysis import image_analysis_node, image_analysis_answer_node

        if state.intent_type == "image_analysis":
            state = image_analysis_node(state)
            state = image_analysis_answer_node(state)
            return self._append_weather_if_spray(state)

        # disease_prevention → RAG 增强
        from ..nodes.rag_retrieval import rag_retrieval_node
        from ..nodes.llm_response import llm_expert_answer
        from knowledge.simple_agriculture_rag import SimpleAgricultureRAG
        from knowledge.faiss_agriculture_rag import FAISSAgricultureRAG
        rag = SimpleAgricultureRAG()
        faiss = FAISSAgricultureRAG() if FAISSAgricultureRAG().is_available else None
        state = rag_retrieval_node(state, rag, faiss)
        state = llm_expert_answer(state)
        return self._append_weather_if_spray(state)

    def _append_weather_if_spray(self, state: AgentState) -> AgentState:
        """如果防治方案涉及喷药，自动查询气象 Agent 判断施药窗口"""
        answer = state.final_answer or ""
        crop = state.short_term_facts.get("crop", "")
        question = state.user_question or ""

        # 判断是否需要喷药建议
        needs_spray = any(d in answer for d in SPRAY_DISEASES) or any(kw in answer for kw in SPRAY_KEYWORDS)
        if not needs_spray:
            return state

        # 调用气象 Agent
        try:
            weather_state = AgentState(
                messages=[], user_profile=state.user_profile,
                short_term_facts=state.short_term_facts,
                intent_type="weather_query",
                user_question=f"{crop} 当前是否适合喷药",
            )
            wx_info = self.call_colleague("weather_query", weather_state)
            if wx_info:
                # 提取关键气象信息（避免整篇天气报告）
                lines = wx_info.split("\n")
                spray_lines = []
                capture = False
                for line in lines:
                    if any(kw in line for kw in ["施药", "喷药", "适宜", "风险", "最佳窗口", "风力", "风速"]):
                        capture = True
                    if capture and line.strip():
                        spray_lines.append(line)
                    if capture and len(spray_lines) > 6:
                        break
                if spray_lines:
                    state.final_answer = answer + "\n\n---\n### 🌤 施药气象建议（自动联动）\n" + "\n".join(spray_lines)
                elif len(wx_info) < 300:
                    state.final_answer = answer + "\n\n---\n### 🌤 气象参考\n" + wx_info
            else:
                state.final_answer = answer + "\n\n💡 *喷药前请查看当地天气预报，避开降雨和大风天气。*"
        except Exception as e:
            logger.warning("病虫害→气象联动失败: %s", e)
            state.final_answer = answer + "\n\n💡 *喷药前请确认天气状况，避免降雨和大风天气。*"

        return state
