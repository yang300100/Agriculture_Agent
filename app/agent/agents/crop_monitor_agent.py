"""作物监测 Agent — 定时拍摄 + Vision 模型分析 + 自主决策执行

独立于用户对话流程，为定时任务提供 analyze_image() 接口。
也可通过 intent="crop_monitoring" 响应用户聊天请求。
"""

import json
import logging
import os
import re
import requests
from typing import Dict, Any, Optional, List

from .base import BaseAgent
from ..state import AgentState
from ..config import (
    VISION_MODEL, VISION_API_KEY, VISION_BASE_URL, VISION_TEMPERATURE,
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TEMPERATURE,
)

VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "4096"))
logger = logging.getLogger(__name__)

# ── 作物监测专用 Prompt ─────────────────────────

MONITOR_PROMPT = """你是农业作物健康监测专家，正在分析一张来自农田摄像头的定期巡检照片。

请仔细观察画面中的作物，返回以下 JSON：

{
    "crop_type": "作物名称（识别不出的写 unknown）",
    "growth_stage": "seedling/vegetative/flowering/fruiting/mature/unknown",
    "health_assessment": {
        "overall": "excellent/good/fair/poor",
        "nutrient_status": "adequate/deficient-N/deficient-P/deficient-K/unknown",
        "water_status": "adequate/drought-stressed/overwatered/unknown",
        "pest_presence": "none/suspected/confirmed",
        "pest_detail": "害虫名称（无则留空）",
        "disease_presence": "none/suspected/confirmed",
        "disease_detail": "病害名称（无则留空）"
    },
    "issues_found": [
        {
            "type": "nutrient/pest/disease/water/other",
            "name": "具体问题名称",
            "severity": "mild/moderate/severe",
            "confidence": 0.85,
            "description": "中文症状描述"
        }
    ],
    "recommended_actions": [
        {
            "action": "irrigate/fertigate/alert/none",
            "urgency": "immediate/today/this_week/routine",
            "detail": "具体操作说明（中文）"
        }
    ],
    "summary": "一段中文总结，描述当前作物状况和关键发现"
}

规则：
- 如果叶片发黄、老叶先黄 → nutrient_status 为 deficient-N，推荐 fertilize
- 如果叶片萎蔫、土壤干裂 → water_status 为 drought-stressed，推荐 irrigate
- 如果看到虫洞、虫粪、成虫 → pest_presence 为 confirmed，推荐 alert
- 如果叶片有斑点、白粉、霉层 → disease_presence 为 confirmed，推荐 alert
- 如果画面中无作物或无法判断 → recommended_actions 为空数组，summary 注明
- urgency: 严重→immediate，中等→today，轻微→this_week，无问题→routine"""


class CropMonitorAgent(BaseAgent):
    """作物监测 Agent — 摄像头定时巡检 + Vision AI 分析

    两种调用方式：
    1. 独立调用（定时任务用）: agent.analyze_image(image_base64, mime_type, user_context)
    2. Agent 流程内（用户对话用）: intent="crop_monitoring" → invoke()
    """

    name = "crop_monitor"
    description = "定时摄像头拍照巡检，AI 分析作物健康状况并自主决策"
    system_prompt = "你是基于摄像头的农作物健康监测专家。"
    intent_types = ["crop_monitoring"]

    # ── 独立接口（定时任务用）────────────────────

    def analyze_image(self, image_base64: str, mime_type: str = "image/jpeg",
                      user_context: Dict = None) -> Dict:
        """分析单张照片，返回结构化结果 + 推荐操作

        这是定时任务的核心入口，不依赖 AgentState。

        Args:
            image_base64: base64 编码的图片数据（不含 data:xxx;base64, 前缀）
            mime_type:   图片 MIME 类型
            user_context: 可选上下文，如 {"username":"123","device_id":"cam_01","crop":"番茄"}

        Returns:
            {
                "success": bool,
                "analysis": { ... Vision 模型返回的完整 JSON },
                "error": str (仅失败时有),
            }
        """
        try:
            # 构建用户提示文本
            extra = ""
            if user_context:
                crop = user_context.get("crop", "")
                if crop:
                    extra = f"\n当前种植作物: {crop}。"
                loc = user_context.get("location", "")
                if loc:
                    extra += f"\n拍摄位置: {loc}。"

            result = self._call_vision_api(image_base64, mime_type, extra)
            return {"success": True, "analysis": result}

        except Exception as e:
            logger.error("CropMonitorAgent 分析失败: %s", e)
            return {
                "success": False,
                "error": str(e),
                "analysis": {
                    "crop_type": "unknown", "growth_stage": "unknown",
                    "health_assessment": {"overall": "unknown"},
                    "issues_found": [],
                    "recommended_actions": [],
                    "summary": f"分析失败: {e}",
                },
            }

    # ── Agent 流程入口（用户对话用）──────────────

    def invoke(self, state: AgentState) -> AgentState:
        """处理用户的作物监测请求（来自聊天对话）"""
        if not state.has_image or not state.image_data:
            state.final_answer = "📷 请上传一张农作物照片，我来帮你分析。"
            return state

        if not os.getenv("VISION_MODEL"):
            state.final_answer = (
                "⚠️ 图片分析未启用。\n\n"
                f"当前 LLM_MODEL={LLM_MODEL} 可能不支持图片。\n"
                "请在 .env 中配置 VISION_MODEL 为支持多模态的模型。"
            )
            return state

        try:
            extra = state.user_question or "请分析这张农作物监测照片"
            result = self._call_vision_api(state.image_data, state.image_mime_type or "image/jpeg", extra)
            state.image_analysis_result = result
            state.final_answer = self._format_result(result)
            state.image_data = None
            state.has_image = False
        except Exception as e:
            logger.error("CropMonitorAgent invoke 失败: %s", e)
            state.final_answer = f"❌ 作物监测分析失败: {e}"

        return state

    # ── Vision API 调用 ──────────────────────────

    def _call_vision_api(self, image_base64: str, mime_type: str,
                         extra_text: str = "") -> Dict:
        """调用 Vision 模型分析图片（复用 image_analysis.py 模式）"""
        url = f"{VISION_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {VISION_API_KEY}",
            "Content-Type": "application/json",
        }

        prompt_text = MONITOR_PROMPT
        if extra_text:
            prompt_text = MONITOR_PROMPT + "\n" + extra_text

        payload = {
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"}},
                ],
            }],
            "max_tokens": VISION_MAX_TOKENS,
            "temperature": VISION_TEMPERATURE,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        if resp.status_code != 200:
            raise Exception(f"Vision API {resp.status_code}: {resp.text[:300]}")

        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.warning("Vision API 响应解析失败: %s", e)
            raise Exception(f"Vision API 返回格式异常: {resp.text[:200]}")

        if not content or not content.strip():
            raise Exception("Vision API 返回了空内容")

        return self._parse_json(content)

    # ── JSON 解析（复用 image_analysis.py 逻辑）──

    def _parse_json(self, content: str) -> dict:
        """从 LLM 响应中提取 JSON，支持截断恢复"""
        # 提取 ```json ... ``` 代码块
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        text = m.group(1).strip() if m else content.strip()

        # 去除首尾非 JSON 字符
        if text and text[0] != '{':
            idx = text.find('{')
            if idx >= 0:
                text = text[idx:]
        if text and text[-1] != '}':
            idx = text.rfind('}')
            if idx >= 0:
                text = text[:idx + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # JSON 截断恢复
        truncated = text.strip()
        if not truncated.endswith('}'):
            for fix in ['}]}]}', '}]}', '}', '"]}', '"}']:
                candidate = truncated + fix
                try:
                    result = json.loads(candidate)
                    if result.get("crop_type"):
                        return result
                except json.JSONDecodeError:
                    continue

        # 兜底
        crop = ""
        m = re.search(r'"crop_type"\s*:\s*"([^"]+)"', text)
        if m:
            crop = m.group(1)
        return {
            "crop_type": crop, "growth_stage": "unknown",
            "health_assessment": {"overall": "unknown"},
            "issues_found": [],
            "recommended_actions": [],
            "summary": content[:300] if content else "解析失败",
        }

    # ── 结果格式化 ───────────────────────────────

    def _format_result(self, a: dict) -> str:
        """将分析结果格式化为用户可读的 Markdown 文本"""
        parts = []

        if a.get("crop_type"):
            parts.append(f"🌾 **作物**: {a['crop_type']}")
        if a.get("growth_stage"):
            stage_map = {
                "seedling": "🌱 苗期", "vegetative": "🌿 生长期",
                "flowering": "🌸 花期", "fruiting": "🍎 结果期",
                "mature": "🌽 成熟期",
            }
            stage_label = stage_map.get(a["growth_stage"], a["growth_stage"])
            parts.append(f"📈 **阶段**: {stage_label}")

        health = a.get("health_assessment", {})
        if not isinstance(health, dict):
            health = {}
        if health.get("overall"):
            emoji = {"excellent": "💚", "good": "💛", "fair": "🧡", "poor": "❤️"}.get(
                health["overall"], "⚪")
            parts.append(f"{emoji} **整体**: {health['overall']}")

        # 详细评估
        if health:
            detail_parts = []
            if health.get("nutrient_status"):
                n = health["nutrient_status"]
                n_emoji = "✅" if "adequate" in n else "⚠️"
                detail_parts.append(f"{n_emoji} 养分: {n}")
            if health.get("water_status"):
                w = health["water_status"]
                w_emoji = "✅" if "adequate" in w else "⚠️"
                detail_parts.append(f"{w_emoji} 水分: {w}")
            if health.get("pest_presence"):
                p = health["pest_presence"]
                p_emoji = "✅" if p == "none" else "🔴"
                detail_parts.append(f"{p_emoji} 虫害: {p}")
            if health.get("disease_presence"):
                d = health["disease_presence"]
                d_emoji = "✅" if d == "none" else "🔴"
                detail_parts.append(f"{d_emoji} 病害: {d}")
            if detail_parts:
                parts.append(" | ".join(detail_parts))

        # 发现的问题
        issues = a.get("issues_found", [])
        if issues:
            parts.append("\n🔍 **检测到的问题**:")
            for issue in issues:
                s = issue.get("severity", "")
                emoji = {"mild": "⚪", "moderate": "🟡", "severe": "🔴"}.get(s, "⚪")
                parts.append(
                    f"  {emoji} **{issue.get('name', '?')}** "
                    f"({issue.get('type', '')}) → {issue.get('description', '')}"
                )

        # 推荐操作
        actions = a.get("recommended_actions", [])
        if actions:
            parts.append("\n💡 **推荐操作**:")
            for act in actions:
                action_type = act.get("action", "")
                action_label = {
                    "irrigate": "💧 灌溉", "fertigate": "🌱 施肥",
                    "alert": "🚨 告警", "none": "✅ 无需操作",
                }.get(action_type, action_type)
                urgency = act.get("urgency", "")
                urgency_label = {
                    "immediate": "🔴 立即", "today": "🟡 今日",
                    "this_week": "🟢 本周", "routine": "🔵 常规",
                }.get(urgency, urgency)
                parts.append(f"  {action_label} | {urgency_label} | {act.get('detail', '')}")

        # 总结
        summary = a.get("summary", "")
        if summary:
            parts.append(f"\n📋 **总结**: {summary}")

        if a.get("error"):
            parts.append(f"\n⚠️ **分析出错**: {a['error']}")

        return "\n".join(parts) if parts else "图片分析未产生结果。"
