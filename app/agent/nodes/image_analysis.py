"""图片分析节点 — 本地DL模型分类 + LLM增强生成"""

import json, logging, os, re

from ..state import AgentState
from ..config import DL_DEFAULT_MODEL, ENABLE_IMAGE_ANALYSIS, LLM_MODEL

logger = logging.getLogger(__name__)

LLM_ENHANCE_PROMPT = """你是一位农业病虫害诊断专家。根据图像识别结果，请提供详细的防治建议。

识别结果：
- 病害/问题：{class_name}
- 置信度：{confidence}

请以 JSON 格式返回（只输出 JSON，不要额外解释）：
{{
    "crop_type": "识别到的作物类型（如 小麦/番茄/水稻/未知）",
    "growth_stage": "作物生长阶段（如 苗期/生长期/开花期/成熟期/未知）",
    "severity": "严重程度（轻微/中等/严重）",
    "overall_health": "整体健康评估（良好/一般/较差）",
    "description": "该病害/虫害的详细描述和典型症状（100字内）",
    "recommendations": ["防治建议1", "防治建议2", "防治建议3"],
    "urgency": "紧急程度（立即处理/近期处理/持续观察）",
    "linked_action": "建议的设备操作（irrigate/ventilate/heat/shade/none）",
    "linked_params": {{"duration": 建议持续分钟数}}
}}"""


def image_analysis_node(state: AgentState) -> AgentState:
    if not state.has_image or not state.image_data:
        return state

    if not ENABLE_IMAGE_ANALYSIS:
        state.image_analysis_result = _fail(
            f"图片分析未启用。请在 .env 中配置 DL_DEFAULT_MODEL 或将模型权重放入 models/weights/。")
        return state

    try:
        state.image_analysis_result = _call_dl_model(state)
    except Exception as e:
        state.image_analysis_result = _fail(f"图片分析失败: {str(e)[:200]}")

    return state


def _call_dl_model(state: AgentState) -> dict:
    """使用本地DL模型进行病虫害分类，然后用LLM增强"""
    import base64
    from core.model_registry_factory import resolve_inference_model
    from core.model_executor import ModelExecutor
    from models.base import ModelInput

    registry, model_id = resolve_inference_model(DL_DEFAULT_MODEL)
    executor = ModelExecutor(registry)

    image_bytes = base64.b64decode(state.image_data)
    model_input = ModelInput(image_bytes=image_bytes, top_k=3)
    result = executor.infer_sync(model_id, model_input)

    if not result.success:
        raise Exception(f"模型推理失败: {result.error_code}")

    predictions = [
        {"class_name": p.class_name, "confidence": round(p.confidence, 4)}
        for p in result.predictions
    ]

    # 用LLM增强top-1预测结果
    top = predictions[0]
    enhanced = _invoke_llm_enhance(top["class_name"], top["confidence"], state)

    return {
        "model_id": result.model_id,
        "predictions": predictions,
        "inference_time_ms": result.inference_time_ms,
        "crop_type": enhanced.get("crop_type", ""),
        "growth_stage": enhanced.get("growth_stage", ""),
        "detected_issues": [{
            "type": "病害" if "病" in top["class_name"] else "虫害",
            "name": top["class_name"],
            "severity": enhanced.get("severity", "中等"),
            "confidence": top["confidence"],
            "description": enhanced.get("description", ""),
        }],
        "overall_health": enhanced.get("overall_health", "一般"),
        "recommendations": enhanced.get("recommendations", []),
        "urgency": enhanced.get("urgency", "近期处理"),
        "llm_advice": enhanced.get("advice", ""),
        # 硬件联动字段：图像识别结果可触发设备操作
        "linked_action": enhanced.get("linked_action", "none"),
        "linked_params": enhanced.get("linked_params", {}),
    }


def _invoke_llm_enhance(class_name: str, confidence: float, state: AgentState) -> dict:
    """调用LLM根据分类结果生成防治建议，解析结构化JSON"""
    from ..utils import _get_llm
    try:
        prompt = LLM_ENHANCE_PROMPT.format(class_name=class_name, confidence=confidence)
        llm = _get_llm()
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        # 使用 _parse() 从 LLM 响应中提取结构化 JSON
        parsed = _parse(text)
        if parsed.get("error"):
            logger.warning("LLM增强JSON解析失败，使用原始文本: %s", parsed["error"])
            return {
                "advice": text,
                "description": "",
                "severity": "中等",
                "overall_health": "一般",
                "recommendations": [],
                "urgency": "近期处理",
                "linked_action": "none",
                "linked_params": {},
            }
        return {
            "advice": text,
            "crop_type": parsed.get("crop_type", ""),
            "growth_stage": parsed.get("growth_stage", ""),
            "severity": parsed.get("severity", "中等"),
            "overall_health": parsed.get("overall_health", "一般"),
            "description": parsed.get("description", ""),
            "recommendations": parsed.get("recommendations", []),
            "urgency": parsed.get("urgency", "近期处理"),
            "linked_action": parsed.get("linked_action", "none"),
            "linked_params": parsed.get("linked_params", {}),
        }
    except Exception as e:
        logger.warning("LLM增强失败: %s", e)
        return {
            "advice": f"模型识别为: {class_name}（置信度: {confidence}）",
            "severity": "中等", "overall_health": "一般",
            "recommendations": [], "urgency": "近期处理",
            "linked_action": "none", "linked_params": {},
        }


def image_analysis_answer_node(state: AgentState) -> AgentState:
    a = state.image_analysis_result or {}
    docs = state.retrieved_docs or []
    parts = []

    if a.get("crop_type"):
        parts.append(f"🌾 **识别作物**: {a['crop_type']}")
    if a.get("growth_stage"):
        parts.append(f"📈 **生长阶段**: {a['growth_stage']}")
    if a.get("overall_health"):
        parts.append(f"💚 **整体健康**: {a['overall_health']}")

    for issue in a.get("detected_issues", []):
        if not parts or parts[-1] != "\n🔍 **检测到的问题**:":
            parts.append("\n🔍 **检测到的问题**:")
        s = issue.get("severity", "")
        emoji = {"轻微": "⚪", "中等": "🟡", "严重": "🔴"}.get(s, "⚪")
        parts.append(f"  {emoji} **{issue.get('name', '未知')}** ({issue.get('type','')}) "
                    f"| {s} | 置信度 {issue.get('confidence',0):.0%}")

    if a.get("recommendations"):
        parts.append("\n💡 **处理建议**:")
        for i, r in enumerate(a["recommendations"], 1):
            parts.append(f"  {i}. {r}")

    if docs:
        parts.append("\n📚 **相关知识**:")
        for d in docs[:2]:
            parts.append(f"  • {d['page_content'][:100]}...")

    urgency = a.get("urgency", "")
    if urgency:
        parts.append(f"\n{'🚨' if '立即' in urgency else '⚠️' if '近期' in urgency else '👁️'} **紧急程度**: {urgency}")

    if a.get("error"):
        parts.append(f"\n⚠️ **分析出错**: {a['error']}")

    # ── 硬件联动：将图像识别触发的设备操作写入 state ──
    linked_action = a.get("linked_action", "none")
    linked_params = a.get("linked_params", {})
    if linked_action and linked_action != "none":
        action_labels = {
            "irrigate": "灌溉", "ventilate": "通风", "heat": "加热",
            "shade": "遮阳", "light": "补光",
        }
        action_label = action_labels.get(linked_action, linked_action)
        parts.append(f"\n🔧 **建议设备操作**: {action_label}（参数: {linked_params}）")
        # 存入 pending_action，由 orchestrator 路由到 DeviceAgent 自动执行
        state.pending_action = {
            "device_id": "",   # 由 DeviceAgent 动态发现匹配设备
            "command": "start",
            "params": linked_params,
            "reason": f"图像识别联动: {a.get('detected_issues', [{}])[0].get('name', '未知病害')}",
            "source": "image_analysis",
            "linked_action": linked_action,
        }

    state.final_answer = "\n".join(parts) if parts else "图片分析未产生结果。"
    state.image_data = None
    state.has_image = False
    return state


def _parse(content: str) -> dict:
    # 提取 ```json ... ``` 代码块
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    text = m.group(1).strip() if m else content.strip()

    # 去除首尾非 JSON 字符（模型可能在 JSON 前后加了解释文字）
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

    # JSON 截断恢复：尝试补全
    truncated = text.strip()
    if not truncated.endswith('}'):
        # 找到最后一个完整的键值对或数组元素
        for fix in ['}]}]}', '}]}', '}', '"]}', '"}']:
            candidate = truncated + fix
            try:
                result = json.loads(candidate)
                if result.get("crop_type"):
                    return result
            except json.JSONDecodeError:
                continue

    # 最后兜底：提取已识别的文本信息
    crop = ""
    m = re.search(r'"crop_type"\s*:\s*"([^"]+)"', text)
    if m:
        crop = m.group(1)
    return {"error": "JSON 解析失败（可能被截断）", "crop_type": crop, "growth_stage": "",
            "detected_issues": [], "overall_health": "",
            "recommendations": [text[:500]], "urgency": ""}


def _fail(msg: str) -> dict:
    return {"error": msg, "crop_type": "", "growth_stage": "",
            "detected_issues": [], "overall_health": "", "recommendations": [], "urgency": ""}
