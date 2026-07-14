"""图片分析节点 — 本地DL模型分类 + LLM增强生成"""

import json, logging, os, re

from ..state import AgentState
from ..config import DL_DEFAULT_MODEL, ENABLE_IMAGE_ANALYSIS, LLM_MODEL

logger = logging.getLogger(__name__)

LLM_ENHANCE_PROMPT = """你是一位农业病虫害诊断专家。根据图像识别结果，请提供详细的防治建议。

识别结果：
- 病害/问题：{class_name}
- 置信度：{confidence}

请提供：
1. 该病害/虫害的详细描述和典型症状
2. 具体防治方法和用药建议
3. 预防措施和后续管理
4. 对当前作物生长阶段的影响评估"""


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
    from core.model_registry_factory import get_model_registry
    from core.model_executor import ModelExecutor
    from models.base import ModelInput

    registry = get_model_registry()
    executor = ModelExecutor(registry)

    model_id = DL_DEFAULT_MODEL
    if not model_id:
        models = registry.list_models()
        if not models:
            raise Exception("没有可用的DL模型。请将模型权重放入 models/weights/ 目录。")
        model_id = models[0].model_id

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
    }


def _invoke_llm_enhance(class_name: str, confidence: float, state: AgentState) -> dict:
    """调用LLM根据分类结果生成防治建议"""
    from ..utils import _get_llm
    try:
        prompt = LLM_ENHANCE_PROMPT.format(class_name=class_name, confidence=confidence)
        llm = _get_llm()
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        # 尝试从LLM回答中提取结构化信息
        return {
            "advice": text,
            "description": "",
            "severity": "中等",
            "overall_health": "一般",
            "recommendations": [],
            "urgency": "近期处理",
        }
    except Exception as e:
        logger.warning("LLM增强失败: %s", e)
        return {"advice": f"模型识别为: {class_name}（置信度: {confidence}）"}


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
