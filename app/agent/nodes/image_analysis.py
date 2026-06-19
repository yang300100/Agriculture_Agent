"""图片分析节点 — 直调多模态 API，不经过 LangChain"""

import json, logging, os, re, requests

from ..state import AgentState
from ..config import VISION_MODEL, VISION_API_KEY, VISION_BASE_URL, VISION_TEMPERATURE, LLM_MODEL

VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "4096"))

logger = logging.getLogger(__name__)

PROMPT = """你是一位农业病虫害诊断专家。请分析农作物图片，返回 JSON：

{
    "crop_type": "作物名称",
    "growth_stage": "生长阶段",
    "detected_issues": [{
        "type": "病害/虫害/营养问题",
        "name": "具体名称",
        "severity": "轻微/中等/严重",
        "confidence": 0.85,
        "description": "症状描述"
    }],
    "overall_health": "良好/一般/较差",
    "recommendations": ["建议1", "建议2"],
    "urgency": "立即处理/近期处理/持续观察"
}"""


def image_analysis_node(state: AgentState) -> AgentState:
    if not state.has_image or not state.image_data:
        return state

    if not os.getenv("VISION_MODEL"):
        state.image_analysis_result = _fail(
            f"图片分析未启用。请在 .env 中配置 VISION_MODEL（支持多模态的模型）。"
            f"当前 LLM_MODEL={LLM_MODEL} 不支持图片。")
        return state

    try:
        state.image_analysis_result = _call_vision_api(state)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "image_url" in msg.lower() or "multipart" in msg.lower():
            hint = f"模型 {VISION_MODEL} 不支持图片输入，请更换为多模态模型。"
        state.image_analysis_result = _fail(f"{hint or msg}")

    return state


def _call_vision_api(state: AgentState) -> dict:
    url = f"{VISION_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {VISION_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:{state.image_mime_type};base64,{state.image_data}"}},
                {"type": "text", "text": state.user_question or "请分析这张农作物图片"},
            ],
        }],
        "max_tokens": VISION_MAX_TOKENS,
        "temperature": VISION_TEMPERATURE,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"API {resp.status_code}: {resp.text[:300]}")

    try:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("Vision API 响应解析失败: %s, body=%s", e, resp.text[:500])
        raise Exception(f"API 返回格式异常: {resp.text[:200]}")

    if not content or not content.strip():
        raise Exception("API 返回了空内容，可能是图片格式不被支持或图片过大")

    return _parse(content)


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
        parts.append(f"  {emoji} **{issue['name']}** ({issue.get('type','')}) "
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
