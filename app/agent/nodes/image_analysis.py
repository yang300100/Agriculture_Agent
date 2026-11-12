"""图片分析节点：保存本地多任务观测，再按用户范围补充建议。"""

import base64
from copy import deepcopy
from io import BytesIO
import json
import logging
import re

from ..state import AgentState
from ..config import DL_DEFAULT_MODEL, ENABLE_IMAGE_ANALYSIS
from ..image_analysis_policy import local_observation_policy, permitted_advice, permitted_device_link, response_policy, valid_confidence

logger = logging.getLogger(__name__)

LLM_ENHANCE_PROMPT = """你是一位农业植保建议助手。以下本地模型观测是唯一图像事实来源。
模型为封闭集合分类器，支持类别见 coverage；不得声称新增病种识别能力，不能将分类分数当成确诊准确率。
不得修改病名、作物、严重度或其置信度；不得推测图像未提供的生长阶段和整体健康。
请直接回应用户问题，遵守 response_policy。仅提供建议，不再输出识别事实。

用户本轮问题：{user_question}
完整本地模型结果（包括全部候选、独立任务头及类别范围）：
{model_result}

只返回以下 JSON（建议不足时返回空数组）：
{{
  "recommendations": [{{"kind": "medication/nonchemical/observation/verification", "action": "record_symptoms"}}],
  "linked_action": "irrigate/ventilate/heat/shade/light/none",
  "linked_params": {{"duration": 5}}
}}
药剂建议必须说明以当地登记作物、病害和标签为准，不给出无依据剂量。
观察、核对、非化学管理建议仅选择以下 action：record_symptoms（记录症状）、photograph_leaves（补充照片）、
inspect_spread（检查症状分布）、check_ventilation（核对通风凝露）、check_watering（核对浇水记录与土壤湿度）、
consult_local_expert（现场复核）。
用户允许用药时，medication 建议还可选择 check_registered_products（核对登记作物、病害和产品标签）、
check_label_instructions（核对标签剂量、次数与安全间隔期）、check_application_conditions（核对施用天气条件）。
用户禁止用药时不能选择这三个 medication action。
所有建议均由系统根据 action 生成文本，不接收自由描述或本图作物、生长阶段、严重度、整体健康断言。
设备建议必须有充分观测依据；没有依据请返回 none。"""


class _ImageAnalysisError(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def image_analysis_node(state: AgentState) -> AgentState:
    if not state.has_image and not state.image_data:
        return state
    if not ENABLE_IMAGE_ANALYSIS:
        _set_failure(state, "IMAGE_DISABLED", "图片分析未启用，请检查本地模型配置。")
        return state
    try:
        state.image_analysis_result = _call_dl_model(state)
    except _ImageAnalysisError as exc:
        _set_failure(state, exc.code, str(exc), **exc.details)
    except Exception:
        logger.exception("图片分析失败")
        _set_failure(state, "IMAGE_ANALYSIS_FAILED", "图片分析失败，请稍后重试或检查本地模型。")
    return state


def _call_dl_model(state: AgentState) -> dict:
    from PIL import Image
    from core.model_registry_factory import resolve_inference_model
    from core.model_executor import ModelExecutor
    from models.base import ModelInput

    try:
        image_bytes = base64.b64decode(state.image_data or "", validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise _ImageAnalysisError("IMAGE_INVALID_INPUT", "图片内容无效，请重新上传完整图片。") from exc
    try:
        registry, model_id = resolve_inference_model(DL_DEFAULT_MODEL)
    except Exception as exc:
        raise _ImageAnalysisError("IMAGE_MODEL_UNAVAILABLE", "本地图片模型未加载，请检查模型权重和推理依赖。") from exc
    result = ModelExecutor(registry).infer_sync(model_id, ModelInput(image_bytes=image_bytes, top_k=3))
    if not result.success:
        raise _ImageAnalysisError("IMAGE_INFERENCE_FAILED", "本地模型推理失败，请稍后重试。", model_error_code=result.error_code)
    if not result.predictions:
        raise _ImageAnalysisError("IMAGE_NO_PREDICTIONS", "本地模型没有返回分类候选，请重新拍摄并复核。")
    if any(not isinstance(p.class_name, str) or not p.class_name.strip() or not valid_confidence(p.confidence) for p in result.predictions):
        raise _ImageAnalysisError("IMAGE_INVALID_OUTPUT", "本地模型返回了无效的分类结果，请检查模型。")

    # 哥哥，保留原始数值；显示时才进行百分比取舍。
    predictions = [{"class_name": p.class_name, "confidence": p.confidence, "index": p.index} for p in result.predictions]
    analysis = {
        "model_id": result.model_id, "predictions": predictions,
        "raw_output": deepcopy(result.raw_output), "inference_time_ms": result.inference_time_ms,
        **local_observation_policy(predictions, result.raw_output, registry.get_model_info(model_id)),
        "response_policy": response_policy(state.user_question or ""),
        "recommendations": [], "recommendation_items": [], "warnings": [],
        "urgency": "", "llm_advice": "", "llm_status": "skipped",
        "linked_action": "none", "linked_params": {}, "needs_spray_weather": False,
    }
    state.image_analysis_result = analysis
    eligible = analysis["classification_kind"] == "issue" and not analysis["uncertainty"]["requires_review"]
    policy = analysis["response_policy"]
    if eligible and not policy["results_only"]:
        enhanced = _invoke_llm_enhance(predictions[0]["class_name"], predictions[0]["confidence"], state)
        if not isinstance(enhanced, dict) or enhanced.get("error") or enhanced.get("llm_status") == "unavailable":
            analysis["llm_status"] = "unavailable"
            analysis["warnings"].append("建议生成不可用；以上本地模型结果已保留，请结合现场情况复核。")
        else:
            analysis["llm_status"] = "ok"
            items = permitted_advice(enhanced, policy)
            analysis["recommendation_items"] = items
            analysis["recommendations"] = [item["text"] for item in items]
            analysis["llm_advice"] = "\n".join(analysis["recommendations"])
            action, params = permitted_device_link(enhanced.get("linked_action"), enhanced.get("linked_params"))
            if items and policy["allow_device_actions"] and action != "none":
                analysis["linked_action"] = action
                analysis["linked_params"] = params
            analysis["needs_spray_weather"] = policy["allow_medication"] and any(item["kind"] == "medication" for item in items)

    if not eligible or not policy["allow_tasks"] or not policy["allow_medication"] or analysis["llm_status"] != "ok" or not analysis["recommendations"]:
        state.skip_task_extraction = True
    return analysis


def _invoke_llm_enhance(class_name: str, confidence: float, state: AgentState) -> dict:
    """哥哥，保留三参数入口，完整观测通过本轮 state 交给建议模型。"""
    from ..utils import _get_llm
    try:
        prompt = LLM_ENHANCE_PROMPT.format(
            user_question=state.user_question or "请分析图片。",
            model_result=json.dumps(state.image_analysis_result, ensure_ascii=False),
        )
        response = _get_llm().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        parsed = _parse(text)
        if parsed.get("error") or not isinstance(parsed.get("recommendations"), list):
            raise ValueError("建议模型未返回有效的 JSON 建议数组")
        return {"recommendations": parsed["recommendations"], "llm_status": "ok",
                "linked_action": parsed.get("linked_action", "none"), "linked_params": parsed.get("linked_params", {})}
    except Exception as exc:
        logger.warning("图像建议生成不可用: %s", exc)
        return {"llm_status": "unavailable", "recommendations": [], "linked_action": "none", "linked_params": {}}


def image_analysis_answer_node(state: AgentState) -> AgentState:
    analysis = state.image_analysis_result or {}
    if analysis.get("error"):
        state.final_answer = f"⚠️ **分析出错**: {analysis['error']}"
        state.skip_task_extraction = True
    elif not analysis.get("predictions"):
        _set_failure(state, "IMAGE_NO_PREDICTIONS", "图片分析未产生有效结果，请重新上传图片。")
        state.final_answer = f"⚠️ **分析出错**: {state.image_analysis_result['error']}"
    else:
        top = analysis["predictions"][0]
        policy = analysis.get("response_policy") or response_policy(state.user_question or "")
        parts = [f"🔍 **模型分类候选**: {top['class_name']}（置信度 {top['confidence']:.2%}）"]
        kind = analysis.get("classification_kind")
        if kind == "background":
            parts.append("未获得可用于病害判断的叶片图像，请对焦叶片并重拍。")
        elif kind == "healthy":
            parts.append("“健康”是模型已知类别中的分类结果，不能据此排除未覆盖病害。")
        else:
            severity = analysis.get("severity") or "未提供"
            severity_confidence = analysis.get("severity_confidence")
            confidence_text = f"（置信度 {severity_confidence:.2%}）" if valid_confidence(severity_confidence) else ""
            parts.append(f"**模型原始严重度**: {severity}{confidence_text}")
            if not policy["results_only"]:
                raw = analysis.get("raw_output")
                if isinstance(raw, dict):
                    for key, title in (("crop", "作物预测"), ("disease", "病害预测")):
                        head = raw.get(key)
                        if isinstance(head, dict) and isinstance(head.get("class_name"), str) and valid_confidence(head.get("confidence")):
                            parts.append(f"**{title}**: {head['class_name']}（置信度 {head['confidence']:.2%}）")
        uncertainty = analysis.get("uncertainty", {})
        if uncertainty.get("requires_review"):
            parts.append("⚠️ 分类或独立预测的置信度不足、候选接近或图像/类别证据不完整，需要补拍并人工复核；当前结果不用于生成处置。")
            state.skip_task_extraction = True
        coverage = analysis.get("coverage", {})
        classes = coverage.get("crop_classes") or coverage.get("supported_classes") or []
        scope = "、".join(classes) if classes else "注册信息未提供可核验类别"
        parts.append(f"**类别范围**: {scope}。{coverage.get('limitation', '当前模型仅支持封闭类别，不能排除未覆盖病害。')}")
        if not policy["results_only"] and analysis.get("recommendations"):
            parts.append("\n💡 **处理建议**:")
            parts.extend(f"{index}. {text}" for index, text in enumerate(analysis["recommendations"], 1))
        parts.extend(f"⚠️ {warning}" for warning in analysis.get("warnings", []))
        action, params = permitted_device_link(analysis.get("linked_action"), analysis.get("linked_params"))
        analysis["linked_action"], analysis["linked_params"] = action, params
        if action != "none" and policy["allow_device_actions"] and not uncertainty.get("requires_review") and kind == "issue":
            labels = {"irrigate": "灌溉", "ventilate": "通风", "heat": "加热", "shade": "遮阳", "light": "补光"}
            parts.append(f"\n🔧 **建议设备操作**: {labels.get(action, action)}（参数: {params}）")
            state.pending_action = {"device_id": "", "command": "start", "params": params,
                                    "reason": f"图像分类建议: {top['class_name']}", "source": "image_analysis", "linked_action": action}
        state.final_answer = "\n".join(parts)
    state.image_data = None
    state.has_image = False
    return state


def _parse(content: str) -> dict:
    """哥哥，截断或非对象 JSON 不进入可操作建议。"""
    if not isinstance(content, str):
        return {"error": "JSON 内容不是文本"}
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    text = match.group(1).strip() if match else content.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"error": "JSON 不是对象"}
    except (json.JSONDecodeError, TypeError):
        return {"error": "JSON 解析失败"}


def _set_failure(state, code, message, **details):
    state.error_code = code
    state.skip_task_extraction = True
    state.image_analysis_result = {"error": message, "error_code": code, "predictions": [],
                                   "detected_issues": [], "recommendations": [], "linked_action": "none", **details}
