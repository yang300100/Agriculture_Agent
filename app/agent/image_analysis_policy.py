"""哥哥，先确定回答边界，再允许图像建议进入后续流程。"""

import math
import re


_REQUEST_PREFIX = r"(?:(?:请|麻烦|帮我|这次|本次|暂时|先|也|并且|同时|另外|但是|但|而且|且|并|再|还|直接))*"
_NEGATIVE_REQUEST = r"(?:不要|别|不必|无需|不用|不需要|不想|禁止|不)(?!只|仅)"
_REQUEST_VERBS = r"(?:(?:给我|向我|你|给出|给|提供|推荐|生成|显示|包含|涉及|讲解|讲|说明|做|去|再|任何|自动|实际|立即|直接|有关|关于|如何|使用|开启|执行|操作|联动|创建|建立|新建|添加|安排|设置))*"
_COORDINATED_OBJECT = r"(?:用药|药剂|农药|药物|化学防治|喷药|施药|打药|浇水|灌溉|通风|设备|处理|防治|任务|提醒|待办)(?:建议|操作)?"
_SAFE_OBSERVATION_ACTIONS = {
    "record_symptoms": "记录病斑范围、颜色及每日变化，供现场复核。",
    "photograph_leaves": "补充叶片正反面和整株照片，保留拍摄日期。",
    "inspect_spread": "检查同一区域其他植株并记录相似症状的分布。",
    "check_ventilation": "检查种植环境通风和叶面凝露情况，结合现场条件调整管理。",
    "check_watering": "核对近期浇水记录与土壤湿度，避免仅凭图像分类调整灌溉。",
    "consult_local_expert": "请当地植保人员结合田间症状复核分类结果。",
}
_SAFE_MEDICATION_ACTIONS = {
    "check_registered_products": "如现场复核确认需要药剂防治，请核对当地登记作物、病害与产品标签；杀菌剂、杀虫剂不能仅凭图像分类选定。",
    "check_label_instructions": "施用前核对产品标签中的剂量、施用次数和安全间隔期，并咨询当地植保人员。",
    "check_application_conditions": "用药前核对降雨、风速及当地施用要求，避开不适宜的施用条件。",
}


def response_policy(question: str) -> dict:
    """只匹配完整请求片段，避免把“不要只返回”等反向语义当成限制。"""
    separators = r"[，,。；;！？!?\n]|(?:但是|但|而且|并且|同时|以及|也)(?=(?:请)?(?:不要|别|不必|无需|不用|不需要|不想|禁止|只|仅))"
    clauses = [re.sub(r"\s+", "", clause) for clause in re.split(separators, question or "")]

    def prohibited(target):
        coordinated = rf"(?:{_COORDINATED_OBJECT}(?:和|及|与|或|以及|、){_REQUEST_VERBS})*"
        pattern = _REQUEST_PREFIX + _NEGATIVE_REQUEST + _REQUEST_VERBS + coordinated + target
        return any(re.match(pattern, clause) for clause in clauses)

    only_pattern = (
        _REQUEST_PREFIX + r"(?:只|仅)(?:需|需要|要)?"
        r"(?:返回|输出|给出|显示|展示|告诉我|给我|提供|说明)?"
        r"(?:识别结果|识别|模型(?:原始)?(?:结果|严重度)|原始结果|结果|病名|置信度|严重度)"
    )
    results_only = any(re.match(only_pattern, clause) for clause in clauses)
    nonchemical_only_pattern = (
        _REQUEST_PREFIX + r"(?:只|仅)(?:需|需要|要)?(?:提供|给出|给我|给|返回|输出)?"
        r"(?:观察(?:(?:与|和)管理)?|管理|非化学防治|非药剂|非用药)(?:的)?建议"
    )
    nonchemical_only = any(re.match(nonchemical_only_pattern, clause) for clause in clauses)
    no_advice = prohibited(r"(?:处理建议|防治建议|建议)")
    no_medication = prohibited(r"(?:用药|药剂|农药|药物|药|化学防治|喷药|施药|打药|杀菌剂|杀虫剂)")
    no_tasks = prohibited(r"(?:任务|提醒|待办)")
    no_devices = prohibited(r"(?:设备|灌溉|浇水|通风|加热|遮阳|补光)")
    return {
        "results_only": results_only or no_advice,
        "allow_medication": not (results_only or no_advice or no_medication or nonchemical_only),
        "allow_tasks": not (results_only or no_advice or no_tasks),
        "allow_device_actions": not (results_only or no_advice or no_medication or nonchemical_only or no_tasks or no_devices),
    }


def normalize_label(label: str) -> str:
    return str(label).replace("___", "-").replace("__", "-")


def label_parts(label: str) -> tuple:
    label = normalize_label(label)
    return tuple(label.split("-", 1)) if "-" in label else ("", label)


def valid_confidence(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def permitted_device_link(action, params) -> tuple:
    """哥哥，图像联动只接收协议内的时长，不携带模型自由描述。"""
    from core.device_action_schema import ACTION_SCHEMAS
    from core.device_safety_policy import ABSOLUTE_CEILINGS

    if not isinstance(action, str) or action not in {"irrigate", "ventilate", "heat", "shade", "light"} or not isinstance(params, dict):
        return "none", {}
    duration = params.get("duration")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        return "none", {}
    try:
        duration = float(duration)
    except (ValueError, OverflowError):
        return "none", {}
    # 哥哥，复用设备参数目录和统一物理上限，部署收紧时这里也会同步生效。
    spec = ACTION_SCHEMAS[action]["parameters"]["duration"]
    ceiling = ABSOLUTE_CEILINGS[action]["max_duration_per_use_minutes"]
    if not math.isfinite(duration) or not math.isfinite(ceiling) or not spec["min"] <= duration <= min(spec["max"], ceiling):
        return "none", {}
    return action, {"duration": int(duration) if duration.is_integer() else duration}


def local_observation_policy(predictions: list, raw_output, model_info) -> dict:
    """模型观测和类别范围由本地结果决定，语言模型不能补写。"""
    top = predictions[0]
    crop, disease = label_parts(top["class_name"])
    raw = raw_output if isinstance(raw_output, dict) else {}
    crop_head = raw.get("crop") if isinstance(raw.get("crop"), dict) else {}
    severity_head = raw.get("severity") if isinstance(raw.get("severity"), dict) else {}
    classes = list(getattr(model_info, "classes", []) or [])
    if not classes:
        metadata = getattr(model_info, "metadata", {}) or {}
        classes = (metadata.get("taxonomy") or {}).get("joint_classes", [])
    supported = list(dict.fromkeys(normalize_label(label) for label in classes if isinstance(label, str) and label.strip()))
    crop_classes = [label for label in supported if label_parts(label)[0] == crop] if crop else supported
    healthy = disease.casefold() in {"健康", "healthy"}
    background = crop.casefold() in {"背景", "background"} or disease.casefold() in {"无叶片", "背景", "background", "no leaf"}
    reasons = []
    if top["confidence"] < 0.7:
        reasons.append("low_confidence")
    margin = top["confidence"] - predictions[1]["confidence"] if len(predictions) > 1 else None
    if margin is not None and margin < 0.1:
        reasons.append("close_candidates")
    if background:
        reasons.append("background")
    if not supported:
        reasons.append("coverage_unavailable")
    elif normalize_label(top["class_name"]) not in supported:
        reasons.append("unsupported_prediction")
    if not healthy and not background:
        for key in ("crop", "disease", "severity"):
            head = raw.get(key)
            if isinstance(head, dict) and head:
                if not valid_confidence(head.get("confidence")) or head["confidence"] < 0.7:
                    reasons.append(f"low_{key}_confidence")
                expected = crop if key == "crop" else disease if key == "disease" else None
                if expected and head.get("class_name") != expected:
                    reasons.append(f"inconsistent_{key}_head")
    severity = severity_head.get("class_name", "")
    severity_confidence = severity_head.get("confidence")
    issue_type = "虫害" if any(word in disease for word in ("虫", "螨", "蚜", "蜘蛛", "虱", "蛾")) else "病害"
    return {
        "crop_type": crop_head.get("class_name") or (crop if not background else ""),
        "growth_stage": "", "overall_health": "",
        "severity": severity if isinstance(severity, str) else "",
        "severity_confidence": severity_confidence if valid_confidence(severity_confidence) else None,
        "classification_kind": "background" if background else "healthy" if healthy else "issue",
        "detected_issues": [] if healthy or background else [{
            "type": issue_type, "name": top["class_name"], "confidence": top["confidence"],
            "severity": severity if isinstance(severity, str) else "",
            "severity_confidence": severity_confidence if valid_confidence(severity_confidence) else None,
            "description": "", "status": "candidate",
        }],
        "coverage": {
            "closed_set": True, "supported_classes": supported, "crop_classes": crop_classes,
            "limitation": "当前模型仅在封闭类别集合内分类；置信度不能证明已排除未覆盖病害，也不是诊断准确率。",
        },
        "uncertainty": {"requires_review": bool(reasons), "reasons": reasons,
                        "confidence_threshold": 0.7, "candidate_margin": margin},
    }


def permitted_advice(enhanced: dict, policy: dict) -> list:
    """哥哥，建议只选择受控行动，自由文本不能重新补写图像事实。"""
    if policy["results_only"]:
        return []
    if not policy["allow_medication"]:
        actions = []
        for item in enhanced.get("recommendations", []) if isinstance(enhanced.get("recommendations"), list) else []:
            action = item.get("action") if isinstance(item, dict) else None
            if isinstance(action, str) and action in _SAFE_OBSERVATION_ACTIONS and action not in actions:
                actions.append(action)
        return [{"kind": "verification", "text": _SAFE_OBSERVATION_ACTIONS[action]} for action in actions] or [
            {"kind": "verification", "text": "补充叶片正反面和整株照片，并请当地植保人员现场复核。"}]
    recommendations = enhanced.get("recommendations")
    if not isinstance(recommendations, list):
        return []
    accepted = []
    accepted_actions = set()
    for item in recommendations[:8]:
        if isinstance(item, str):
            kind = "medication" if any(term in item for term in ("药", "喷施", "喷雾", "喷洒", "杀菌", "杀虫")) else "general"
            item = {"kind": kind, "text": item}
        if not isinstance(item, dict) or not isinstance(item.get("kind"), str) or item["kind"] not in {"medication", "nonchemical", "observation", "verification", "general"}:
            continue
        action = item.get("action")
        if isinstance(action, str) and action in _SAFE_OBSERVATION_ACTIONS:
            kind, text = "verification", _SAFE_OBSERVATION_ACTIONS[action]
        elif item["kind"] == "medication":
            legacy_text = item.get("text")
            if (not action or action == "none") and isinstance(legacy_text, str) and re.match(
                r"^(?:请|建议)?(?:按|按照|核对|确认|咨询|选用|选择|使用|喷施|喷洒|轮换)", legacy_text.strip()
            ):
                # 哥哥，旧格式仅用于选择核对行动，原文里的作物和病情不会展示。
                action = "check_registered_products"
            if not isinstance(action, str) or action not in _SAFE_MEDICATION_ACTIONS:
                continue
            kind, text = "medication", _SAFE_MEDICATION_ACTIONS[action]
        else:
            continue
        if action not in accepted_actions:
            accepted_actions.add(action)
            accepted.append({"kind": kind, "action": action, "text": text})
    return accepted
