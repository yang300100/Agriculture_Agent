"""设备动作参数目录。

前后端共用同一份能力参数定义，避免规则页面只能保存固定的
``duration=30``，也避免向设备下发无法识别的任意字段。
"""

from copy import deepcopy
from typing import Any, Dict


ACTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "irrigate": {
        "label": "灌溉",
        "commands": ["start", "stop", "set_param"],
        "parameters": {
            "duration": {"label": "持续时间", "unit": "分钟", "min": 1, "max": 120, "default": 30},
            "volume_liters": {"label": "目标水量", "unit": "升", "min": 1, "max": 100000, "default": 500},
            "target_soil_moisture": {"label": "目标土壤湿度", "unit": "%", "min": 1, "max": 100, "default": 35},
            "flow_rate": {"label": "目标流量", "unit": "L/min", "min": 0.1, "max": 10000, "default": 20},
        },
    },
    "fertigate": {
        "label": "施肥",
        "commands": ["start", "stop", "set_param"],
        "parameters": {
            "duration": {"label": "持续时间", "unit": "分钟", "min": 1, "max": 120, "default": 20},
            "amount_kg": {"label": "施肥量", "unit": "kg", "min": 0.1, "max": 50, "default": 5},
            "concentration_percent": {"label": "肥液浓度", "unit": "%", "min": 0.1, "max": 100, "default": 1.5},
        },
    },
    "ventilate": {
        "label": "通风",
        "commands": ["start", "stop", "set_param"],
        "parameters": {
            "duration": {"label": "持续时间", "unit": "分钟", "min": 1, "max": 120, "default": 30},
            "speed_percent": {"label": "风机速度", "unit": "%", "min": 1, "max": 100, "default": 70},
            "target_temp": {"label": "目标温度", "unit": "℃", "min": -20, "max": 60, "default": 28},
        },
    },
    "light": {
        "label": "补光",
        "commands": ["start", "stop", "set_param"],
        "parameters": {
            "duration": {"label": "持续时间", "unit": "分钟", "min": 1, "max": 720, "default": 120},
            "brightness_percent": {"label": "亮度", "unit": "%", "min": 1, "max": 100, "default": 70},
            "target_lux": {"label": "目标光照", "unit": "lux", "min": 1, "max": 200000, "default": 15000},
        },
    },
    "heat": {
        "label": "加热",
        "commands": ["start", "stop", "set_param"],
        "parameters": {
            "duration": {"label": "持续时间", "unit": "分钟", "min": 1, "max": 240, "default": 30},
            "target_temp": {"label": "目标温度", "unit": "℃", "min": -20, "max": 60, "default": 25},
            "power_percent": {"label": "功率", "unit": "%", "min": 1, "max": 100, "default": 70},
        },
    },
    "cool": {
        "label": "降温",
        "commands": ["start", "stop", "set_param"],
        "parameters": {
            "duration": {"label": "持续时间", "unit": "分钟", "min": 1, "max": 240, "default": 30},
            "target_temp": {"label": "目标温度", "unit": "℃", "min": -20, "max": 60, "default": 24},
            "power_percent": {"label": "功率", "unit": "%", "min": 1, "max": 100, "default": 70},
        },
    },
    "shade": {
        "label": "遮阳",
        "commands": ["start", "stop", "set_param"],
        "parameters": {
            "duration": {"label": "持续时间", "unit": "分钟", "min": 1, "max": 240, "default": 30},
            "position_percent": {"label": "遮阳位置", "unit": "%", "min": 0, "max": 100, "default": 100},
        },
    },
}


def get_action_catalog() -> Dict[str, Dict[str, Any]]:
    return deepcopy(ACTION_SCHEMAS)


def normalize_action(capability: str, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """校验并规范化规则动作；停止操作不携带运行参数。"""
    capability = str(capability or "").lower()
    schema = ACTION_SCHEMAS.get(capability)
    if not schema:
        raise ValueError(f"不支持的动作能力: {capability}")
    command = str(command or "start").lower()
    if command not in schema["commands"]:
        raise ValueError(f"{capability} 不支持指令 {command}")
    if command == "stop":
        return {}
    if not isinstance(params, dict):
        raise ValueError("动作参数必须是 JSON 对象")

    normalized: Dict[str, Any] = {}
    allowed = schema["parameters"]
    for key, value in params.items():
        if key not in allowed:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"参数 {key} 必须是数字") from exc
        spec = allowed[key]
        if number < spec["min"] or number > spec["max"]:
            raise ValueError(
                f"参数 {key} 必须在 {spec['min']}～{spec['max']} 之间"
            )
        normalized[key] = int(number) if number.is_integer() else number

    if command == "start" and not normalized:
        raise ValueError("启动操作至少需要设置一个动作参数")
    if command == "set_param" and len(normalized) != 1:
        raise ValueError("设置参数操作必须且只能选择一个参数")
    return normalized
