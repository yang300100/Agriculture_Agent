"""设备控制 Agent — 解析用户意图，调度设备操作"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional

from .base import BaseAgent
from ..state import AgentState
from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

ACTION_KEYWORDS = {
    "irrigate": ("浇水", "灌溉", "浇灌", "水泵"),
    "fertigate": ("施肥", "追肥", "施肥机"),
    "ventilate": ("通风", "换气", "开窗", "风机"),
    "light": ("补光", "开灯", "关灯", "灯"),
    "shade": ("遮阳", "遮光"),
    "heat": ("加热", "加温", "升温", "加热器"),
    "cool": ("降温", "冷却"),
}


def _non_immediate_request(question: str, allow_deferred: bool = False) -> bool:
    """哥哥，否定、完成记录和预约语境都不能误变成即时启动。"""
    negated = re.search(
        r"不要|不必|无需|不需要|暂不|先不|别(?!忘)|"
        r"不(?:浇水|灌溉|施肥|通风|补光|加热|启动|开启|执行|停止|关闭|关灯|关机)|"
        r"如何|怎么|怎样|是否|能否|建议|该不该|多久|适合|可以吗|能不能|"
        r"(?:浇水|灌溉|施肥|通风|补光|加热)(?:过|了|完)", question
    )
    deferred = re.search(
        r"(?:添加|创建|新建|设置|安排|帮我建|帮我加).*?(?:任务|提醒|待办)|"
        r"明天|后天|下周|下个月|预约|定时|每(?:天|周|月)|"
        r"(?:\d+|[一二三四五六七八九十两半]+)\s*(?:分钟|小时|天)后", question
    )
    return bool(negated or (deferred and not allow_deferred))


def _discover_devices(username: str = "default") -> list:
    """哥哥，每次发现都使用本次请求的用户名，不在共享 Agent 中留状态。"""
    from core.device_registry_factory import setup_registry, close_registry

    registry, loop = setup_registry(username=username)
    try:
        return loop.run_until_complete(registry.discover_all())
    finally:
        close_registry(loop, registry)


def _has_capability(device, capability: str) -> bool:
    return any(getattr(cap, "value", str(cap)) == capability for cap in device.capabilities)


def _devices_matching_hint(devices: list, hint: str) -> list:
    hint = str(hint or "").strip().lower()
    exact = [device for device in devices if hint in {device.device_id.lower(), device.name.lower()}]
    if exact:
        return exact
    named = [device for device in devices if any(
        value and value.lower() in hint for value in (device.device_id, device.name)
    )]
    if named:
        return named
    return [device for device in devices if hint and any(
        hint in str(value or "").lower() for value in (device.device_id, device.name, device.location)
    )]

# ── 设备动作类型 → 设备能力映射 ──
ACTION_TO_CAPABILITY = {
    "irrigate": "irrigate",
    "fertigate": "fertigate",
    "ventilate": "ventilate",
    "light": "light",
    "heat": "heat",
    "cool": "heat",     # 降温：找加热器设备（反向控制）
    "shade": "light",   # 遮阳：找补光灯设备（反向控制）
    "read_sensor": "read_sensor",
}


def _discover_device_for_capability(capability: str, username: str = "default", device_hint: str = "") -> Optional[str]:
    """动态发现具备指定能力的设备ID（从用户注册的设备列表中查找）。

    每次调用都会重新加载 registry 以获取最新设备列表。
    只有目标唯一时才返回设备ID，避免同能力设备之间串用。
    """
    try:
        devices = [device for device in _discover_devices(username) if _has_capability(device, capability)]
        if device_hint:
            devices = _devices_matching_hint(devices, device_hint)
        if len(devices) == 1:
            return devices[0].device_id
    except Exception:
        pass
    return None


def _discover_sensor_device(username: str = "default") -> Optional[str]:
    """动态发现传感器设备ID（用于读取温湿度等环境数据）"""
    return _discover_device_for_capability("read_sensor", username) or \
        _discover_device_for_capability("irrigate", username)


class DeviceAgent(BaseAgent):
    name = "device"
    description = "智能设备控制专家，负责灌溉、施肥、通风、补光等设备自主操作与调度"
    system_prompt = """你是一位智能农业设备控制专家。
你能：
1. 理解用户的设备控制需求（浇水、施肥、通风、补光等）
2. 根据上下文（传感器数据、天气、作物阶段）推荐最佳操作参数
3. 在安全规则边界内自主决策和执行设备指令
4. 与其他 Agent（气象、病虫害）协作，实现联动控制

关键原则：
- 永远在规则引擎的安全边界内操作
- 当操作超出用户设定边界时，生成待确认操作而非直接执行
- 执行前必须检查当前天气和设备状态"""

    intent_types = ["device_control"]

    def invoke(self, state: AgentState) -> AgentState:
        question = state.user_question or ""

        try:
            safe_intent = self._keyword_parse(question)
            if _non_immediate_request(question) and not (safe_intent and safe_intent["action"] == "status"):
                return self._reply(state, "哥哥，这条消息属于记录、预约或暂缓操作，请在任务列表中管理对应安排～")
            parsed = self._parse_device_intent(question, state)
            if not parsed:
                return self._reply(state, "抱歉哥哥，我没理解你想操作哪个设备呢～能再说详细一点吗？比如「帮小麦浇30分钟水」")

            username = state.username
            devices = _discover_devices(username)
            if parsed.get("action") == "status":
                return self._read_device_status(parsed, state, devices)
            device_id, selection_message = self._select_device(parsed, state, devices)
            if not device_id:
                return self._reply(state, selection_message)
            parsed["device_id"] = device_id
            if parsed.get("action") == "start":
                target = next(device for device in devices if device.device_id == device_id)
                parsed["capability"] = next(
                    (getattr(cap, "value", str(cap)) for cap in target.capabilities
                     if getattr(cap, "value", str(cap)) in ACTION_TO_CAPABILITY.values()
                     and getattr(cap, "value", str(cap)) != "read_sensor"), ""
                )
            from core.device_rule_engine import RuleEngine
            engine = RuleEngine(username=username)
            if parsed.get("action") == "stop":
                return self._do_execute(device_id, "stop", {}, state, engine, capability=parsed.get("capability"))
            matched_rules = self._match_rules(parsed, state, engine)
            state.matched_rules = [str(rule["id"]) for rule in matched_rules]

            if matched_rules:
                return self._execute_with_rule(matched_rules[0], parsed, state, engine)
            else:
                return self._execute_direct(parsed, state, engine)

        except Exception as e:
            logger.exception("DeviceAgent 处理失败")
            # 不暴露原始异常信息给用户
            return self._reply(state, "设备控制处理出错了，请稍后再试～")

    def _parse_device_intent(self, question: str, state: AgentState) -> Optional[Dict]:
        """用 LLM 从用户自然语言中提取设备操作参数"""
        safe_intent = self._keyword_parse(question)
        if safe_intent and safe_intent["action"] in {"stop", "status"}:
            return safe_intent
        if _non_immediate_request(question):
            return None
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
        from ..config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL

        context = self._get_context(state)
        history = "\n".join(f"{message.type}：{message.content}" for message in state.messages[-6:])

        prompt = f"""分析用户的设备控制需求，提取操作参数。

用户输入："{question}"

上下文：当前作物：{context.get('crop', '未指定')}，地区：{context.get('region', '未指定')}
最近对话（仅用于理解当前指令及设备指代）：
{history}

请以 JSON 格式返回：
{{
    "action": "irrigate|fertigate|ventilate|light|heat|cool|shade|status|stop",
    "device_hint": "用户明确指定的设备ID或设备名称，未指定时留空",
    "capability": "停止操作对应的设备能力，未指定时留空",
    "crop": "目标作物",
    "params": {{"duration": 数字分钟, "amount_kg": 数字kg, "target_temp": 数字°C, "brightness_percent": 数字}},
    "reasoning": "操作理由"
}}

停止、关闭、关灯必须返回 stop。查看设备、状态、传感器必须返回 status。
只有用户当前明确要求立即执行时才返回运行动作。否定、描述过去、预约及创建任务均返回 unknown。
历史只能用于识别当前目标，不能重新执行历史中的指令。
如果无法判断具体操作，action 设为 "unknown"。"""

        try:
            llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE,
                             api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            resp = llm.invoke([HumanMessage(content=prompt)])
            content = resp.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or parsed.get("action") not in {*ACTION_TO_CAPABILITY, "status", "stop", "start"}:
                return None
            if not isinstance(parsed.get("params", {}), dict):
                return None
            if parsed.get("command") == "stop" and parsed.get("action") in ACTION_TO_CAPABILITY:
                parsed["capability"] = ACTION_TO_CAPABILITY[parsed["action"]]
                parsed["action"] = "stop"
            return parsed
        except Exception as e:
            logger.warning("DeviceAgent 意图解析失败: %s, 回退到关键词匹配", e)
            return self._keyword_parse(question)

    def _keyword_parse(self, question: str, allow_deferred: bool = False) -> Optional[Dict]:
        """关键词回退解析 — 先提取时长，再匹配操作类型。
        未识别任何关键词时返回 None（不默认触发灌溉！）。"""
        if re.search(r"设备列表|设备状态|哪些设备|有哪些.*设备|(?:查看|查询|看看|读取).*?(?:设备|状态|温度|湿度|传感器)", question):
            return {"action": "status", "params": {}}
        if _non_immediate_request(question, allow_deferred):
            return None
        if any(word in question for word in ("关闭", "停止", "关灯", "关机", "停机")):
            capability = next((ACTION_TO_CAPABILITY[action] for action, words in ACTION_KEYWORDS.items()
                               if any(word in question for word in words)), "")
            return {"action": "stop", "capability": capability, "params": {}}

        dur_match = re.search(r'(\d+(?:\.\d+)?)\s*(分钟|分|小时|秒)', question)
        duration = float(dur_match.group(1)) if dur_match else 30
        # 小时/秒 转换
        if dur_match and dur_match.group(2) == "小时":
            duration = duration * 60
        elif dur_match and dur_match.group(2) == "秒":
            duration = max(1, duration // 60)

        # 按优先级匹配操作类型
        if any(kw in question for kw in ["浇水", "灌溉", "浇灌"]) or re.search(r"浇\s*\d+(?:\.\d+)?\s*(?:分钟|分|小时|秒)\s*水", question):
            return {"action": "irrigate", "params": {"duration": duration}}
        if any(kw in question for kw in ["施肥", "追肥"]):
            amount = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)", question, re.IGNORECASE)
            return {"action": "fertigate", "params": {"amount_kg": float(amount.group(1)) if amount else 5, "duration": duration}}
        if any(kw in question for kw in ["通风", "换气", "开窗"]):
            return {"action": "ventilate", "params": {"duration": duration}}
        if any(kw in question for kw in ["补光", "开灯"]):
            return {"action": "light", "params": {"brightness_percent": 80, "duration": duration}}
        if any(kw in question for kw in ["遮阳", "遮光"]):
            return {"action": "shade", "params": {}}
        if any(kw in question for kw in ["加热", "加温", "升温"]):
            return {"action": "heat", "params": {"target_temp": 22, "duration": duration}}
        if any(kw in question for kw in ["降温", "冷却"]):
            return {"action": "cool", "params": {"target_temp": 20, "duration": duration}}
        if any(word in question for word in ("开启", "启动", "打开", "开机")):
            return {"action": "start", "params": {"duration": duration}}
        # 未识别任何操作类型 → 返回 None，不默认触发灌溉
        return None

    def _select_device(self, parsed: Dict, state: AgentState, devices: list) -> tuple:
        """哥哥，明确目标优先；多个候选必须由当前请求或历史指代唯一确定。"""
        action = parsed.get("action", "")
        capability = parsed.get("capability") or ACTION_TO_CAPABILITY.get(action, "")
        candidates = [device for device in devices if not capability or _has_capability(device, capability)]
        if action in {"stop", "start"} and not capability:
            candidates = [device for device in candidates if any(
                _has_capability(device, cap) for cap in ACTION_TO_CAPABILITY.values() if cap != "read_sensor"
            )]
        question = state.user_question or ""
        named = [device for device in candidates if any(
            value and value.lower() in question.lower() for value in (device.device_id, device.name)
        )]
        hint = parsed.get("device_id") or parsed.get("device_hint", "")
        if named:
            candidates = named
        elif hint:
            candidates = _devices_matching_hint(candidates, hint)
        elif re.search(r"它|刚才|上次|之前那|这台|那台", question):
            for message in reversed(state.messages):
                if message.content == question:
                    continue
                mentioned = [device for device in candidates if any(
                    value and value.lower() in str(message.content).lower() for value in (device.device_id, device.name)
                )]
                if mentioned:
                    candidates = mentioned
                    break
        if len(candidates) == 1:
            return candidates[0].device_id, ""
        if not candidates:
            return None, "哥哥，当前账号没有找到符合要求的设备，请检查设备名称、能力和在线状态～"
        choices = "、".join(f"{device.name}（{device.device_id}）" for device in candidates)
        return None, f"哥哥，有多个可选设备，请指定名称或ID～\n\n{choices}"

    def _read_device_status(self, parsed: Dict, state: AgentState, devices: list) -> AgentState:
        """哥哥，查询只读取实际设备状态，不走设备启动或任务创建。"""
        from core.device_registry_factory import setup_registry, close_registry

        if not devices:
            return self._reply(state, "哥哥，当前账号还没有可用设备～")
        question = state.user_question or ""
        has_named_device = any(value and value.lower() in question.lower()
                               for device in devices for value in (device.device_id, device.name))
        if parsed.get("device_hint") or has_named_device:
            device_id, message = self._select_device(parsed, state, devices)
            if not device_id:
                return self._reply(state, message)
            devices = [device for device in devices if device.device_id == device_id]
        registry, loop = setup_registry(username=state.username)
        try:
            loop.run_until_complete(registry.discover_all())
            readings = []
            for device in devices:
                readings.append({
                    "device_id": device.device_id,
                    "name": device.name,
                    "state": loop.run_until_complete(registry.read_state(device.device_id)),
                })
            state.device_result = {"success": True, "devices": readings}
            content = "\n".join(f"- {item['name']}（{item['device_id']}）\n  {json.dumps(item['state'], ensure_ascii=False)}"
                                for item in readings)
            return self._reply(state, f"哥哥，当前设备状态如下～\n\n{content}")
        finally:
            close_registry(loop, registry)

    def _match_rules(self, parsed: Dict, state: AgentState, engine) -> list:
        """查找与当前操作匹配的规则"""
        try:
            username = getattr(state, 'username', 'default')
            sensor_context = self._get_sensor_context(parsed.get("action", ""), username)
            context = {"sensor_data": sensor_context, "weather": {}, "crop": parsed.get("crop", "")}
            capability = parsed.get("capability") or ACTION_TO_CAPABILITY.get(parsed.get("action", ""), "")
            command = "stop" if parsed.get("action") in {"cool", "shade", "stop"} else "start"
            return [rule for rule in engine.find_matching_rules(context)
                    if rule.get("action", {}).get("device_id") == parsed.get("device_id")
                    and rule.get("action", {}).get("command", "start") == command
                    and (rule.get("action", {}).get("capability") or capability) == capability]
        except Exception as e:
            logger.warning("规则匹配失败: %s", e)
            return []

    def _get_sensor_context(self, action: str, username: str = "default") -> Dict:
        """获取当前传感器数据 — 动态发现传感器设备并读取数值。

        使用共享的 registry 而非自己创建 SimulatorDriver，
        避免事件循环冲突和数据不一致。
        """
        try:
            from core.device_registry_factory import setup_registry, close_registry

            # 动态发现传感器设备
            sensor_id = _discover_sensor_device(username)
            if not sensor_id:
                return {}

            registry, loop = setup_registry(username=username)
            try:
                loop.run_until_complete(registry.discover_all())
                state = loop.run_until_complete(registry.read_state(sensor_id))
                return state
            finally:
                close_registry(loop, registry)
        except Exception:
            return {}

    def _execute_with_rule(self, rule: Dict, parsed: Dict, state: AgentState, engine) -> AgentState:
        """有匹配规则时的执行逻辑"""
        from core.device_rule_engine import RuleDecision, apply_autonomy
        from ..config import get_autonomy_level

        autonomy = get_autonomy_level(state.username)
        action = rule.get("action", {})
        device_id = action.get("device_id", "")
        proposed_params = {**action.get("params", {}), **parsed.get("params", {})}
        sensor_context = self._get_sensor_context(
            parsed.get("action", ""), getattr(state, "username", "default")
        )
        decision, reason, final_params = engine.evaluate_action(
            rule, proposed_params,
            {"device_id": device_id, "sensor_data": sensor_context},
        )

        # 应用自主权级别
        decision = apply_autonomy(decision, autonomy)
        execution_mode = rule.get("execution_mode", "auto")
        if execution_mode == "notify":
            reason = f"规则「{rule.get('name', '')}」仅通知，不操作设备"
            self._write_decision_log(
                state, device_id, action.get("command", "start"),
                final_params, "notify", reason, rule.get("id"),
                capability=parsed.get("capability") or ACTION_TO_CAPABILITY.get(parsed.get("action", "")),
                policy_context={"device_id": device_id, "sensor_data": sensor_context},
            )
            return self._reply(state, f"📢 {reason}")
        if execution_mode == "confirm" and decision != RuleDecision.REJECTED:
            # 哥哥，规则明确要求的确认不能被高自主模式跳过。
            decision = RuleDecision.NEED_CONFIRM
            reason = f"规则「{rule.get('name', '')}」设置为执行前确认"

        if decision == RuleDecision.AUTO_EXECUTE:
            extra = f"✅ 规则「{rule.get('name', '')}」校验通过"
            if autonomy == "high":
                extra += "（高自主模式：自动执行）"
            return self._do_execute(
                device_id, action.get("command", "start"), final_params, state, engine,
                rule_id=rule["id"], extra=extra,
                capability=parsed.get("capability") or ACTION_TO_CAPABILITY.get(parsed.get("action", "")),
                policy_context={"device_id": device_id, "sensor_data": sensor_context},
            )
        else:
            # 记录未执行决策（need_confirm / rejected）到执行日志
            pending_id = self._write_decision_log(
                state, device_id, action.get("command", "start"),
                final_params, decision, reason, rule.get("id"),
                capability=parsed.get("capability") or ACTION_TO_CAPABILITY.get(parsed.get("action", "")),
                policy_context={"device_id": device_id, "sensor_data": sensor_context},
            )
            if decision == "need_confirm":
                if not pending_id:
                    state.error_code = "PENDING_ACTION_SAVE_FAILED"
                    return self._reply(state, "哥哥，待确认操作保存失败，请稍后重试。")
                if autonomy == "low":
                    extra_note = "\n（低自主模式：所有操作均需确认）"
                else:
                    extra_note = ""
                state.pending_action = {
                    "id": pending_id,
                    "device_id": device_id,
                    "command": action.get("command", "start"),
                    "params": final_params,
                    "reason": reason,
                    "rule_id": rule["id"],
                }
                return self._reply(state, f"⚠️ {reason}\n\n📋 操作预览：{device_id} → {action.get('command')} 参数：{final_params}{extra_note}\n\n请在「设备中心」中确认此操作。")
            else:
                return self._reply(state, f"❌ {reason}")

    def _execute_direct(self, parsed: Dict, state: AgentState, engine) -> AgentState:
        """哥哥，没有适用规则时也使用同一套安全评估和明确目标。"""
        from core.device_rule_engine import RuleDecision, apply_autonomy
        from ..config import get_autonomy_level

        action_type = parsed.get("action", "")
        capability = parsed.get("capability") or ACTION_TO_CAPABILITY.get(action_type, "")
        device_id = parsed.get("device_id") or self._find_device_for_action(
            action_type, state.username, parsed.get("device_hint", "")
        )
        if not device_id:
            return self._reply(state, "哥哥，没有找到唯一匹配的设备，请指定设备名称或ID～")
        command = "stop" if action_type in {"cool", "shade", "stop"} else "start"
        params = {} if command == "stop" else dict(parsed.get("params", {}))
        context = {"device_id": device_id, "sensor_data": self._get_sensor_context(action_type, state.username)}
        temp_rule = {
            "id": "temp_direct",
            "action": {"device_id": device_id, "capability": capability, "command": command, "params": params},
            "constraints": {"max_duration_per_use": 60, "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5]},
        }
        decision, reason, final_params = engine.evaluate_action(temp_rule, params, context)
        if command != "stop":
            decision = apply_autonomy(decision, get_autonomy_level(state.username))
        if decision != RuleDecision.AUTO_EXECUTE:
            pending_id = self._write_decision_log(
                state, device_id, command, final_params, decision, reason,
                capability=capability, policy_context=context,
            )
            if decision == RuleDecision.NEED_CONFIRM:
                if not pending_id:
                    state.error_code = "PENDING_ACTION_SAVE_FAILED"
                    return self._reply(state, "哥哥，待确认操作保存失败，请稍后重试。")
                state.pending_action = {
                    "id": pending_id, "device_id": device_id, "command": command,
                    "params": final_params, "reason": reason, "capability": capability,
                }
                return self._reply(state, f"⚠️ {reason}\n\n请在「设备中心」中确认此操作。")
            return self._reply(state, f"❌ {reason}")
        extra = {"cool": "（降温模式：关闭加热器）", "shade": "（遮阳模式：关闭补光灯）"}.get(action_type, "")
        return self._do_execute(
            device_id, command, final_params, state, engine, extra=extra,
            capability=capability, policy_context=context,
        )

    def _write_decision_log(self, state: AgentState, device_id: str, command: str,
                            params: Dict, decision: str, reason: str, rule_id: str = None,
                            capability: str = None, policy_context: Dict = None):
        """将 Agent 未执行决策写入数据库，并持久化待确认操作。"""
        try:
            from devices.base import DeviceCommand
            from devices.registry import DeviceDriverRegistry
            from core.device_executor import DeviceExecutor

            username = getattr(state, 'username', 'default')
            executor = DeviceExecutor(DeviceDriverRegistry(), username=username)
            result = executor.record_decision(
                device_id,
                DeviceCommand(command=command, params=params),
                decision=decision,
                reason=reason,
                trigger="agent",
                rule_id=rule_id,
                add_pending=decision == "need_confirm",
                capability=capability or params.get("action"),
                policy_context=policy_context,
            )
            logger.info("DeviceAgent 决策已记录: device=%s decision=%s reason=%s", device_id, decision, reason[:80])
            return result.get("pending_id")
        except Exception as e:
            logger.warning("DeviceAgent 写决策日志失败: %s", e)
            return None

    def _do_execute(self, device_id: str, command: str, params: Dict, state: AgentState, engine, rule_id: str = None, extra: str = "", capability: str = None, policy_context: Dict = None) -> AgentState:
        """实际执行设备指令 — 使用共享工厂加载所有驱动(含自定义设备)"""
        try:
            from devices.base import DeviceCommand
            from core.device_executor import DeviceExecutor
            from core.device_registry_factory import setup_registry, close_registry

            username = getattr(state, 'username', 'default')
            registry, loop = setup_registry(username=username)
            try:
                loop.run_until_complete(registry.discover_all())

                cmd = DeviceCommand(command=command, params=params)
                executor = DeviceExecutor(registry, username=username)
                result = executor.execute_sync(
                    device_id, cmd, trigger="agent", rule_id=rule_id, loop=loop,
                    capability=capability, policy_context=policy_context,
                )

                engine.record_execution(device_id, params, success=result.get("success", False))

                state.device_command = {"device_id": device_id, "command": command, "params": params, "action": capability or params.get("action", command)}
                res_obj = result.get("result")
                res_msg = res_obj.message if res_obj and hasattr(res_obj, 'message') else str(res_obj or "")
                state.device_result = {"success": result["success"], "message": res_msg, "decision": result.get("decision", "auto_execute")}

                if result.get("decision") == "need_confirm":
                    if not result.get("pending_id"):
                        state.error_code = "PENDING_ACTION_SAVE_FAILED"
                        return self._reply(state, "哥哥，待确认操作保存失败，请稍后重试。")
                    state.pending_action = {
                        "id": result.get("pending_id"), "device_id": device_id,
                        "command": command, "params": params, "reason": res_msg,
                        "capability": capability, "rule_id": rule_id,
                    }
                    return self._reply(state, f"⚠️ {res_msg}\n\n请在「设备中心」中确认此操作。")

                if result["success"]:
                    msg = f"✅ 指令已执行！\n\n🔧 设备：{device_id}\n⚡ 操作：{command}\n📊 参数：{params}\n📝 结果：{res_msg}\n"
                    if extra:
                        msg += f"\n{extra}"

                    # ── 同步创建任务记录（确保出现在"今日执行记录"中）──
                    try:
                        from core.planting_tracker import PlantingTracker
                        crop = state.short_term_facts.get("crop", "") or params.get("crop", "")
                        action_labels = {
                            "irrigate": "浇水", "fertigate": "施肥", "ventilate": "通风",
                            "light": "补光", "heat": "加热", "cool": "降温", "shade": "遮阳",
                        }
                        # 从 device_command 推断操作类型
                        action_type = state.device_command.get("action", command) if state.device_command else command
                        task_type = "停止设备" if command == "stop" else action_labels.get(action_type, "设备操作")
                        tracker = PlantingTracker(os.path.join(DEFAULT_DATA_DIR, username))
                        tracker.create_task({
                            "crop": crop or "未指定作物",
                            "task_type": task_type,
                            "title": f"[Agent执行] {task_type} ({device_id})",
                            "description": f"LLM Agent 自动执行：{command}，参数：{params}",
                            "status": "已完成",
                            "priority": "medium",
                            "progress_percent": 100,
                            "device_id": device_id,
                            "device_command": command,
                            "device_params": params,
                            "rule_id": rule_id,
                        })
                        logger.info("DeviceAgent 已同步创建任务记录: %s", task_type)
                    except Exception as e:
                        logger.warning("DeviceAgent 创建任务记录失败: %s", e)
                else:
                    msg = f"❌ 执行失败：{res_msg}"

                return self._reply(state, msg)
            finally:
                close_registry(loop, registry)
        except Exception as e:
            logger.exception("设备执行异常")
            return self._reply(state, f"❌ 设备执行出错：{e}")

    def _find_device_for_action(self, action: str, username: str = "default", device_hint: str = "") -> Optional[str]:
        """动态发现匹配操作类型的设备ID。

        从用户注册的设备中查找具备对应能力的设备，
        不再依赖硬编码的虚拟设备ID。
        """
        capability = ACTION_TO_CAPABILITY.get(action)
        if not capability:
            return None
        return _discover_device_for_capability(capability, username, device_hint)
