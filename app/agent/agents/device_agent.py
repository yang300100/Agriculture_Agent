"""设备控制 Agent — 解析用户意图，调度设备操作"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional

from .base import BaseAgent
from ..state import AgentState

logger = logging.getLogger(__name__)

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


def _discover_device_for_capability(capability: str, username: str = "default") -> Optional[str]:
    """动态发现具备指定能力的设备ID（从用户注册的设备列表中查找）。

    每次调用都会重新加载 registry 以获取最新设备列表。
    返回第一个匹配的设备ID，无匹配时返回 None。
    """
    try:
        from core.device_registry_factory import setup_registry, close_registry

        registry, loop = setup_registry(username=username)
        try:
            loop.run_until_complete(registry.discover_all())
            # 枚举所有驱动下的所有设备
            for driver_name in registry._drivers:
                driver = registry._drivers[driver_name]
                devices = loop.run_until_complete(driver.discover())
                for d in devices:
                    # 检查设备能力是否匹配（DeviceCapability 枚举值比较）
                    for cap in d.capabilities:
                        if hasattr(cap, 'value') and cap.value == capability:
                            return d.device_id
                        elif str(cap) == capability:
                            return d.device_id
        finally:
            close_registry(loop, registry)
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
        self._state_username = getattr(state, 'username', 'default')

        try:
            parsed = self._parse_device_intent(question, state)
            if not parsed:
                return self._reply(state, "抱歉哥哥，我没理解你想操作哪个设备呢～能再说详细一点吗？比如「帮小麦浇30分钟水」")

            from core.device_rule_engine import RuleEngine
            username = self._state_username
            engine = RuleEngine(username=username)
            matched_rules = self._match_rules(parsed, state, engine)
            state.matched_rules = [r["id"] for r in matched_rules]

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
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
        from ..config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL

        context = self._get_context(state)

        prompt = f"""分析用户的设备控制需求，提取操作参数。

用户输入："{question}"

上下文：当前作物：{context.get('crop', '未指定')}，地区：{context.get('region', '未指定')}

请以 JSON 格式返回：
{{
    "action": "irrigate|fertigate|ventilate|light|heat|cool|shade|status",
    "device_hint": "设备名或类型关键词（可选）",
    "crop": "目标作物",
    "params": {{"duration": 数字分钟, "amount_kg": 数字kg, "target_temp": 数字°C, "brightness_percent": 数字}},
    "reasoning": "操作理由"
}}

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
            if parsed.get("action") == "unknown":
                return None
            return parsed
        except Exception as e:
            logger.warning("DeviceAgent 意图解析失败: %s, 回退到关键词匹配", e)
            return self._keyword_parse(question)

    def _keyword_parse(self, question: str) -> Optional[Dict]:
        """关键词回退解析 — 先提取时长，再匹配操作类型。
        未识别任何关键词时返回 None（不默认触发灌溉！）。"""
        dur_match = re.search(r'(\d+)\s*(分钟|分|小时|秒)', question)
        duration = int(dur_match.group(1)) if dur_match else 30
        # 小时/秒 转换
        if dur_match and dur_match.group(2) == "小时":
            duration = duration * 60
        elif dur_match and dur_match.group(2) == "秒":
            duration = max(1, duration // 60)

        # 按优先级匹配操作类型
        if any(kw in question for kw in ["浇水", "灌溉", "浇灌"]):
            return {"action": "irrigate", "params": {"duration": duration}}
        if any(kw in question for kw in ["施肥", "追肥"]):
            return {"action": "fertigate", "params": {"amount_kg": 5, "duration": duration}}
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
        if any(kw in question for kw in ["设备状态", "查看设备", "设备列表"]):
            return {"action": "status", "params": {}}
        if any(kw in question for kw in ["关闭", "停止", "关灯", "关机"]):
            return {"action": "stop", "params": {}}
        # 未识别任何操作类型 → 返回 None，不默认触发灌溉
        return None

    def _match_rules(self, parsed: Dict, state: AgentState, engine) -> list:
        """查找与当前操作匹配的规则"""
        try:
            username = getattr(state, 'username', 'default')
            sensor_context = self._get_sensor_context(parsed.get("action", ""), username)
            context = {"sensor_data": sensor_context, "weather": {}, "crop": parsed.get("crop", "")}
            return engine.find_matching_rules(context)
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
                # 无传感器设备注册，返回默认兜底值
                return {"soil_moisture": 45, "temperature": 22, "humidity": 65}

            registry, loop = setup_registry(username=username)
            try:
                loop.run_until_complete(registry.discover_all())
                state = loop.run_until_complete(registry.read_state(sensor_id))
                return state
            finally:
                close_registry(loop, registry)
        except Exception:
            return {"soil_moisture": 45, "temperature": 22, "humidity": 65}

    def _execute_with_rule(self, rule: Dict, parsed: Dict, state: AgentState, engine) -> AgentState:
        """有匹配规则时的执行逻辑"""
        from core.device_rule_engine import RuleDecision, apply_autonomy
        from ..config import get_autonomy_level

        autonomy = get_autonomy_level()
        action = rule.get("action", {})
        device_id = action.get("device_id", "")
        proposed_params = {**action.get("params", {}), **parsed.get("params", {})}
        decision, reason, final_params = engine.evaluate_action(rule, proposed_params, {"device_id": device_id})

        # 应用自主权级别
        decision = apply_autonomy(decision, autonomy)

        if decision == RuleDecision.AUTO_EXECUTE:
            extra = f"✅ 规则「{rule.get('name', '')}」校验通过"
            if autonomy == "high":
                extra += "（高自主模式：自动执行）"
            return self._do_execute(device_id, action.get("command", "start"), final_params, state, engine, rule_id=rule["id"], extra=extra)
        else:
            # 记录未执行决策（need_confirm / rejected）到执行日志
            self._write_decision_log(state, device_id, action.get("command", "start"),
                                     final_params, decision, reason, rule.get("id"))
            if decision == "need_confirm":
                if autonomy == "low":
                    extra_note = "\n（低自主模式：所有操作均需确认）"
                else:
                    extra_note = ""
                state.pending_action = {"device_id": device_id, "command": action.get("command", "start"), "params": final_params, "reason": reason, "rule_id": rule["id"]}
                return self._reply(state, f"⚠️ {reason}\n\n📋 操作预览：{device_id} → {action.get('command')} 参数：{final_params}{extra_note}\n\n请在「设备仪表盘」中确认此操作。")
            else:
                return self._reply(state, f"❌ {reason}")

    def _execute_direct(self, parsed: Dict, state: AgentState, engine) -> AgentState:
        """无匹配规则时的直接执行"""
        from core.device_rule_engine import RuleDecision, apply_autonomy
        from ..config import get_autonomy_level

        autonomy = get_autonomy_level()
        action_type = parsed.get("action", "")
        params = parsed.get("params", {})

        # cool → 关闭加热器（反向控制）— 也需要走规则校验
        if action_type == "cool":
            device_id = self._find_device_for_action("heat")
            if not device_id:
                return self._reply(state, f"😅 没找到加热器设备来执行降温呢～")
            params["action"] = "cool"
            temp_rule = {"id": "temp_direct_cool", "action": {"device_id": device_id, "command": "stop", "params": params}, "constraints": {"max_duration_per_use": 60}}
            decision, reason, final_params = engine.evaluate_action(temp_rule, params, {"device_id": device_id})
            decision = apply_autonomy(decision, autonomy)
            if decision == RuleDecision.REJECTED:
                self._write_decision_log(state, device_id, "stop", final_params, "rejected", reason, None)
                return self._reply(state, f"❌ {reason}")
            elif decision == RuleDecision.NEED_CONFIRM:
                self._write_decision_log(state, device_id, "stop", final_params, "need_confirm", reason, None)
                state.pending_action = {"device_id": device_id, "command": "stop", "params": final_params, "reason": reason}
                return self._reply(state, f"⚠️ {reason}\n\n请在「设备仪表盘」中确认此操作。")
            return self._do_execute(device_id, "stop", final_params, state, engine,
                                   extra="（降温模式：关闭加热器）")

        # shade → 关闭补光灯（反向控制）— 也需要走规则校验
        if action_type == "shade":
            device_id = self._find_device_for_action("light")
            if not device_id:
                return self._reply(state, f"😅 没找到补光灯设备来执行遮阳呢～")
            params["action"] = "shade"
            temp_rule = {"id": "temp_direct_shade", "action": {"device_id": device_id, "command": "stop", "params": params}, "constraints": {"max_duration_per_use": 60}}
            decision, reason, final_params = engine.evaluate_action(temp_rule, params, {"device_id": device_id})
            decision = apply_autonomy(decision, autonomy)
            if decision == RuleDecision.REJECTED:
                self._write_decision_log(state, device_id, "stop", final_params, "rejected", reason, None)
                return self._reply(state, f"❌ {reason}")
            elif decision == RuleDecision.NEED_CONFIRM:
                self._write_decision_log(state, device_id, "stop", final_params, "need_confirm", reason, None)
                state.pending_action = {"device_id": device_id, "command": "stop", "params": final_params, "reason": reason}
                return self._reply(state, f"⚠️ {reason}\n\n请在「设备仪表盘」中确认此操作。")
            return self._do_execute(device_id, "stop", final_params, state, engine,
                                   extra="（遮阳模式：关闭补光灯）")

        device_id = self._find_device_for_action(action_type)
        if not device_id:
            return self._reply(state, f"😅 没找到{action_type}类型的设备呢～请先在「设备仪表盘」中添加设备吧！")

        temp_rule = {"id": "temp_direct", "action": {"device_id": device_id, "command": "start", "params": params}, "constraints": {"max_duration_per_use": 60, "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5]}}
        decision, reason, final_params = engine.evaluate_action(temp_rule, params, {"device_id": device_id})

        # 应用自主权级别
        decision = apply_autonomy(decision, autonomy)

        if decision == RuleDecision.REJECTED:
            self._write_decision_log(state, device_id, "start", final_params, "rejected", reason, None)
            return self._reply(state, f"❌ {reason}")
        elif decision == RuleDecision.NEED_CONFIRM:
            self._write_decision_log(state, device_id, "start", final_params, "need_confirm", reason, None)
            state.pending_action = {"device_id": device_id, "command": "start", "params": final_params, "reason": reason}
            return self._reply(state, f"⚠️ {reason}\n\n请在「设备仪表盘」中确认此操作。")

        return self._do_execute(device_id, "start", final_params, state, engine)

    def _write_decision_log(self, state: AgentState, device_id: str, command: str,
                            params: Dict, decision: str, reason: str, rule_id: str = None):
        """将 Agent 决策写入设备执行日志（包括未执行的 need_confirm/rejected）"""
        try:
            from datetime import datetime
            username = getattr(state, 'username', 'default')
            log_path = os.path.join("data", username, "device_log.json")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            logs = []
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                except Exception:
                    pass

            logs.append({
                "timestamp": datetime.now().isoformat(),
                "device_id": device_id,
                "command": command,
                "params": params,
                "trigger": "agent",
                "rule_id": rule_id,
                "decision": decision,
                "success": decision == "auto_execute",
                "attempts": 1,
                "message": reason,
                "error_code": "",
            })

            # 最多保留 500 条
            if len(logs) > 500:
                logs = logs[-500:]

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)

            logger.info("DeviceAgent 决策已记录: device=%s decision=%s reason=%s", device_id, decision, reason[:80])
        except Exception as e:
            logger.warning("DeviceAgent 写决策日志失败: %s", e)

    def _do_execute(self, device_id: str, command: str, params: Dict, state: AgentState, engine, rule_id: str = None, extra: str = "") -> AgentState:
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
                result = executor.execute_sync(device_id, cmd, trigger="agent", rule_id=rule_id)

                engine.record_execution(device_id, params, success=result.get("success", False))

                state.device_command = {"device_id": device_id, "command": command, "params": params, "action": params.get("action", command)}
                res_obj = result.get("result")
                res_msg = res_obj.message if res_obj and hasattr(res_obj, 'message') else str(res_obj or "")
                state.device_result = {"success": result["success"], "message": res_msg}

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
                        task_type = action_labels.get(action_type, "设备操作")
                        tracker = PlantingTracker(os.path.join("data", username))
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

    def _find_device_for_action(self, action: str) -> Optional[str]:
        """动态发现匹配操作类型的设备ID。

        从用户注册的设备中查找具备对应能力的设备，
        不再依赖硬编码的虚拟设备ID。
        """
        capability = ACTION_TO_CAPABILITY.get(action)
        if not capability:
            return None
        username = getattr(self, '_state_username', 'default')
        return _discover_device_for_capability(capability, username)
