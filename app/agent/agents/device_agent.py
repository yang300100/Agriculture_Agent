"""设备控制 Agent — 解析用户意图，调度设备操作"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional

from .base import BaseAgent
from ..state import AgentState

logger = logging.getLogger(__name__)

# ── 设备动作类型 → 默认设备ID 映射（供 extract_tasks 等模块复用）──
ACTION_TO_DEFAULT_DEVICE = {
    "irrigate": "virtual_irrigation_01",
    "fertigate": "virtual_fertigator_01",
    "ventilate": "virtual_ventilation_01",
    "light": "virtual_light_01",
    "heat": "virtual_heater_01",
    "cool": "virtual_heater_01",   # 降温复用加热器（反转控制逻辑）
    "shade": "virtual_light_01",   # 遮阳复用补光灯（反转控制逻辑）
}


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
            parsed = self._parse_device_intent(question, state)
            if not parsed:
                return self._reply(state, "抱歉哥哥，我没理解你想操作哪个设备呢～能再说详细一点吗？比如「帮小麦浇30分钟水」")

            from core.device_rule_engine import RuleEngine
            username = getattr(state, 'username', 'default')
            engine = RuleEngine(username=username)
            matched_rules = self._match_rules(parsed, state, engine)
            state.matched_rules = [r["id"] for r in matched_rules]

            if matched_rules:
                return self._execute_with_rule(matched_rules[0], parsed, state, engine)
            else:
                return self._execute_direct(parsed, state, engine)

        except Exception as e:
            logger.exception("DeviceAgent 处理失败")
            return self._reply(state, f"设备控制出错了：{e}")

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
        """关键词回退解析 — 先提取时长，再匹配操作类型"""
        dur_match = re.search(r'(\d+)\s*(分钟|分|小时|秒)', question)
        duration = int(dur_match.group(1)) if dur_match else 30
        # 小时/秒 转换
        if dur_match and dur_match.group(2) == "小时":
            duration = duration * 60
        elif dur_match and dur_match.group(2) == "秒":
            duration = max(1, duration // 60)

        if any(kw in question for kw in ["浇水", "灌溉"]):
            return {"action": "irrigate", "params": {"duration": duration}}
        if any(kw in question for kw in ["施肥"]):
            return {"action": "fertigate", "params": {"amount_kg": 5}}
        if any(kw in question for kw in ["通风", "开窗"]):
            return {"action": "ventilate", "params": {"duration": duration}}
        if any(kw in question for kw in ["补光", "开灯"]):
            return {"action": "light", "params": {"brightness_percent": 80}}
        if any(kw in question for kw in ["加热"]):
            return {"action": "heat", "params": {"target_temp": 22}}
        if any(kw in question for kw in ["设备状态"]):
            return {"action": "status", "params": {}}
        return {"action": "irrigate", "params": {"duration": duration}}

    def _match_rules(self, parsed: Dict, state: AgentState, engine) -> list:
        """查找与当前操作匹配的规则"""
        try:
            sensor_context = self._get_sensor_context(parsed.get("action", ""))
            context = {"sensor_data": sensor_context, "weather": {}, "crop": parsed.get("crop", "")}
            return engine.find_matching_rules(context)
        except Exception as e:
            logger.warning("规则匹配失败: %s", e)
            return []

    def _get_sensor_context(self, action: str) -> Dict:
        """获取当前传感器数据"""
        loop = None
        try:
            from devices.simulator_driver import SimulatorDriver
            import asyncio
            sim = SimulatorDriver(simulated_latency_ms=0)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(sim.connect())
            state = loop.run_until_complete(sim.read_state("virtual_soil_sensor_01"))
            return state
        except Exception:
            return {"soil_moisture": 45, "temperature": 22, "humidity": 65}
        finally:
            if loop is not None:
                from core.device_registry_factory import close_registry
                close_registry(loop)

    def _execute_with_rule(self, rule: Dict, parsed: Dict, state: AgentState, engine) -> AgentState:
        """有匹配规则时的执行逻辑"""
        from core.device_rule_engine import apply_autonomy
        from ..config import get_autonomy_level

        autonomy = get_autonomy_level()
        action = rule.get("action", {})
        proposed_params = {**action.get("params", {}), **parsed.get("params", {})}
        decision, reason, final_params = engine.evaluate_action(rule, proposed_params, {"device_id": action.get("device_id", "")})

        # 应用自主权级别
        decision = apply_autonomy(decision, autonomy)

        if decision == "auto_execute":
            extra = f"✅ 规则「{rule.get('name', '')}」校验通过"
            if autonomy == "high":
                extra += "（高自主模式：自动执行）"
            return self._do_execute(action.get("device_id", ""), action.get("command", "start"), final_params, state, engine, rule_id=rule["id"], extra=extra)
        elif decision == "need_confirm":
            if autonomy == "low":
                extra_note = "\n（低自主模式：所有操作均需确认）"
            else:
                extra_note = ""
            state.pending_action = {"device_id": action.get("device_id"), "command": action.get("command", "start"), "params": final_params, "reason": reason, "rule_id": rule["id"]}
            return self._reply(state, f"⚠️ {reason}\n\n📋 操作预览：{action.get('device_id')} → {action.get('command')} 参数：{final_params}{extra_note}\n\n请在「设备仪表盘」中确认此操作。")
        else:
            return self._reply(state, f"❌ {reason}")

    def _execute_direct(self, parsed: Dict, state: AgentState, engine) -> AgentState:
        """无匹配规则时的直接执行"""
        from core.device_rule_engine import RuleDecision, apply_autonomy
        from ..config import get_autonomy_level

        autonomy = get_autonomy_level()
        action_type = parsed.get("action", "")
        params = parsed.get("params", {})
        device_id = self._find_device_for_action(action_type)
        if not device_id:
            return self._reply(state, f"😅 没找到{action_type}类型的设备呢～请先在「设备仪表盘」中添加设备吧！")

        temp_rule = {"id": "temp_direct", "action": {"device_id": device_id, "command": "start", "params": params}, "constraints": {"max_duration_per_use": 60, "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5]}}
        decision, reason, final_params = engine.evaluate_action(temp_rule, params, {"device_id": device_id})

        # 应用自主权级别
        decision = apply_autonomy(decision, autonomy)

        if decision == RuleDecision.REJECTED:
            return self._reply(state, f"❌ {reason}")
        elif decision == RuleDecision.NEED_CONFIRM:
            state.pending_action = {"device_id": device_id, "command": "start", "params": final_params, "reason": reason}
            return self._reply(state, f"⚠️ {reason}\n\n请在「设备仪表盘」中确认此操作。")

        return self._do_execute(device_id, "start", final_params, state, engine)

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
                executor = DeviceExecutor(registry)
                result = executor.execute_sync(device_id, cmd, trigger="agent", rule_id=rule_id)

                engine.record_execution(device_id, params)

                state.device_command = {"device_id": device_id, "command": command, "params": params}
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
                        task_type = action_labels.get(
                            state.device_command.get("action", "") if state.device_command else "",
                            "设备操作"
                        )
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
                        })
                        logger.info("DeviceAgent 已同步创建任务记录: %s", task_type)
                    except Exception as e:
                        logger.warning("DeviceAgent 创建任务记录失败: %s", e)
                else:
                    msg = f"❌ 执行失败：{res_msg}"

                return self._reply(state, msg)
            finally:
                close_registry(loop)
        except Exception as e:
            logger.exception("设备执行异常")
            return self._reply(state, f"❌ 设备执行出错：{e}")

    def _find_device_for_action(self, action: str) -> Optional[str]:
        return ACTION_TO_DEFAULT_DEVICE.get(action)
