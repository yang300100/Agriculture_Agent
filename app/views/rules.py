"""自动化规则编辑器。"""

import json

import streamlit as st

from app.api_client import api, invalidate_cache


CAPABILITY_LABELS = {
    "irrigate": "灌溉",
    "fertigate": "施肥",
    "ventilate": "通风",
    "light": "补光",
    "heat": "加热",
    "cool": "降温",
    "shade": "遮阳",
}

COMMAND_LABELS = {
    "start": "启动并执行",
    "stop": "停止设备",
    "set_param": "设置单个参数",
}

EXECUTION_LABELS = {
    "auto": "自动执行",
    "confirm": "执行前确认",
    "notify": "仅通知，不操作设备",
}


def render_rules_page():
    """渲染自动化规则页面。"""
    st.markdown("## 📋 自动化规则")
    st.caption(
        "这里负责设置“何时触发、操作哪个设备、使用什么参数”。"
        "统一运行边界请前往「安全策略」页面配置。"
    )

    rules = api("/api/rules", cache_ttl=5) or []
    devices = api("/api/devices", cache_ttl=10) or []
    catalog = api("/api/device-actions/catalog", cache_ttl=300) or {}

    col_list, col_edit = st.columns([1, 2])
    with col_list:
        st.markdown("### 我的自动化")
        if not rules:
            st.info("还没有自动化规则。")
        for rule in rules:
            enabled = rule.get("enabled", True)
            icon = "✅" if enabled else "⏸️"
            c1, c2 = st.columns([4, 1])
            with c1:
                if st.button(
                    f"{icon} {rule.get('name', '未命名')}",
                    key=f"select_rule_{rule['id']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_rule = rule
                    st.rerun()
            with c2:
                if st.button("🗑️", key=f"delete_rule_{rule['id']}"):
                    api(f"/api/rules/{rule['id']}", method="delete")
                    invalidate_cache("/api/rules")
                    st.session_state.selected_rule = None
                    st.rerun()
        if st.button("➕ 新建自动化规则", use_container_width=True):
            st.session_state.selected_rule = None
            st.rerun()

    with col_edit:
        selected = st.session_state.get("selected_rule")
        _render_rule_form(selected, devices, catalog)


def _render_rule_form(rule, devices, catalog):
    is_new = rule is None
    rule = rule or {}
    action = rule.get("action", {}) or {}
    current_params = action.get("params", {}) or {}
    title = "新建自动化规则" if is_new else f"编辑：{rule.get('name', '')}"
    st.markdown(f"### {title}")

    if not devices:
        st.warning("暂无设备，请先在「设备仪表盘」注册设备。")
        return

    form_key = f"automation_rule_{rule.get('id', 'new')}"
    with st.form(form_key):
        name = st.text_input(
            "规则名称",
            value=rule.get("name", ""),
            placeholder="例如：A区土壤过干自动灌溉",
        )
        enabled = st.checkbox("启用规则", value=rule.get("enabled", True))

        st.markdown("**1. 触发条件**")
        current_logic = str(rule.get("trigger", {}).get("logic", "AND")).upper()
        trigger_logic = st.radio(
            "多个条件之间",
            ["AND", "OR"],
            index=0 if current_logic == "AND" else 1,
            horizontal=True,
            format_func=lambda value: "全部满足（AND）" if value == "AND" else "任一满足（OR）",
        )
        default_conditions = [
            {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30}
        ]
        trigger_conditions = rule.get("trigger", {}).get(
            "conditions", default_conditions
        )
        trigger_json = st.text_area(
            "条件 JSON",
            value=json.dumps(trigger_conditions, ensure_ascii=False, indent=2),
            height=155,
            help=(
                "传感器字段既可写 soil_moisture，也可写 "
                "sensor_01.soil_moisture，以区分多个传感器。"
            ),
        )

        st.markdown("**2. 执行目标**")
        device_map = {row["device_id"]: row for row in devices if row.get("device_id")}
        device_ids = list(device_map)
        current_device = action.get("device_id", "")
        device_id = st.selectbox(
            "目标设备",
            device_ids,
            index=device_ids.index(current_device) if current_device in device_ids else 0,
            format_func=lambda value: (
                f"{device_map[value].get('name', value)}（{value}）"
            ),
        )
        selected_device = device_map[device_id]
        supported = [
            value for value in selected_device.get("capabilities", [])
            if value in catalog
        ]
        if not supported:
            st.error("该设备没有可用于自动化控制的能力。")
            st.form_submit_button("💾 保存规则", disabled=True)
            return

        current_capability = action.get("capability", "")
        capability = st.selectbox(
            "执行能力",
            supported,
            index=supported.index(current_capability) if current_capability in supported else 0,
            format_func=lambda value: CAPABILITY_LABELS.get(value, value),
        )
        commands = catalog[capability].get("commands", ["start", "stop", "set_param"])
        current_command = action.get("command", "start")
        command = st.selectbox(
            "动作",
            commands,
            index=commands.index(current_command) if current_command in commands else 0,
            format_func=lambda value: COMMAND_LABELS.get(value, value),
        )

        st.markdown("**3. 动作参数**")
        params, ai_enhance = _render_action_parameters(
            capability,
            command,
            catalog[capability].get("parameters", {}),
            current_params,
            rule.get("ai_enhance", {}),
            form_key,
        )

        st.markdown("**4. 执行方式**")
        modes = list(EXECUTION_LABELS)
        current_mode = rule.get("execution_mode", "auto")
        execution_mode = st.radio(
            "条件满足后",
            modes,
            index=modes.index(current_mode) if current_mode in modes else 0,
            horizontal=True,
            format_func=lambda value: EXECUTION_LABELS[value],
        )
        st.info("动作参数还会经过「安全策略」统一校验，自动化规则无法突破安全上限。")

        submitted = st.form_submit_button("💾 保存规则", use_container_width=True)
        if submitted:
            try:
                conditions = json.loads(trigger_json)
                if not isinstance(conditions, list):
                    raise ValueError("条件必须是 JSON 数组")
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"触发条件格式错误：{exc}")
                return

            payload = {
                "name": name or "未命名自动化规则",
                "enabled": enabled,
                "trigger": {"logic": trigger_logic, "conditions": conditions},
                "action": {
                    "device_id": device_id,
                    "capability": capability,
                    "command": command,
                    "params": params,
                },
                # 新规则不再重复维护安全边界；保留空对象兼容旧结构。
                "constraints": {},
                "ai_enhance": ai_enhance,
                "execution_mode": execution_mode,
            }
            if is_new:
                result = api("/api/rules", method="post", json_data=payload)
            else:
                result = api(
                    f"/api/rules/{rule['id']}",
                    method="put",
                    json_data=payload,
                )
            if result and result.get("success"):
                invalidate_cache("/api/rules")
                st.success("自动化规则已保存。")
                st.rerun()
            else:
                st.error(result.get("error", "保存失败") if result else "保存失败")

    if not is_new:
        if st.button("▶️ 使用真实设备快照测试（仅评估，不执行）", key=f"test_rule_{rule['id']}"):
            result = api(f"/api/rules/{rule['id']}/test", method="post")
            if result and result.get("success"):
                if result.get("rule_matched"):
                    st.success("条件匹配，真实运行时将进入安全策略校验。")
                else:
                    st.warning("当前真实传感器数据不满足条件。")
                st.json(result.get("sensor_snapshot", {}))
            else:
                st.error(result.get("error", "测试失败") if result else "测试失败")


def _render_action_parameters(capability, command, specs, current, current_ai, key_prefix):
    if command == "stop":
        st.caption("停止属于减险操作，不需要运行参数。")
        return {}, {"enabled": False}

    parameter_names = list(specs)
    if command == "set_param":
        current_key = next((key for key in current if key in specs), parameter_names[0])
        selected = st.selectbox(
            "要设置的参数",
            parameter_names,
            index=parameter_names.index(current_key),
            format_func=lambda value: f"{specs[value]['label']}（{value}）",
            key=f"{key_prefix}_set_param_name",
        )
        spec = specs[selected]
        value = st.number_input(
            f"{spec['label']}（{spec.get('unit', '')}）",
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(current.get(selected, spec["default"])),
            key=f"{key_prefix}_set_param_value",
        )
        return {selected: value}, {"enabled": False}

    mode = st.radio(
        "参数来源",
        ["fixed", "ai_range"],
        index=1 if current_ai.get("enabled") else 0,
        horizontal=True,
        format_func=lambda value: (
            "固定参数" if value == "fixed" else "允许智能规划器在限定范围内调整"
        ),
        key=f"{key_prefix}_parameter_mode",
    )
    if mode == "ai_range":
        st.caption(
            "这里定义可调整范围；只有上层智能规划器提供新参数时才会裁剪，"
            "普通定时触发仍使用下方基准值。"
        )
    params = {}
    selected_fields = st.multiselect(
        "本次要设置的参数",
        parameter_names,
        default=[key for key in current if key in specs] or [parameter_names[0]],
        format_func=lambda value: f"{specs[value]['label']}（{value}）",
        key=f"{key_prefix}_selected_params",
    )
    for name in selected_fields:
        spec = specs[name]
        value = st.number_input(
            f"{spec['label']}（{spec.get('unit', '')}）",
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(current.get(name, spec["default"])),
            key=f"{key_prefix}_{capability}_{name}",
        )
        params[name] = value

    if mode == "ai_range":
        adjustable = st.selectbox(
            "允许AI调整的参数",
            selected_fields or parameter_names,
            key=f"{key_prefix}_ai_field",
        )
        spec = specs[adjustable]
        base = float(params.get(adjustable, spec["default"]))
        suggested_lower = max(float(spec["min"]), min(base * 0.8, base * 1.2))
        suggested_upper = min(float(spec["max"]), max(base * 0.8, base * 1.2))
        lower, upper = st.slider(
            "AI允许范围",
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=(
                suggested_lower,
                suggested_upper,
            ),
            key=f"{key_prefix}_ai_range",
        )
        ai_enhance = {
            "enabled": True,
            "can_adjust": [adjustable],
            "absolute_range": {adjustable: [lower, upper]},
        }
    else:
        ai_enhance = {"enabled": False}
    return params, ai_enhance
