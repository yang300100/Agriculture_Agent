"""设备安全策略页面。"""

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

SCOPE_LABELS = {
    "global": "全部设备",
    "capability": "某类设备能力",
    "device": "单台设备",
    "plot": "指定地块",
    "zone": "指定作业分区（预留）",
}


def render_safety_page():
    """渲染安全策略管理页。"""
    st.markdown("## 🛡️ 设备安全策略")
    st.caption(
        "所有手动、自动化规则、LLM 自主巡检和 API 操作都会经过这里。"
        "用户策略可在设备物理上限以内调整。"
    )

    catalog = api("/api/safety-policies/catalog", cache_ttl=300) or {}
    policies = api("/api/safety-policies", cache_ttl=5) or []
    devices = api("/api/devices", cache_ttl=10) or []
    fields = api("/api/fields", cache_ttl=30) or []

    with st.expander("查看设备物理绝对上限", expanded=False):
        ceilings = catalog.get("absolute_ceilings", {})
        rows = []
        for capability, limits in ceilings.items():
            rows.append({
                "能力": CAPABILITY_LABELS.get(capability, capability),
                "单次最长（分钟）": limits.get("max_duration_per_use_minutes", "—"),
                "单次最大用量（kg）": limits.get("max_amount_per_use_kg", "—"),
                "最小间隔（秒）": limits.get("min_interval_seconds", "—"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.info("物理上限不可由普通策略突破；如设备规格变化，请通过部署环境变量调整。")

    left, right = st.columns([1, 2])
    with left:
        st.markdown("### 已配置策略")
        if not policies:
            st.info("尚未配置用户安全策略，当前仅应用设备物理上限。")
        for policy in policies:
            icon = "✅" if policy.get("enabled", True) else "⏸️"
            c1, c2 = st.columns([4, 1])
            with c1:
                if st.button(
                    f"{icon} {policy.get('name', '未命名策略')}",
                    key=f"select_safety_{policy['id']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_safety_policy = policy
                    st.rerun()
            with c2:
                if st.button("🗑️", key=f"delete_safety_{policy['id']}"):
                    api(f"/api/safety-policies/{policy['id']}", method="delete")
                    invalidate_cache("/api/safety-policies")
                    st.session_state.selected_safety_policy = None
                    st.rerun()
        if st.button("➕ 新建安全策略", use_container_width=True):
            st.session_state.selected_safety_policy = None
            st.rerun()

    with right:
        _render_policy_form(
            st.session_state.get("selected_safety_policy"),
            devices,
            fields,
            catalog,
        )


def _render_policy_form(policy, devices, fields, catalog):
    is_new = not policy
    policy = policy or {}
    title = "新建安全策略" if is_new else f"编辑：{policy.get('name', '')}"
    st.markdown(f"### {title}")
    limits = policy.get("limits", {}) or {}
    form_key = f"safety_form_{policy.get('id', 'new')}"

    with st.form(form_key):
        name = st.text_input(
            "策略名称",
            value=policy.get("name", ""),
            placeholder="例如：A区灌溉安全上限",
        )
        enabled = st.checkbox("启用策略", value=policy.get("enabled", True))

        scope_options = list(SCOPE_LABELS)
        current_scope = policy.get("scope_type", "capability")
        scope_type = st.selectbox(
            "适用范围",
            scope_options,
            index=scope_options.index(current_scope) if current_scope in scope_options else 1,
            format_func=lambda value: SCOPE_LABELS[value],
        )

        capability_options = list(CAPABILITY_LABELS)
        current_capability = policy.get("capability") or "irrigate"
        capability = st.selectbox(
            "设备能力",
            capability_options,
            index=(capability_options.index(current_capability)
                   if current_capability in capability_options else 0),
            format_func=lambda value: CAPABILITY_LABELS[value],
        )

        device_id = ""
        plot_id = None
        zone_id = ""
        if scope_type == "device":
            options = [row.get("device_id", "") for row in devices if row.get("device_id")]
            if not options:
                st.warning("暂无设备，请先注册设备。")
            current = policy.get("device_id", "")
            device_id = st.selectbox(
                "目标设备", options or [""],
                index=options.index(current) if current in options else 0,
            )
        elif scope_type in {"plot", "zone"}:
            field_map = {
                f"{row.get('name', '未命名')}（ID={row.get('id', '')}）": row.get("id")
                for row in fields
            }
            labels = list(field_map) or ["暂无地块"]
            current_plot = str(policy.get("plot_id", ""))
            current_label = next(
                (label for label, value in field_map.items()
                 if str(value) == current_plot),
                labels[0],
            )
            selected = st.selectbox(
                "所属地块", labels,
                index=labels.index(current_label),
            )
            plot_id = field_map.get(selected)
            if scope_type == "zone":
                zone_id = st.text_input(
                    "作业分区ID",
                    value=policy.get("zone_id", ""),
                    help="分区数据模型落地前先作为预留字段使用。",
                )

        st.markdown("**可调整硬限制**")
        ceiling = catalog.get("absolute_ceilings", {}).get(capability, {})
        max_physical_duration = int(ceiling.get("max_duration_per_use_minutes", 1440))
        c1, c2 = st.columns(2)
        with c1:
            max_duration = st.number_input(
                "单次最长（分钟，0=不额外限制）",
                min_value=0,
                max_value=max_physical_duration,
                value=int(limits.get("max_duration_per_use_minutes", 0)),
            )
            max_daily_duration = st.number_input(
                "每日最长（分钟，0=不额外限制）",
                min_value=0,
                max_value=100000,
                value=int(limits.get("max_duration_per_day_minutes", 0)),
            )
        with c2:
            min_interval = st.number_input(
                "最小操作间隔（分钟，0=不限制）",
                min_value=0,
                max_value=10080,
                value=int(limits.get("min_interval_minutes", 0)),
            )
            forbidden_hours = st.multiselect(
                "禁止运行小时",
                list(range(24)),
                default=limits.get("forbidden_hours", []),
                format_func=lambda hour: f"{hour:02d}:00",
            )

        max_volume = 0.0
        max_daily_volume = 0.0
        rated_flow = 0.0
        max_amount = 0.0
        max_daily_amount = 0.0
        if capability == "irrigate":
            c1, c2, c3 = st.columns(3)
            with c1:
                max_volume = st.number_input(
                    "单次最大水量（L）", min_value=0.0,
                    value=float(limits.get("max_volume_per_use_liters", 0)),
                )
            with c2:
                max_daily_volume = st.number_input(
                    "每日最大水量（L）", min_value=0.0,
                    value=float(limits.get("max_volume_per_day_liters", 0)),
                )
            with c3:
                rated_flow = st.number_input(
                    "标称流量（L/min）", min_value=0.0,
                    value=float(limits.get("rated_flow_lpm", 0)),
                    help="没有实时流量计时，用标称流量估算灌溉水量。",
                )
        if capability == "fertigate":
            c1, c2 = st.columns(2)
            physical_amount = float(ceiling.get("max_amount_per_use_kg", 50))
            with c1:
                max_amount = st.number_input(
                    "单次最大施肥量（kg）", min_value=0.0,
                    max_value=physical_amount,
                    value=float(limits.get("max_amount_per_use_kg", 0)),
                )
            with c2:
                max_daily_amount = st.number_input(
                    "每日最大施肥量（kg）", min_value=0.0,
                    value=float(limits.get("max_amount_per_day_kg", 0)),
                )

        require_sensor = st.checkbox(
            "缺少传感器数据时不允许自动执行",
            value=bool(limits.get("require_sensor_data", False)),
        )
        violation_action = st.radio(
            "超过用户限制时",
            ["reject", "confirm"],
            index=0 if policy.get("violation_action", "reject") == "reject" else 1,
            horizontal=True,
            format_func=lambda value: "拒绝执行" if value == "reject" else "请求确认",
        )

        submitted = st.form_submit_button("💾 保存安全策略", use_container_width=True)
        if submitted:
            saved_limits = {
                "forbidden_hours": forbidden_hours,
                "require_sensor_data": require_sensor,
            }
            numeric = {
                "max_duration_per_use_minutes": max_duration,
                "max_duration_per_day_minutes": max_daily_duration,
                "min_interval_minutes": min_interval,
                "max_volume_per_use_liters": max_volume,
                "max_volume_per_day_liters": max_daily_volume,
                "rated_flow_lpm": rated_flow,
                "max_amount_per_use_kg": max_amount,
                "max_amount_per_day_kg": max_daily_amount,
            }
            saved_limits.update({key: value for key, value in numeric.items() if value > 0})
            payload = {
                "name": name or "未命名安全策略",
                "enabled": enabled,
                "scope_type": scope_type,
                "capability": capability,
                "device_id": device_id,
                "plot_id": plot_id,
                "zone_id": zone_id,
                "limits": saved_limits,
                "violation_action": violation_action,
            }
            if is_new:
                result = api("/api/safety-policies", method="post", json_data=payload)
            else:
                result = api(
                    f"/api/safety-policies/{policy['id']}",
                    method="put",
                    json_data=payload,
                )
            if result and result.get("success"):
                invalidate_cache("/api/safety-policies")
                st.success("安全策略已保存。")
                st.rerun()
            else:
                st.error(result.get("error", "保存失败") if result else "保存失败")
