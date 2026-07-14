"""规则编辑器 — 自动规则的 CRUD 管理"""

import json
import streamlit as st
from app.api_client import api, invalidate_cache


def render_rules_page():
    """渲染规则编辑器"""
    st.markdown("## 📋 规则管理")

    rules = api("/api/rules") or []
    devices = api("/api/devices") or []

    # ── 规则列表 + 编辑区 ──────────────────────
    col_list, col_edit = st.columns([1, 2])

    with col_list:
        st.markdown("### 我的规则")

        if not rules:
            st.info("还没有规则，点击右侧创建～")
        else:
            for rule in rules:
                enabled = rule.get("enabled", True)
                icon = "✅" if enabled else "⏸️"
                with st.container():
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        if st.button(f"{icon} {rule.get('name', '未命名')}", key=f"select_{rule['id']}"):
                            st.session_state.selected_rule = rule
                            st.rerun()
                    with c2:
                        if st.button("🗑️", key=f"del_{rule['id']}"):
                            api(f"/api/rules/{rule['id']}", method="delete")
                            invalidate_cache("/api/rules")
                            if st.session_state.get("selected_rule", {}).get("id") == rule["id"]:
                                st.session_state.selected_rule = None
                            st.rerun()

    with col_edit:
        selected = st.session_state.get("selected_rule")

        if selected:
            st.markdown(f"### ✏️ 编辑: {selected.get('name', '未命名')}")
            _render_rule_form(selected, devices)
        else:
            st.markdown("### ➕ 新建规则")
            st.caption("从左侧选择一个规则编辑，或填写下方表单新建")
            _render_rule_form(None, devices)


def _render_rule_form(rule, devices):
    """渲染规则编辑表单"""
    is_new = rule is None
    form_key = f"rule_form_{rule['id'] if rule else 'new'}"

    with st.form(key=form_key):
        name = st.text_input("规则名称", value=rule.get("name", "") if rule else "",
                             placeholder="如：小麦自动灌溉")

        enabled = st.checkbox("启用规则", value=rule.get("enabled", True) if rule else True)

        st.markdown("**触发条件**")
        trigger_logic = st.radio("逻辑", ["AND", "OR"], horizontal=True, key=f"logic_{rule['id'] if rule else 'new'}")

        trigger_conditions = rule.get("trigger", {}).get("conditions", [
            {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30},
        ]) if rule else [{"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30}]

        trigger_json = st.text_area(
            "触发条件 (JSON)",
            value=json.dumps(trigger_conditions, ensure_ascii=False, indent=2),
            height=150,
            key=f"trigger_{rule['id'] if rule else 'new'}"
        )

        st.markdown("**执行动作**")
        device_ids = [d["device_id"] for d in devices] if devices else []
        if not device_ids:
            st.warning("暂无可用设备，请先在「设备仪表盘」中注册设备后再创建规则")
            st.stop()
        default_dev = rule["action"]["device_id"] if rule and rule.get("action", {}).get("device_id") in device_ids else device_ids[0]
        device_id = st.selectbox("目标设备", device_ids,
                                 index=device_ids.index(default_dev) if default_dev in device_ids else 0,
                                 key=f"dev_{rule['id'] if rule else 'new'}")

        command = st.selectbox("指令", ["start", "stop", "set_param"],
                               index=0, key=f"cmd_{rule['id'] if rule else 'new'}")

        st.markdown("**安全边界**")
        c1, c2 = st.columns(2)
        with c1:
            max_dur = st.number_input("单次最长(分)", 1, 120,
                                      value=rule.get("constraints", {}).get("max_duration_per_use", 60) if rule else 60,
                                      key=f"maxdur_{rule['id'] if rule else 'new'}")
        with c2:
            max_daily = st.number_input("每日上限(分)", 1, 600,
                                        value=rule.get("constraints", {}).get("max_duration_per_day", 180) if rule else 180,
                                        key=f"maxday_{rule['id'] if rule else 'new'}")

        ai_enabled = st.checkbox("启用 AI 微调",
                                 value=rule.get("ai_enhance", {}).get("enabled", False) if rule else False,
                                 key=f"ai_{rule['id'] if rule else 'new'}")

        submitted = st.form_submit_button("💾 保存规则")

        if submitted:
            try:
                conditions = json.loads(trigger_json)
            except json.JSONDecodeError:
                st.error("触发条件 JSON 格式错误，请检查！")
                return

            new_rule = {
                "name": name or "未命名规则",
                "enabled": enabled,
                "trigger": {"logic": trigger_logic, "conditions": conditions},
                "action": {"device_id": device_id, "command": command, "params": {"duration": 30}},
                "constraints": {
                    "max_duration_per_use": max_dur,
                    "max_duration_per_day": max_daily,
                    "min_interval_minutes": 120,
                    "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5],
                },
                "ai_enhance": {
                    "enabled": ai_enabled,
                    "can_adjust": ["duration"],
                    "adjust_range": {"duration": [-10, 10]},
                },
            }

            if not is_new:
                new_rule["id"] = rule["id"]
                result = api(f"/api/rules/{rule['id']}", method="put", json_data=new_rule)
            else:
                result = api("/api/rules", method="post", json_data=new_rule)

            if result and result.get("success"):
                invalidate_cache("/api/rules")
                st.success("规则已保存！")
                st.rerun()
            else:
                st.error(f"保存失败: {result.get('error', '未知错误') if result else '无响应'}")

    # 测试按钮（仅编辑已有规则时显示）
    if not is_new and rule:
        if st.button("▶️ 测试规则（仅评估不执行）", key=f"test_{rule['id']}"):
            result = api(f"/api/rules/{rule['id']}/test", method="post")
            if result and result.get("success"):
                if result.get("rule_matched"):
                    st.success("✅ 规则条件匹配！传感器快照：")
                    st.json(result.get("sensor_snapshot", {}))
                else:
                    st.warning("❌ 条件不满足，规则不会触发")
            else:
                st.error(f"测试失败: {result.get('error', '') if result else '无响应'}")
