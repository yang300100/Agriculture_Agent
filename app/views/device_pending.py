"""待确认设备操作组件——支持安全编辑、确认、重试与拒绝。"""

import json

import streamlit as st

from app.api_client import api, invalidate_cache


def _clear_editor(action_id: str) -> None:
    """清理当前操作的编辑状态，避免切换操作时串用参数。"""
    st.session_state.pop("_pending_edit_id", None)
    st.session_state.pop(f"_pending_params_{action_id}", None)


def _render_parameter_editor(action: dict) -> None:
    action_id = str(action.get("id", ""))
    params_key = f"_pending_params_{action_id}"
    if params_key not in st.session_state:
        st.session_state[params_key] = json.dumps(
            action.get("params") or {}, ensure_ascii=False, indent=2
        )

    st.caption("保存后不会立即执行；点击确认时仍会重新检查物理上限和安全策略。")
    raw_params = st.text_area(
        "操作参数（JSON 对象）",
        key=params_key,
        height=140,
    )
    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button(
            "💾 保存参数",
            key=f"save_pending_{action_id}",
            use_container_width=True,
        ):
            try:
                params = json.loads(raw_params)
                if not isinstance(params, dict):
                    raise ValueError("参数必须是 JSON 对象")
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"参数格式错误：{exc}")
            else:
                result = api(
                    f"/api/actions/{action_id}",
                    method="put",
                    json_data={"params": params},
                )
                if result and result.get("success"):
                    _clear_editor(action_id)
                    invalidate_cache("/api/actions/pending")
                    st.success("参数已保存，确认执行时会重新进行安全检查。")
                    st.rerun()
                elif result:
                    st.error(result.get("message") or "参数保存失败")
    with cancel_col:
        if st.button(
            "取消编辑",
            key=f"cancel_pending_{action_id}",
            use_container_width=True,
        ):
            _clear_editor(action_id)
            st.rerun()


def render_pending_actions(pending_actions: list[dict]) -> None:
    """渲染可处理的待确认或执行失败操作。"""
    st.markdown("### ⚠️ 待确认操作")
    if not pending_actions:
        st.success("暂无待确认操作～")
        return

    for action in pending_actions:
        action_id = str(action.get("id", ""))
        status = action.get("status", "pending")
        command = action.get("command", "设备操作")
        device_id = action.get("device_id", "未知设备")
        with st.container():
            if status == "failed":
                st.error(f"**{device_id}** — {command}（上次执行失败，可重试）")
                if action.get("last_error"):
                    st.caption(f"失败原因：{action['last_error']}")
            else:
                st.warning(f"**{device_id}** — {command}")
            st.caption(f"参数：{action.get('params', {})}")
            st.caption(f"确认原因：{action.get('reason', '需要用户确认')}")

            confirm_col, edit_col, reject_col = st.columns(3)
            with confirm_col:
                confirm_label = "🔁 重试执行" if status == "failed" else "✅ 确认执行"
                if st.button(
                    confirm_label,
                    key=f"confirm_{action_id}",
                    use_container_width=True,
                ):
                    result = api(
                        f"/api/actions/{action_id}/confirm", method="post"
                    )
                    invalidate_cache("/api/actions/pending", "/api/actions/log")
                    if result and result.get("success"):
                        _clear_editor(action_id)
                        st.success("设备操作已执行。")
                        st.rerun()
                    elif result:
                        st.error(
                            result.get("message")
                            or result.get("error")
                            or "设备执行失败，请检查设备状态后重试。"
                        )
            with edit_col:
                if st.button(
                    "✏️ 修改参数",
                    key=f"edit_{action_id}",
                    use_container_width=True,
                ):
                    st.session_state["_pending_edit_id"] = action_id
                    st.session_state[f"_pending_params_{action_id}"] = json.dumps(
                        action.get("params") or {}, ensure_ascii=False, indent=2
                    )
                    st.rerun()
            with reject_col:
                if st.button(
                    "❌ 拒绝",
                    key=f"reject_{action_id}",
                    use_container_width=True,
                ):
                    result = api(
                        f"/api/actions/{action_id}/reject", method="post"
                    )
                    if result and result.get("success"):
                        _clear_editor(action_id)
                        invalidate_cache("/api/actions/pending")
                        st.rerun()
                    elif result:
                        st.error(result.get("message") or "拒绝操作失败")

            if st.session_state.get("_pending_edit_id") == action_id:
                _render_parameter_editor(action)

