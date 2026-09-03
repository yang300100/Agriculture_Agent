"""
Common sidebar content — shared across all pages.
Extracted from main.py sidebar (lines 2079-2525).
Excludes: 我的信息, 我的地块, 财务管理 (moved to top-nav pages).
"""

from datetime import datetime, timedelta

import streamlit as st
from app.agent.config import TASK_DEFAULT_DAYS
from app.api_client import api, invalidate_cache


def _list_chat_sessions(limit=20):
    """通过后端获取会话索引。"""
    return api(f"/api/chat/sessions?limit={limit}", cache_ttl=30) or []


def _invalidate_chat_sessions():
    invalidate_cache("/api/chat/sessions")


def _save_chat_session(session_id, messages):
    result = api(
        "/api/chat/sessions", "post",
        {"session_id": session_id, "messages": messages},
    )
    _invalidate_chat_sessions()
    return result


def _render_reminder_notifications(check_interval_minutes=5):
    """经由后端检查提醒，同时保留原有侧栏展示与五分钟节流。"""
    now = datetime.now()
    last_check = st.session_state.get("_reminder_last_check_at")
    if last_check and (now - last_check).total_seconds() < check_interval_minutes * 60:
        return
    st.session_state["_reminder_last_check_at"] = now
    result = api(
        "/api/reminders/check", "post",
        {"phone": st.session_state.get("user_phone", "")},
    ) or {}
    fired = result.get("fired", [])
    upcoming = result.get("upcoming", [])
    if fired:
        st.toast("⏰ 农事提醒已到期！", icon="🌾")
        for reminder in fired:
            st.warning(
                f"🌾 **{reminder.get('crop','')}** · {reminder.get('reminder_type','')}\n\n"
                f"{reminder.get('task_description','')}"
            )
            sms = reminder.get("sms_result")
            if sms:
                st.caption("📱 短信已发送" if sms.get("success") else f"📱 短信发送失败: {sms.get('error','')}")
    if upcoming:
        with st.expander(f"📋 即将到期 ({len(upcoming)})", expanded=False):
            for reminder in upcoming:
                st.markdown(
                    f"- 🌾 **{reminder.get('crop','')}** · {reminder.get('reminder_type','')} · "
                    f"⏰ {reminder.get('next_trigger','')}\n  {reminder.get('task_description','')}"
                )


def render_common_sidebar():
    """Render the shared sidebar with dashboard-style panels."""
    is_mobile = st.session_state.get("is_mobile", False)

    with st.sidebar:

        # ---- Mobile Toggle ----
        st.markdown(
            '<style>.st-cb {margin-bottom: 0 !important;}</style>',
            unsafe_allow_html=True,
        )
        mobile_on = st.toggle("手机模式",
                              value=st.session_state.get("is_mobile", False),
                              key="mobile_toggle", help="切换手机/桌面布局")
        st.session_state.is_mobile = mobile_on

        # 登出
        if st.button("🚪 退出登录", key="sidebar_logout"):
            st.session_state.pop("username", None)
            st.session_state.pop("auth_token", None)
            st.session_state.pop("chat_history", None)
            st.session_state.pop("user_profile_submitted", None)
            st.rerun()

        # ---- Reminder Notifications (throttled check) ----
        _render_reminder_notifications()

        st.markdown("---")

        # ---- SMS Settings ----
        _render_sms_settings()

        st.markdown("---")

        # ---- Session Management ----
        current_sid = st.session_state.get("session_id", "default")
        sessions = _list_chat_sessions(limit=20)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ 新对话", type="secondary", key="sidebar_new_session"):
                # 保存当前
                msgs = st.session_state.get("chat_history", [])
                if msgs:
                    _save_chat_session(current_sid, msgs)
                # 新会话
                st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.chat_history = []
                st.rerun()
        with c2:
            if st.button("🗑️ 清空", type="secondary", key="sidebar_clear_history"):
                st.session_state.chat_history = []
                st.rerun()

        # 历史对话切换
        if sessions:
            opts = {f"{s['title'][:20]} ({s['message_count']}条)": s["id"] for s in sessions}
            labels = list(opts.keys())
            ids = list(opts.values())
            # 当前会话不在列表中（新建/未保存）→ 追加临时选项
            if current_sid not in ids:
                labels.insert(0, "当前对话（未保存）")
                ids.insert(0, current_sid)
            idx = ids.index(current_sid)
            selected = st.selectbox("历史对话", labels, index=idx, key="sidebar_session_select",
                                    label_visibility="collapsed")
            if selected:
                sid = opts.get(selected, current_sid)
                if sid != current_sid:
                    msgs = st.session_state.get("chat_history", [])
                    if msgs:
                        _save_chat_session(current_sid, msgs)
                    loaded = api(f"/api/chat/sessions/{sid}", cache_ttl=0) or {}
                    st.session_state.chat_history = loaded.get("messages", [])
                    st.session_state.session_id = sid
                    st.rerun()

            # 删除按钮
            if st.button("🗑️ 删除当前对话", type="secondary", key="sidebar_delete_session"):
                api(f"/api/chat/sessions/{current_sid}", "delete")
                _invalidate_chat_sessions()
                st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("---")

        if is_mobile:
            # 手机端：显示数量摘要
            _render_mobile_summary()
        else:
            # ---- Planting Progress ----
            _render_planting_progress()

            st.markdown("---")

            # ---- Farm Tasks ----
            _render_task_manager()

        st.markdown("---")

        # ---- Harvest Countdown ----
        _render_harvest_countdown()

        st.markdown("---")

        # ---- Weather ----
        _render_weather_panel()

        st.markdown("---")

        # ---- Lunar Calendar ----
        _render_lunar_calendar()

        if not is_mobile:
            st.markdown("---")
            # ---- Usage Guide ----
            _render_usage_guide()

            # ---- Docs Button ----
            st.markdown("")
            if st.button("📚 文档中心", use_container_width=True, key="sidebar_docs_btn",
                         help="使用手册 · API接口 · 技术手册 · 硬件示例"):
                st.session_state.current_page = "docs"
                st.rerun()


def _render_progress_bar(percent: float, status: str = "进行中") -> None:
    """渲染自定义进度条，颜色随进度和状态变化"""
    pct = max(0, min(100, int(percent)))
    if status == "已完成":
        bar_color = "#27ae60"
        bg_color = "#e8f5e9"
        icon = "✅"
    elif pct >= 75:
        bar_color = "#2ecc71"
        bg_color = "#e8f8f0"
        icon = "📈"
    elif pct >= 40:
        bar_color = "#f39c12"
        bg_color = "#fef9e7"
        icon = "🌱"
    elif pct > 0:
        bar_color = "#e67e22"
        bg_color = "#fdf2e9"
        icon = "🌰"
    else:
        bar_color = "#b0b0b0"
        bg_color = "#f0f0f0"
        icon = "⚪"

    bar_html = f"""
    <div style="
        background: {bg_color};
        border-radius: 8px;
        height: 20px;
        width: 100%;
        position: relative;
        margin: 4px 0 8px 0;
        overflow: hidden;
    ">
        <div style="
            background: linear-gradient(90deg, {bar_color}cc, {bar_color});
            border-radius: 8px;
            height: 100%;
            width: {pct}%;
            transition: width 0.6s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
        ">
            <span style="
                color: #fff;
                font-size: 11px;
                font-weight: 600;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            ">{icon} {pct}%</span>
        </div>
        <span style="
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 10px;
            color: #666;
        ">{status}</span>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def _render_task_progress_bar(percent: float, status: str, priority: str,
                              days_left: int | None) -> None:
    """渲染农事任务进度条，颜色结合优先级和截止日期"""
    pct = max(0, min(100, int(percent)))
    if status == "已完成":
        bar_color = "#27ae60"
        bg_color = "#e8f5e9"
        label = "✅ 已完成"
    elif status == "已逾期":
        bar_color = "#e74c3c"
        bg_color = "#fdecea"
        label = f"⚠️ 已逾期 {abs(days_left)} 天" if days_left is not None else "⚠️ 已逾期"
    elif days_left is not None and days_left == 0:
        bar_color = "#e74c3c"
        bg_color = "#fdecea"
        label = "🔥 今天截止"
    elif days_left is not None and days_left <= 3:
        bar_color = "#f39c12"
        bg_color = "#fef9e7"
        label = f"⏰ 剩{days_left}天"
    elif priority == "high":
        bar_color = "#e67e22"
        bg_color = "#fef5ec"
        label = "🔴 高优先"
    elif priority == "medium":
        bar_color = "#3498db"
        bg_color = "#eaf2f8"
        label = "🟡 中优先"
    else:
        bar_color = "#95a5a6"
        bg_color = "#f0f3f4"
        label = "🟢 低优先"

    bar_html = f"""
    <div style="
        background: {bg_color};
        border-radius: 8px;
        height: 20px;
        width: 100%;
        position: relative;
        margin: 4px 0 8px 0;
        overflow: hidden;
    ">
        <div style="
            background: linear-gradient(90deg, {bar_color}cc, {bar_color});
            border-radius: 8px;
            height: 100%;
            width: {pct}%;
            transition: width 0.6s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
        ">
            <span style="
                color: #fff;
                font-size: 11px;
                font-weight: 600;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            ">{pct}%</span>
        </div>
        <span style="
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 10px;
            color: #333;
        ">{label}</span>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def _render_planting_progress():
    """Planting progress cards with custom progress bars and action buttons."""
    st.header("种植进度")
    try:
        progress_cards = api("/api/progress", cache_ttl=30) or []

        if progress_cards:
            for card in progress_cards:
                with st.container():
                    title_cols = st.columns([3, 1])
                    with title_cols[0]:
                        status_color = {"进行中": "🟢", "已完成": "✅", "待开始": "⚪"}
                        status_icon = status_color.get(card.get('status', ''), "🟡")
                        st.markdown(f"**{status_icon} {card['crop']}** — {card.get('stage','')}")
                    with title_cols[1]:
                        if st.button("🗑️", key=f"del_prog_{card['id']}", help="删除此进度"):
                            api(f"/api/progress/{card['id']}", "delete")
                            invalidate_cache("/api/progress", "/api/dashboard")
                            st.rerun()

                    _render_progress_bar(card.get('progress', 0), card.get('status', ''))

                    st.caption(f"阶段 {card.get('stage_number',0)}/{card.get('total_stages',0)}")

                    if card.get('status') != "已完成":
                        if st.button("▶ 完成阶段", key=f"complete_prog_{card['id']}",
                                     width='stretch'):
                            result = api(f"/api/progress/{card['id']}/advance", "post")
                            if result and result.get("success"):
                                invalidate_cache("/api/progress", "/api/dashboard")
                                st.rerun()

                    st.markdown("---")
        else:
            st.info("暂无种植进度记录")

        if st.button("+ 添加种植进度", key="sidebar_add_progress"):
            st.session_state.show_add_progress = True

        if st.session_state.get("show_add_progress", False):
            with st.container():
                st.markdown("**添加新种植进度**")
                new_crop = st.text_input("作物名称", key="sidebar_new_crop_name")
                new_stage = st.text_input("当前阶段", key="sidebar_new_stage_name")
                total_stages = st.number_input("总阶段数", min_value=1, max_value=20, value=5, key="sidebar_new_total_stages")
                cols = st.columns(2)
                with cols[0]:
                    if st.button("保存进度", key="sidebar_save_progress"):
                        if new_crop and new_stage:
                            r = api("/api/progress", "post", {
                                "crop": new_crop, "stage": new_stage,
                                "stage_number": 1, "total_stages": total_stages,
                                "start_date": datetime.now().strftime("%Y-%m-%d"),
                                "status": "进行中",
                            })
                            if r:
                                invalidate_cache("/api/progress", "/api/dashboard")
                                st.success(f"已添加 {new_crop}")
                                st.session_state.show_add_progress = False
                                st.rerun()
                        else:
                            st.warning("请填写作物名称和当前阶段")
                with cols[1]:
                    if st.button("取消", key="sidebar_cancel_progress"):
                        st.session_state.show_add_progress = False
                        st.rerun()
    except Exception as e:
        st.error(f"加载进度失败: {e}")


def _render_task_manager():
    """Farm task cards with custom progress bars, priority, and deadlines."""
    st.header("农事任务")
    try:
        task_cards = api("/api/tasks", cache_ttl=30) or []

        if task_cards:
            for card in task_cards:
                with st.container():
                    title_cols = st.columns([3, 1])
                    with title_cols[0]:
                        status_emoji = {"待办": "📝", "进行中": "🌱", "已完成": "✅", "已逾期": "⚠️"}
                        status_icon = status_emoji.get(card['status'], "📋")
                        priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                        priority_icon = priority_color.get(card['priority'], "⚪")
                        st.markdown(f"**{status_icon} {card['title']}** {priority_icon}")
                    with title_cols[1]:
                        if st.button("🗑️", key=f"del_task_{card['id']}", help="删除此任务"):
                            api(f"/api/tasks/{card['id']}", "delete")
                            invalidate_cache("/api/tasks", "/api/dashboard")
                            st.rerun()

                    desc = card.get('description', '')[:40] if card.get('description') else ""
                    st.caption(f"🌾 {card['crop']} | {desc}")

                    _render_task_progress_bar(
                        card['progress'], card['status'], card['priority'],
                        card.get('days_left'),
                    )

                    if card['status'] not in ("已完成", "已逾期"):
                        has_device = bool(card.get('device_id') and card.get('device_command'))
                        if has_device:
                            c1, c2 = st.columns([1, 1])
                        else:
                            c1 = st.container()
                            c2 = None

                        with c1:
                            if st.button("✅ 标记完成", key=f"complete_{card['id']}"):
                                api(f"/api/tasks/{card['id']}/complete", "post")
                                invalidate_cache("/api/tasks", "/api/dashboard")
                                st.rerun()

                        if has_device and c2:
                            with c2:
                                btn_label = "🔄 重试" if card['status'] == "进行中" else "⚡ 执行"
                                if st.button(btn_label, key=f"exec_{card['id']}", type="primary"):
                                    with st.spinner(f"正在执行: {card['title']}..."):
                                        result = api(f"/api/tasks/{card['id']}/execute", "post")
                                    if result and result.get("success"):
                                        st.success(f"✅ {card['title']} 执行成功！")
                                        invalidate_cache("/api/tasks", "/api/dashboard")
                                        st.rerun()
                                    else:
                                        error_msg = (result or {}).get("error", "未知错误")
                                        st.error(f"❌ 执行失败: {error_msg}")
                                        st.rerun()

                    st.markdown("---")
        else:
            st.info("暂无农事任务")

        # Add task
        if st.button("+ 添加任务", key="sidebar_add_task"):
            st.session_state.show_add_task = True

        if st.session_state.get("show_add_task", False):
            with st.container():
                st.markdown("**添加新农事任务**")
                task_crop = st.text_input("作物", key="sidebar_task_crop")
                task_title = st.text_input("任务标题", key="sidebar_task_title")
                task_type = st.selectbox(
                    "任务类型",
                    ["浇水", "施肥", "除草", "病虫害防治", "修剪", "播种", "收获", "其他"],
                    key="sidebar_task_type"
                )
                task_priority = st.selectbox(
                    "优先级", ["high", "medium", "low"],
                    format_func=lambda x: {"high": "高", "medium": "中", "low": "低"}[x],
                    key="sidebar_task_priority"
                )

                cols = st.columns(2)
                with cols[0]:
                    if st.button("保存任务", key="sidebar_save_task"):
                        if task_crop and task_title:
                            try:
                                end_date = (datetime.now() + timedelta(days=TASK_DEFAULT_DAYS)).strftime("%Y-%m-%d")
                                api("/api/tasks", "post", json_data={
                                    "crop": task_crop,
                                    "task_type": task_type,
                                    "title": task_title,
                                    "description": f"{task_type}任务",
                                    "status": "待办",
                                    "priority": task_priority,
                                    "end_date": end_date,
                                    "progress_percent": 0
                                })
                                invalidate_cache("/api/tasks", "/api/dashboard")
                                st.success(f"已添加任务: {task_title}")
                                st.session_state.show_add_task = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"添加失败: {e}")
                        else:
                            st.warning("请填写作物和任务标题")
                with cols[1]:
                    if st.button("取消", key="sidebar_cancel_task"):
                        st.session_state.show_add_task = False
                        st.rerun()

    except Exception as e:
        st.error(f"加载任务失败: {e}")


def _render_sms_settings():
    """短信通知设置"""
    with st.expander("📱 短信通知设置", expanded=False):
        phone = st.text_input(
            "手机号码", key="sidebar_phone",
            value=st.session_state.get("user_phone", ""),
            placeholder="如: 13800138000"
        )
        if phone != st.session_state.get("user_phone", ""):
            st.session_state.user_phone = phone
            # Sync to agent_state
            if st.session_state.get("agent_state"):
                st.session_state.agent_state.user_profile["phone"] = phone

        if st.button("测试短信发送", key="test_sms"):
            if not phone:
                st.warning("请先输入手机号码")
            else:
                result = api("/api/sms/test", "post", {"phone": phone}) or {}
                if result.get("success"):
                    st.success("测试短信发送成功！")
                else:
                    st.error(f"发送失败: {result.get('error', '未知错误')}")


def _render_weather_panel():
    """Weather query and forecast for user's region (cached in session_state)."""
    st.header("天气服务")
    try:
        location = st.session_state.get("user_region", "北京")

        # session_state 缓存：30分钟内不重复请求
        cache_key = f"_weather_cache_{location}"
        cached = st.session_state.get(cache_key)
        weather_cache_ttl = 1800  # 30分钟
        if cached and (datetime.now() - cached["ts"]).total_seconds() < weather_cache_ttl:
            current = cached.get("current")
            alerts = cached.get("alerts")
            forecast = cached.get("forecast")
        else:
            current = None
            alerts = None
            forecast = None

        if st.button("查询天气", key="sidebar_query_weather"):
            with st.spinner("正在获取天气信息..."):
                wdata = api(f"/api/weather/{location}", cache_ttl=1800) or {}
                current = wdata.get("current")
                forecast = wdata.get("forecast", [])
                alerts = wdata.get("alerts")
                st.session_state[cache_key] = {
                    "ts": datetime.now(), "current": current, "forecast": forecast, "alerts": alerts
                }

        if current:
            st.markdown(f"**{location} 当前天气**")
            st.markdown(f"🌡️ {current.get('temperature','-')}℃ ({current.get('temperature_low','-')}℃~{current.get('temperature_high','-')}℃)")
            st.markdown(f"☁️ {current.get('weather_desc','-')}")
            st.markdown(f"💧 湿度: {current.get('humidity','-')}%")

        if alerts:
            st.warning("有气象预警，请注意防护！")

        if forecast:
            st.markdown("**未来3天预报**")
            for w in forecast:
                st.caption(f"{w.get('date','')}: {w.get('weather_desc','')} {w.get('temperature_low','-')}~{w.get('temperature_high','-')}℃")
        elif not current:
            st.caption("点击上方按钮查询天气")

    except Exception:
        st.info("天气服务暂未配置")


def _render_harvest_countdown():
    """收获倒计时面板"""
    st.header("收获倒计时")
    try:
        from core.weather_alerts import calculate_harvest_countdown, format_harvest_countdown
        progresses = api("/api/progress", cache_ttl=30) or []
        if progresses:
            countdowns = calculate_harvest_countdown(
                [{"crop": p.get("crop",""), "stage": p.get("stage",""),
                  "start_date": p.get("start_date",""), "progress_percent": p.get("progress",0)}
                 for p in progresses]
            )
            if countdowns:
                text = format_harvest_countdown(countdowns)
                if text:
                    st.markdown(text)
                else:
                    st.caption("暂无收获倒计时")
            else:
                st.caption("暂无种植进度")
        else:
            st.caption("创建种植计划后将显示收获倒计时")
    except Exception:
        st.caption("倒计时暂不可用")


def _render_mobile_summary():
    """手机端侧边栏摘要：进度 + 任务数量统计"""
    try:
        progresses = api("/api/progress", cache_ttl=30) or []
        tasks = api("/api/tasks", cache_ttl=30) or []

        p_active = len([p for p in progresses if p.get("status") == "进行中"])
        p_done = len([p for p in progresses if p.get("status") == "已完成"])
        t_pending = len([t for t in tasks if t.get("status") in ("待办", "进行中")])
        t_overdue = len([t for t in tasks if t.get("status") == "已逾期"])

        cols = st.columns(2)
        with cols[0]:
            st.metric("种植进度", f"{p_active} 进行中", f"{p_done} 已完成")
        with cols[1]:
            st.metric("农事任务", f"{t_pending} 待办", f"{t_overdue} 逾期" if t_overdue else None)
    except Exception:
        st.caption("进度数据加载中...")


def _render_lunar_calendar():
    """农历和节气信息面板"""
    with st.expander("农历节气", expanded=False):
        try:
            from core.lunar_calendar import get_lunar_today
            info = get_lunar_today()
            if info["lunar_month"]:
                st.caption(f"农历 {info['lunar_month']}{info['lunar_day']}")
            if info["solar_term_current"]:
                st.markdown(f"**当前节气：{info['solar_term_current']}**")
            if info["solar_term_next"] and info["solar_term_next"] != info["solar_term_current"]:
                st.caption(f"下个节气：{info['solar_term_next']}")
            if info["solar_term_advice"]:
                st.caption(info["solar_term_advice"])
        except Exception:
            st.caption("农历信息暂不可用")


def _render_usage_guide():
    """Static usage tips."""
    st.markdown("### 使用说明：")
    st.markdown('1. **作物选择** - 询问"华北地区适合种什么？"')
    st.markdown('2. **种植时间** - 询问"小麦什么时候播种？"')
    st.markdown('3. **农事提醒** - 说"为玉米设置浇水提醒"')
    st.markdown('4. **病虫害防治** - 描述症状或上传图片获取建议')
    st.markdown('5. **进度跟踪** - 询问"我的番茄现在该做什么？"')
    st.markdown('6. **天气查询** - 询问"明天适合喷药吗？"')
    st.markdown('7. **图片诊断** - 点击上方"上传农作物图片"进行分析')
