"""仪表盘概览页 — 纯展示，数据从 API 获取"""

import os, json, streamlit as st
from app.api_client import api


def render_dashboard_page():
    st.markdown("## 📊 农场概览")
    data = api("/api/dashboard", cache_ttl=30)
    if not data:
        st.warning("后端服务未连接。请启动: python app/api_server.py")
        return

    is_mobile = st.session_state.get("is_mobile", False)

    # Row 1: 进度 + 任务
    ca, cb = st.columns(2) if not is_mobile else (st.container(), st.container())
    with ca:
        _render_progress(data.get("progress", []))
    with cb:
        _render_tasks(data.get("tasks", {}))

    st.markdown("---")

    # Row 2: 财务 + 节气
    cc, cd = st.columns(2) if not is_mobile else (st.container(), st.container())
    with cc:
        _render_finance(data.get("finance", {}))
    with cd:
        _render_seasonal(data.get("lunar", {}))

    st.markdown("---")

    # Row 3: 预警
    alerts = data.get("weather_alerts")
    if alerts and alerts.get("has_alert"):
        st.warning(f"⚠️ {alerts.get('region','')}：{alerts.get('count',0)} 条气象预警")
        for a in alerts.get("alerts", [])[:3]:
            st.caption(f"- {a.get('type','')}（{a.get('level','')}）")

    # 持续异常天气
    import json
    persist_path = os.path.join("data", "weather_persistence.json")
    if os.path.exists(persist_path):
        with open(persist_path, encoding="utf-8") as f:
            pdata = json.load(f)
            for a in pdata.get("alerts", [])[:2]:
                st.warning(f"🌧 **{a['type']}** 已持续 {a['days']} 天（{a.get('period','')}）\n\n{a.get('advice','')[:200]}")

    # 病害风险
    risk_path = os.path.join("data", "disease_risks.json")
    if os.path.exists(risk_path):
        with open(risk_path, encoding="utf-8") as f:
            risks = json.load(f).get("risks", [])
        if risks:
            st.markdown("---")
            st.warning("🦠 病虫害风险提示")
            for r in risks[:5]:
                st.caption(f"- {r['crop']}：{r['disease']} — {r.get('advice','')}")


def _render_progress(progresses):
    st.markdown("### 🌱 种植进度")
    if progresses:
        from app.ui.sidebar import _render_progress_bar
        for p in progresses[:4]:
            icon = {"进行中": "🌱", "已完成": "✅", "待开始": "⚪"}.get(p.get("status", ""), "")
            st.markdown(f"{icon} **{p['crop']}** — {p['stage']} ({p.get('stage_number',0)}/{p.get('total_stages',0)})")
            _render_progress_bar(p.get("progress_percent", 0), p.get("status", ""))
    else:
        st.info("暂无种植进度。")


def _render_tasks(tasks):
    st.markdown("### 📋 待办任务")
    active = tasks.get("active", [])
    overdue = tasks.get("overdue", [])
    if overdue:
        for t in overdue[:3]:
            st.warning(f"⚠️ **{t['title']}**（{t['crop']}）— 已逾期")
    if active:
        for t in active[:5]:
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority", ""), "")
            st.markdown(f"- {icon} **{t['title']}**（{t['crop']}）")
    elif not overdue:
        st.info("暂无待办任务。")


def _render_finance(fin):
    st.markdown("### 💰 本月财务")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("收入", f"¥{fin.get('month_income', 0):.0f}")
    with c2: st.metric("成本", f"¥{fin.get('month_cost', 0):.0f}")
    with c3:
        p = fin.get("profit", 0)
        st.metric("利润", f"¥{p:.0f}", delta_color="normal" if p >= 0 else "inverse")


def _render_seasonal(lunar):
    st.markdown("### 📅 节气农事")
    if lunar.get("lunar_month"):
        st.caption(f"农历 {lunar['lunar_month']}{lunar['lunar_day']}")
    if lunar.get("solar_term_current"):
        st.markdown(f"**当前节气：{lunar['solar_term_current']}**")
    if lunar.get("solar_term_advice"):
        st.info(lunar["solar_term_advice"])
