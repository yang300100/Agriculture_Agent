"""农事日历页 — 数据走 API"""

import streamlit as st
from datetime import datetime
from app.api_client import api

def render_calendar_page():
    st.markdown("## 农事日历")
    progresses = api("/api/progress") or []
    tasks = api("/api/tasks") or []

    items = []
    for p in progresses:
        sd = p.get("start_date", "")
        ed = p.get("expected_end_date", "") or sd
        items.append({
            "group": f"🌾 {p['crop']}",
            "task": p.get("stage", ""),
            "start": _parse_date(sd) or datetime.now().date(),
            "end": _parse_date(ed) or datetime.now().date(),
            "progress": p.get("progress_percent", 0),
        })
    for t in tasks:
        # 只显示未完成且有截止日期的任务，避免历史任务堆满日历
        if t.get("status") == "已完成":
            continue
        sd = t.get("start_date", "") or t.get("end_date", "")
        ed = t.get("end_date", "")
        if not ed:
            continue  # 无截止日期的任务不显示在时间线上
        items.append({
            "group": f"📋 {t.get('crop','任务')}",
            "task": t.get("title", ""),
            "start": _parse_date(sd) or datetime.now().date(),
            "end": _parse_date(ed) or datetime.now().date(),
            "progress": t.get("progress", 0),
        })

    if items:
        _render_gantt_chart(items)
    else:
        st.info("暂无种植数据。创建种植计划后将显示时间线。")

    # Recent tasks
    st.markdown("### 近期任务")
    if tasks:
        for t in tasks[:10]:
            icon = {"待办": "📝", "进行中": "🌱", "已完成": "✅", "已逾期": "⚠️"}.get(t.get("status", ""), "📋")
            end_str = t.get("end_date", "")
            days_left = ""
            if end_str:
                try:
                    delta = (datetime.strptime(end_str, "%Y-%m-%d") - datetime.now()).days
                    days_left = f"还有{delta}天" if delta > 0 else f"已逾期{abs(delta)}天" if delta < 0 else "今天"
                except Exception:
                    pass
            st.markdown(f"{icon} **{t.get('title','')}** — {t.get('crop','')} | {end_str} {days_left}")
            st.progress(t.get("progress", 0) / 100)
    else:
        st.caption("暂无待办任务")


def _render_gantt_chart(items):
    import plotly.figure_factory as ff
    import plotly.express as px
    df_data = []
    groups = sorted(set(i["group"] for i in items))
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    for item in items:
        df_data.append(dict(Task=item["group"], Start=item["start"].strftime("%Y-%m-%d"), Finish=item["end"].strftime("%Y-%m-%d"), Description=item["task"], Complete=item["progress"]))
    fig = ff.create_gantt(df_data, colors=colors, index_col="Task", show_colorbar=True, showgrid_x=True, showgrid_y=True, title="种植计划时间线")
    is_mobile = st.session_state.get("is_mobile", False)
    fig.update_layout(height=max(200, len(items) * 35) if is_mobile else max(300, len(items) * 50), margin=dict(l=20, r=20, t=40, b=20), xaxis_title="日期", xaxis_tickangle=-45 if is_mobile else 0)
    st.plotly_chart(fig, width='stretch')


def _parse_date(s):
    if not s: return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except Exception: continue
    return None
