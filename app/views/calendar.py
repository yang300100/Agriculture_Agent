"""农事日历页 — 数据走 API"""

import streamlit as st
from datetime import datetime, timedelta
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
            "group": p['crop'],
            "task": f"📅 {p.get('stage', '')}（进度）",
            "start": _parse_date(sd) or datetime.now().date(),
            "end": _parse_date(ed) or datetime.now().date(),
            "progress": p.get("progress_percent", 0),
        })
    for t in tasks:
        if t.get("status") == "已完成":
            continue
        ed = t.get("end_date", "")
        if not ed:
            continue
        end_date = _parse_date(ed) or datetime.now().date()
        sd = t.get("start_date", "")
        start_date = _parse_date(sd) if sd else None
        # 没有明确起始日期的任务：取截止日期前3-7天作为展示起始
        if not start_date or start_date == datetime.now().date():
            days_before = 7 if t.get("priority") == "high" else 3
            start_date = max(
                datetime.now().date(),
                end_date - timedelta(days=days_before),
            )
        crop = t.get('crop', '任务')
        items.append({
            "group": crop,
            "task": f"📋 {t.get('title', '')}",
            "start": start_date,
            "end": end_date,
            "progress": t.get("progress", 0),
        })

    # 按作物分组排序：先进度后任务
    items.sort(key=lambda x: (x["group"], 0 if "📅" in x["task"] else 1, x["start"]))

    if items:
        _render_gantt_chart(items)
    else:
        st.info("暂无种植数据。创建种植计划后将显示时间线。")

    # Recent tasks — 只显示待办/进行中/已逾期，按截止日期排序
    st.markdown("### 近期任务")
    active_tasks = [t for t in tasks if t.get("status") not in ("已完成",)]
    active_tasks.sort(key=lambda t: t.get("end_date", "9999") or "9999")
    if active_tasks:
        for t in active_tasks[:15]:
            icon = {"待办": "📝", "进行中": "🌱", "已逾期": "⚠️"}.get(t.get("status", ""), "📋")
            end_str = t.get("end_date", "")
            days_left = ""
            if end_str:
                try:
                    delta = (datetime.strptime(end_str[:10], "%Y-%m-%d") - datetime.now()).days
                    days_left = f"还有{delta}天" if delta > 0 else f"已逾期{abs(delta)}天" if delta < 0 else "今天"
                except Exception:
                    pass
            st.markdown(f"{icon} **{t.get('title','')}** — {t.get('crop','')} | {end_str[:10]} {days_left}")
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
