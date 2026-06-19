"""作物百科 — 数据走 API"""

import streamlit as st
from app.api_client import api

def render_encyclopedia_page():
    st.markdown("## 📖 作物百科")
    all_crops = api("/api/encyclopedia") or {}
    if not all_crops:
        st.warning("作物知识库为空")
        return

    names = list(all_crops.keys())
    ca, cb = st.columns([1, 2])
    with ca:
        search = st.text_input("搜索作物", key="ency_search")
        filtered = [c for c in names if search in c] if search else names
        selected = st.selectbox("选择作物", filtered, key="ency_select")
    if selected and selected in all_crops:
        with cb:
            _render_detail(all_crops[selected])

    st.markdown("---")
    st.markdown("### 双作物对比")
    cc1, cc2 = st.columns(2)
    with cc1: crop_a = st.selectbox("作物 A", names, key="cmp_a", index=0)
    with cc2: crop_b = st.selectbox("作物 B", names, key="cmp_b", index=min(1, len(names)-1))
    if crop_a and crop_b and crop_a != crop_b:
        _render_comparison(all_crops[crop_a], all_crops[crop_b])


def _render_detail(data):
    name = data["crop_name"]
    aliases = "、".join(data.get("aliases", []))
    regions = "、".join(data.get("suitable_regions", []))
    st.markdown(f"### {name}")
    st.caption(f"别名：{aliases}  |  适宜地区：{regions}")
    t1, t2, t3, t4, t5 = st.tabs(["生长阶段", "施肥灌溉", "病虫害", "产量市场", "种植季节"])
    with t1:
        for s in data.get("growth_stages", []):
            st.markdown(f"**{s.get('stage','')}**（约{s.get('duration_days','?')}天）")
            tasks = s.get("key_tasks", [])
            if tasks: st.caption("关键农事：" + "、".join(tasks))
            if s.get("notes"): st.caption(s["notes"])
            st.markdown("---")
    with t2:
        st.markdown("**施肥指导**")
        for f in data.get("fertilization_guide", []):
            st.markdown(f"- **{f.get('time','')}**：{f.get('type','')}，{f.get('amount','')}（{f.get('method','')}）")
        st.markdown("**灌溉指导**")
        for ir in data.get("irrigation_guide", []):
            st.markdown(f"- **{ir.get('stage','')}**：{ir.get('purpose','')}，{ir.get('amount','')}")
    with t3:
        st.markdown("**常见病害**")
        for d in data.get("common_diseases", []):
            with st.expander(f"🦠 {d.get('name','')}"):
                st.markdown(f"症状：{d.get('symptoms','')}  |  防治：{d.get('prevention','')}")
                st.caption(f"发生期：{d.get('occurrence_stage','')}")
        st.markdown("**常见虫害**")
        for p in data.get("common_pests", []):
            with st.expander(f"🐛 {p.get('name','')}"):
                st.markdown(f"危害：{p.get('symptoms','')}  |  防治：{p.get('control','')}")
    with t4:
        yi = data.get("yield_info", {})
        if yi:
            st.markdown(f"低产：{yi.get('low_yield','-')}  |  中产：{yi.get('medium_yield','-')}  |  高产：{yi.get('high_yield','-')}")
        mi = data.get("market_info", {})
        if mi:
            st.markdown(f"上市旺季：{mi.get('peak_season','-')}")
            st.caption(f"储存提示：{mi.get('storage_tips','-')}")
    with t5:
        for k, info in data.get("planting_seasons", {}).items():
            st.markdown(f"**{info.get('name',k)}**：播种 {info.get('sowing_time','')} → 收获 {info.get('harvest_time','')}")
            st.caption(info.get('notes',''))


def _render_comparison(a, b):
    st.markdown(f"### {a['crop_name']} vs {b['crop_name']}")
    import pandas as pd
    rows = [
        {"对比项": "生育期", a["crop_name"]: f"{sum(s.get('duration_days',0) for s in a.get('growth_stages',[]))}天", b["crop_name"]: f"{sum(s.get('duration_days',0) for s in b.get('growth_stages',[]))}天"},
        {"对比项": "病害数", a["crop_name"]: f"{len(a.get('common_diseases',[]))}种", b["crop_name"]: f"{len(b.get('common_diseases',[]))}种"},
        {"对比项": "中等产量", a["crop_name"]: a.get("yield_info",{}).get("medium_yield","-"), b["crop_name"]: b.get("yield_info",{}).get("medium_yield","-")},
    ]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
