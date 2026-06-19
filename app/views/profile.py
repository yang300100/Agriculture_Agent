"""Profile page — 纯展示，数据走 API"""

import streamlit as st
from app.api_client import api

def render_profile_page():
    st.markdown("## 基本信息")
    st.markdown("管理您的个人种植档案。")
    with st.form("profile_page_form"):
        c1, c2 = st.columns(2)
        with c1:
            region = st.text_input("所在地区", value=st.session_state.get("user_region", ""))
            soil_options = ["请选择", "壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"]
            cur = st.session_state.get("user_soil_type", "请选择")
            idx = soil_options.index(cur) if cur in soil_options else 0
            soil_type = st.selectbox("土壤类型", soil_options, index=idx)
        with c2:
            farm_size = st.number_input("种植面积（亩）", min_value=0.0, value=st.session_state.get("user_farm_size", 1.0), step=0.5)
            exp_opts = ["请选择", "新手（1年以下）", "初级（1-3年）", "中级（3-5年）", "高级（5-10年）", "专家（10年以上）"]
            cur_e = st.session_state.get("user_experience", "请选择")
            e_idx = exp_opts.index(cur_e) if cur_e in exp_opts else 0
            experience = st.selectbox("种植经验", exp_opts, index=e_idx)
        goals = st.multiselect("种植目标", ["高产", "优质", "省工", "节水", "有机", "多样化种植", "经济效益", "自用为主"], default=st.session_state.get("user_goals", []))
        phone = st.text_input("手机号码", value=st.session_state.get("user_phone", ""), placeholder="如: 13800138000")

        # Agent 自主权级别
        st.divider()
        st.markdown("### 🤖 Agent 自主权")
        autonomy_labels = {
            "low": "🔒 低自主 — 所有操作均需确认后才执行（最安全）",
            "medium": "⚖️ 中等 — 规则边界内自动执行，超出边界需确认（推荐）",
            "high": "🚀 高自主 — 完全自主决策执行，无需确认（仅适合高度信任场景）",
        }
        current_autonomy = st.session_state.get("autonomy_level", "medium")
        autonomy_opts = list(autonomy_labels.keys())
        idx = autonomy_opts.index(current_autonomy) if current_autonomy in autonomy_opts else 1
        autonomy_level = st.radio(
            "自主权级别",
            autonomy_opts,
            index=idx,
            format_func=lambda x: autonomy_labels[x],
            help="控制 Agent 操作硬件时是否需要用户确认。硬限制（如超过最大时长）始终不可突破。"
        )

        if st.form_submit_button("保存修改", type="primary", width='stretch'):
            st.session_state.user_region = region
            st.session_state.user_soil_type = soil_type if soil_type != "请选择" else ""
            st.session_state.user_farm_size = farm_size
            st.session_state.user_experience = experience if experience != "请选择" else ""
            st.session_state.user_goals = goals
            st.session_state.user_phone = phone
            st.session_state.autonomy_level = autonomy_level
            api("/api/profile", "post", {
                "user_region": region, "user_soil_type": soil_type if soil_type != "请选择" else "",
                "user_farm_size": farm_size, "user_experience": experience if experience != "请选择" else "",
                "user_goals": goals, "user_phone": phone,
                "autonomy_level": autonomy_level,
            })
            st.success("信息已更新！")
            st.rerun()
