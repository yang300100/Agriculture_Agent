"""种植方案向导 — 数据走 API"""

import os, requests, json, streamlit as st
from app.api_client import api, API_BASE

def render_wizard_page():
    st.markdown("## 🪄 种植方案向导")
    st.caption("三步生成完整种植计划。")

    # 获取作物列表
    crops = {}
    try:
        crops = requests.get(f"{API_BASE}/api/encyclopedia", timeout=10).json()
    except Exception:
        pass
    # fallback to local file list
    if not crops:
        import json
        crops_dir = os.path.join("agriculture_knowledge", "crops")
        crops = {}
        if os.path.exists(crops_dir):
            for f in sorted(os.listdir(crops_dir)):
                if f.endswith(".json"):
                    with open(os.path.join(crops_dir, f), encoding="utf-8") as fh:
                        d = json.load(fh)
                        crops[d["crop_name"]] = d

    names = list(crops.keys())
    if not names:
        st.warning("作物知识库为空")
        return

    region = st.session_state.get("user_region", "")
    soil = st.session_state.get("user_soil_type", "")
    area = st.session_state.get("user_farm_size", 1.0)

    st.markdown("### 第一步：选择作物")
    selected = st.selectbox("选择要种植的作物", names, key="wiz_crop")

    st.markdown("### 第二步：确认信息")
    c1, c2, c3 = st.columns(3)
    with c1: w_region = st.text_input("地区", value=region, key="wiz_region")
    with c2:
        soils = ["壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"]
        si = soils.index(soil) if soil in soils else 0
        w_soil = st.selectbox("土壤类型", soils, index=si, key="wiz_soil")
    with c3: w_area = st.number_input("面积（亩）", min_value=0.1, value=area, step=0.5, key="wiz_area")
    w_goals = st.multiselect("种植目标", ["高产", "优质", "省工", "节水", "有机", "多样化种植", "经济效益"], default=st.session_state.get("user_goals", [])[:3], key="wiz_goals")

    st.markdown("### 第三步：生成方案")
    if st.button("🚀 一键生成完整种植方案", type="primary", width='stretch'):
        if not selected:
            st.warning("请先选择作物")
            return
        with st.spinner("正在生成..."):
            try:
                data = api("/api/plan", "post", {
                    "region": w_region, "soil_type": w_soil,
                    "farm_size": w_area, "goals": w_goals,
                    "experience": st.session_state.get("user_experience", ""),
                    "crop": selected,
                })
                if data:
                    st.success("✅ 方案生成完毕！")
                    st.markdown(data.get("plan_text", ""))
                    st.info(f"已创建：1条进度 + {data.get('task_count',0)}个任务 + {data.get('reminder_count',0)}条提醒")
                else:
                    st.error("后端服务错误")
            except Exception as e:
                st.error(f"请求失败：{e}")
