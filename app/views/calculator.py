"""农资计算器 — 播种量 / 施肥量 / 农药稀释"""

import streamlit as st

# 作物千粒重参考（克）
SEED_DATA = {
    "小麦": {"weight_1000": 40, "germ_rate": 0.90, "seed_per_mu": 15},
    "玉米": {"weight_1000": 300, "germ_rate": 0.92, "seed_per_mu": 2.5},
    "水稻": {"weight_1000": 28, "germ_rate": 0.90, "seed_per_mu": 3},
    "大豆": {"weight_1000": 180, "germ_rate": 0.88, "seed_per_mu": 4},
    "棉花": {"weight_1000": 100, "germ_rate": 0.85, "seed_per_mu": 1.5},
    "花生": {"weight_1000": 500, "germ_rate": 0.90, "seed_per_mu": 10},
    "油菜": {"weight_1000": 3.5, "germ_rate": 0.85, "seed_per_mu": 0.3},
    "谷子": {"weight_1000": 3, "germ_rate": 0.85, "seed_per_mu": 0.5},
    "高粱": {"weight_1000": 28, "germ_rate": 0.88, "seed_per_mu": 1.5},
    "甘薯": {"weight_1000": 0, "germ_rate": 0, "seed_per_mu": 0},
    "土豆": {"weight_1000": 0, "germ_rate": 0, "seed_per_mu": 0},
    "甘蔗": {"weight_1000": 0, "germ_rate": 0, "seed_per_mu": 0},
    "烟草": {"weight_1000": 0.08, "germ_rate": 0.80, "seed_per_mu": 0.0004},
    "茶叶": {"weight_1000": 0, "germ_rate": 0, "seed_per_mu": 0},
    "番茄": {"weight_1000": 3, "germ_rate": 0.85, "seed_per_mu": 0.02},
}

# 化肥养分含量（%）
FERTILIZER_NPK = {
    "尿素": {"N": 46, "P": 0, "K": 0},
    "磷酸二铵": {"N": 18, "P": 46, "K": 0},
    "氯化钾": {"N": 0, "P": 0, "K": 60},
    "硫酸钾": {"N": 0, "P": 0, "K": 50},
    "复合肥(15-15-15)": {"N": 15, "P": 15, "K": 15},
    "过磷酸钙": {"N": 0, "P": 16, "K": 0},
    "硝酸铵": {"N": 34, "P": 0, "K": 0},
    "碳酸氢铵": {"N": 17, "P": 0, "K": 0},
}

# 作物目标产量需肥参考（kg/亩）
CROP_FERTILIZER_NEED = {
    "小麦": {"N": 15, "P": 6, "K": 8, "yield": 500},
    "玉米": {"N": 18, "P": 7, "K": 10, "yield": 600},
    "水稻": {"N": 14, "P": 5, "K": 8, "yield": 500},
    "大豆": {"N": 5, "P": 5, "K": 6, "yield": 200},
    "棉花": {"N": 18, "P": 7, "K": 12, "yield": 300},
    "花生": {"N": 10, "P": 6, "K": 10, "yield": 350},
    "油菜": {"N": 12, "P": 5, "K": 8, "yield": 200},
}


def render_calculator_page():
    st.markdown("## 🧮 农资计算器")

    tab1, tab2, tab3 = st.tabs(["播种量计算", "施肥量计算", "农药稀释"])

    with tab1:
        _render_seed_calc()
    with tab2:
        _render_fertilizer_calc()
    with tab3:
        _render_pesticide_calc()


def _render_seed_calc():
    st.markdown("### 播种量计算")
    st.caption("根据千粒重、发芽率和目标亩株数计算所需种子量")

    crop = st.selectbox("作物", list(SEED_DATA.keys()), key="seed_crop")
    area = st.number_input("种植面积（亩）", min_value=0.1, value=1.0, step=0.5, key="seed_area")

    data = SEED_DATA[crop]
    if data["weight_1000"] == 0:
        st.info(f"{crop}通常使用块茎/扦插繁殖，不适用种子计算。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        weight_1000 = st.number_input("千粒重（克）", value=float(data["weight_1000"]), step=0.1, key="seed_w")
    with col2:
        germ_rate = st.number_input("发芽率", value=data["germ_rate"], min_value=0.5, max_value=1.0, step=0.01, key="seed_g")
    with col3:
        seed_per_mu = st.number_input("目标亩播量（万株/亩）", value=float(data["seed_per_mu"]), step=0.1, key="seed_s")

    if st.button("计算播种量", key="seed_calc_btn"):
        seed_kg_per_mu = (seed_per_mu * 10000 * weight_1000) / (germ_rate * 1000 * 1000)
        total_kg = seed_kg_per_mu * area
        st.success(f"**每亩用种：{seed_kg_per_mu:.2f} kg**  |  **总计：{total_kg:.2f} kg**（{area}亩）")
        st.caption(f"计算依据：千粒重 {weight_1000}g × 亩株数 {seed_per_mu}万株 ÷ （发芽率 {germ_rate} × 1000）")


def _render_fertilizer_calc():
    st.markdown("### 施肥量计算")
    st.caption("根据目标产量和土壤养分，折算具体化肥品种的亩用量")

    crop = st.selectbox("作物", list(CROP_FERTILIZER_NEED.keys()), key="fert_crop")
    area = st.number_input("种植面积（亩）", min_value=0.1, value=1.0, step=0.5, key="fert_area")

    base = CROP_FERTILIZER_NEED[crop]
    st.caption(f"参考：{crop} 目标产量 {base['yield']}kg/亩时，需纯 N:{base['N']} P:{base['P']} K:{base['K']} kg/亩")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_need = st.number_input("需纯N（kg/亩）", value=float(base["N"]), step=0.5, key="fert_n")
        n_fert = st.selectbox("氮肥品种", ["尿素", "硝酸铵", "碳酸氢铵", "磷酸二铵", "复合肥(15-15-15)"], key="fert_n_type")
    with col2:
        p_need = st.number_input("需纯P（kg/亩）", value=float(base["P"]), step=0.5, key="fert_p")
        p_fert = st.selectbox("磷肥品种", ["磷酸二铵", "过磷酸钙", "复合肥(15-15-15)"], key="fert_p_type")
    with col3:
        k_need = st.number_input("需纯K（kg/亩）", value=float(base["K"]), step=0.5, key="fert_k")
        k_fert = st.selectbox("钾肥品种", ["氯化钾", "硫酸钾", "复合肥(15-15-15)"], key="fert_k_type")

    if st.button("计算施肥量", key="fert_calc_btn"):
        lines = ["**每亩施肥量：**\n"]
        total_n_given = 0
        total_p_given = 0
        total_k_given = 0

        for need, fert_name, label in [(n_need, n_fert, "N"), (p_need, p_fert, "P"), (k_need, k_fert, "K")]:
            npk = FERTILIZER_NPK[fert_name]
            pct = npk[label] / 100 if npk[label] > 0 else 0
            if pct > 0:
                amount = need / pct
                lines.append(f"- {fert_name}：**{amount:.1f} kg/亩**（提供纯{label} {need}kg）")
                total_n_given += npk["N"] * amount / 100
                total_p_given += npk["P"] * amount / 100
                total_k_given += npk["K"] * amount / 100
            else:
                lines.append(f"- {fert_name}：此肥料不含{label}，请另选")

        lines.append(f"\n**总养分供给**：N {total_n_given:.1f} / P {total_p_given:.1f} / K {total_k_given:.1f} kg/亩")
        lines.append(f"**{area}亩总用量**：")
        for fert_name in {n_fert, p_fert, k_fert}:
            # already calculated per-mu above
            pass

        st.success("\n".join(lines))


def _render_pesticide_calc():
    st.markdown("### 农药稀释计算")
    st.caption("按使用倍数或亩用量计算兑水量和取药量")

    mode = st.radio("计算方式", ["按稀释倍数", "按亩用量"], horizontal=True, key="pest_mode")

    if mode == "按稀释倍数":
        col1, col2 = st.columns(2)
        with col1:
            ratio = st.number_input("稀释倍数", min_value=100, value=1000, step=100, key="pest_ratio")
        with col2:
            water = st.number_input("用水量（升/亩）", min_value=1.0, value=15.0, step=5.0, key="pest_water")

        if st.button("计算", key="pest_calc_1"):
            pesticide_ml = (water * 1000) / ratio
            st.success(f"每亩取药量：**{pesticide_ml:.1f} ml**（{pesticide_ml / water:.2f} ml/升水）")
            st.caption(f"即 {water}升水中加入 {pesticide_ml:.1f}ml 药剂")
    else:
        col1, col2 = st.columns(2)
        with col1:
            mu_dose = st.number_input("亩用药量（ml或g）", min_value=1.0, value=50.0, step=5.0, key="pest_dose")
        with col2:
            water = st.number_input("用水量（升/亩）", min_value=1.0, value=15.0, step=5.0, key="pest_water2")

        if st.button("计算", key="pest_calc_2"):
            per_liter = mu_dose / water
            st.success(f"每升水加药：**{per_liter:.1f} ml**  |  每亩总量：{mu_dose}ml 兑 {water}升水")
