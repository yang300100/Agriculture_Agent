"""政策补贴查询页 — 数据走 API"""
import streamlit as st
from app.api_client import api


def render_policy_page():
    st.markdown("## 政策补贴查询")
    q = st.text_input("搜索补贴政策", placeholder="例如：小麦补贴、耕地补贴...")
    if q:
        results = api(f"/api/policy/search?q={q}") or []
        if results:
            for r in results:
                with st.container():
                    st.markdown(f"**{r.get('metadata', {}).get('source', '政策文档')}**")
                    st.caption(r.get("content", "")[:400])
        else:
            st.info("未找到相关政策。")

    st.subheader("常见补贴类型速查")
    subsidies = [
        ("耕地地力保护补贴", "对耕地承包权农民补贴，一般每亩50-100元。"),
        ("农机购置补贴", "购买纳入目录的农机具，享30%左右补贴。"),
        ("农业保险保费补贴", "中央和地方财政对保费给予50%-80%补贴。"),
        ("最低收购价政策", "小麦、水稻实行最低收购价保护。"),
        ("大豆玉米复合种植补贴", "带状复合种植每亩150-200元。"),
        ("棉花目标价格补贴", "新疆等主产区市场价低于目标时给予差价补贴。"),
        ("生产者补贴", "东北玉米、大豆、水稻生产者可获补贴。"),
        ("良种补贴", "使用优良品种的农民可获补贴。"),
    ]
    for title, desc in subsidies:
        with st.expander(title):
            st.write(desc)

    st.caption("政策信息仅供参考，具体以当地政府文件为准。")
