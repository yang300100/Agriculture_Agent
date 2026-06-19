"""财务管理页 — 数据走 API"""

import streamlit as st, pandas as pd
from datetime import datetime
from app.api_client import api, invalidate_cache

def render_finance_page():
    st.markdown("## 财务管理")
    st.markdown("记录种植成本和销售收入。")

    if st.session_state.get("_finance_lock"):
        st.session_state["_finance_lock"] = False

    # Quick Record Form
    st.markdown("### 快速记账")
    with st.form("finance_quick_form"):
        c1, c2 = st.columns(2)
        with c1:
            record_type = st.selectbox("类型", ["成本支出", "销售收入"])
            crop = st.text_input("作物", value=st.session_state.get("user_crop", ""))
        with c2:
            if record_type == "成本支出":
                amount = st.number_input("金额(元)", min_value=0.0, step=10.0, key="cost_amount")
                cost_type = st.selectbox("成本类型", ["种子", "肥料", "农药", "人工", "农机", "其他"])
                item_name = st.text_input("项目说明")
            else:
                amount = st.number_input("销售总额(元)", min_value=0.0, step=10.0, key="income_amount")
                quantity = st.number_input("销售数量(kg)", min_value=0.0, step=10.0, key="income_qty", help="如不清楚可不填")

        submitted = st.form_submit_button("保存记录", type="primary", width='stretch')
        if submitted and not st.session_state.get("_finance_lock"):
            try:
                st.session_state["_finance_lock"] = True
                if record_type == "成本支出":
                    api("/api/finance/costs", "post", {"crop": crop, "cost_type": cost_type, "item_name": item_name, "unit_price": amount})
                    st.success(f"已记录{cost_type}成本 ¥{amount}")
                else:
                    qty = quantity if quantity > 0 else 1
                    up = amount / qty if qty > 0 else amount
                    api("/api/finance/income", "post", {"crop": crop, "quantity": qty, "unit_price": up})
                    st.success(f"已记录销售收入 ¥{amount}")
                st.rerun()
            except Exception as e:
                st.session_state["_finance_lock"] = False
                st.error(f"保存失败: {e}")

    st.markdown("---")

    # Financial Charts
    st.markdown("### 财务概览")
    report = api("/api/finance/summary")
    if report and report.get("crop_reports"):
        _render_finance_charts(report)
        table_data = []
        for cr in report["crop_reports"]:
            table_data.append({"作物": cr.get("crop",""), "收入 (¥)": f"{cr.get('total_income',0):.2f}", "成本 (¥)": f"{cr.get('total_cost',0):.2f}", "净利润 (¥)": f"{cr.get('net_profit',0):.2f}"})
        df = pd.DataFrame(table_data)
        ti = sum(cr.get("total_income",0) for cr in report["crop_reports"])
        tc = sum(cr.get("total_cost",0) for cr in report["crop_reports"])
        tp = sum(cr.get("net_profit",0) for cr in report["crop_reports"])
        if st.session_state.get("is_mobile"):
            ca, cb = st.columns(2)
            with ca: st.metric("总收入", f"¥{ti:.2f}")
            with cb: st.metric("总成本", f"¥{tc:.2f}")
            st.metric("总净利润", f"¥{tp:.2f}")
        else:
            ca, cb, cc = st.columns(3)
            with ca: st.metric("总收入", f"¥{ti:.2f}")
            with cb: st.metric("总成本", f"¥{tc:.2f}")
            with cc: st.metric("总净利润", f"¥{tp:.2f}")
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("暂无财务记录。")

    st.markdown("---")
    st.markdown("### 数据导出")
    if st.button("导出财务 CSV"):
        data = api("/api/finance/export")
        if data:
            st.download_button("下载 CSV", data["csv"], f"finance_{datetime.now():%Y%m%d}.csv", "text/csv")


def _render_finance_charts(report):
    import plotly.express as px
    import plotly.graph_objects as go
    costs = api("/api/finance/costs") or []
    income = api("/api/finance/income") or []

    monthly = {}
    for c in costs:
        m = c.get("date","")[:7]
        if m:
            monthly.setdefault(m, {"cost":0,"income":0})
            monthly[m]["cost"] += c.get("total_amount",0)
    for i in income:
        m = i.get("date","")[:7]
        if m:
            monthly.setdefault(m, {"cost":0,"income":0})
            monthly[m]["income"] += i.get("total_amount",0)

    if monthly:
        months = sorted(monthly.keys())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=[monthly[m]["cost"] for m in months], mode="lines+markers", name="成本", line=dict(color="#cc785c", width=2)))
        fig.add_trace(go.Scatter(x=months, y=[monthly[m]["income"] for m in months], mode="lines+markers", name="收入", line=dict(color="#5db872", width=2)))
        fig.update_layout(title="月度收支趋势", height=350, margin=dict(l=20,r=20,t=40,b=20), legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig, width='stretch')

    if costs:
        cost_by_type = {}
        for c in costs:
            ct = c.get("cost_type","其他")
            cost_by_type[ct] = cost_by_type.get(ct,0) + c.get("total_amount",0)
        if cost_by_type:
            fig2 = px.pie(names=list(cost_by_type.keys()), values=list(cost_by_type.values()), title="成本构成")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, width='stretch')
