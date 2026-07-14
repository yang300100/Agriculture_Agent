"""
Field Management page — my fields (地块管理).
Shows field overview map with temperature & precipitation weather layers,
plus field drawing for adding new fields.
"""

import streamlit as st


def _map_size():
    """返回地图尺寸，手机端缩小"""
    if st.session_state.get("is_mobile", False):
        return {"width": 1100, "height": 350}
    return {"width": 1100, "height": 600}

import os, streamlit as st
from app.api_client import api


def render_fields_page():
    """Render the full field management page."""
    st.markdown("## 地块管理")
    st.markdown("管理您的农田地块，在地图上绘制边界，查看气温和降水天气图层。")

    if st.session_state.get("_field_save_lock"):
        st.session_state["_field_save_lock"] = False

    try:
        # 保留 MapManager 用于地图渲染（folium 本地操作）
        from core.map_manager import MapManager, create_folium_map, extract_polygon_from_map_data
        map_manager = MapManager()
        fields = map_manager.get_all_fields()

        # ---- Existing Fields Overview Map ----
        if fields:
            _render_field_overview_map(fields, map_manager)

        # ---- Field Cards ----
        if fields:
            st.markdown("### 已有地块")
            n_cols = 1 if st.session_state.get("is_mobile", False) else min(len(fields), 4)
            cols = st.columns(n_cols)
            for i, field in enumerate(fields):
                with cols[i % n_cols]:
                    st.markdown(
                        f'<div style="background:#efe9de;border:1px solid #e6dfd8;'
                        f'border-radius:12px;padding:12px;margin-bottom:4px">'
                        f'<strong style="font-size:16px;color:#141413">'
                        f'{"🌾" if field.current_crop else "📍"} {field.name}</strong><br>'
                        f'<span style="color:#3d3d3a">{field.area_mu:.2f}亩</span><br>'
                        f'<span style="color:#6c6a64">土壤: {field.soil_type or "未设置"}</span><br>'
                        f'{"<span style=color:#6c6a64>作物: " + field.current_crop + "</span>" if field.current_crop else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    if st.button("删除", key=f"field_del_{field.id}"):
                        api(f"/api/fields/{field.id}", "delete")
                        st.rerun()

            total_area = map_manager.get_total_area()
            st.caption(f"总计: {len(fields)}个地块, 共{total_area:.2f}亩")
        else:
            st.info("暂无地块记录，点击下方按钮添加。")

        # ---- Field Comparison ----
        if len(fields) >= 2:
            _render_field_comparison(fields, map_manager)

        # ---- Add New Field Button ----
        if not st.session_state.get("show_add_field", False):
            if st.button("添加新地块", type="primary", key="field_add_btn"):
                st.session_state.show_add_field = True
                st.rerun()

        # ---- Map Drawing Interface ----
        if st.session_state.get("show_add_field", False):
            _render_drawing_interface(fields, map_manager)

    except Exception as e:
        st.error(f"加载地块失败: {e}")


def _render_field_overview_map(fields, map_manager):
    """Render an overview map with GPS locate, field jump, and weather overlay layers."""
    st.markdown("### 地块总览与天气地图")

    # ---- Compute default map center ----
    valid_centers = [f for f in fields if f.center_lat and f.center_lon]
    if valid_centers:
        default_lat = sum(f.center_lat for f in valid_centers) / len(valid_centers)
        default_lon = sum(f.center_lon for f in valid_centers) / len(valid_centers)
    else:
        default_lat, default_lon = 39.9, 116.4

    # ---- Feature 2: Field jump selector ----
    field_options = ["全部地块"] + [f"{f.name}" for f in fields]
    selected = st.selectbox("跳转到地块", field_options, key="field_jump_select")

    if selected != "全部地块":
        target = next((f for f in fields if f.name == selected), None)
        if target and target.center_lat and target.center_lon:
            center_lat, center_lon = target.center_lat, target.center_lon
            zoom = 16
        else:
            center_lat, center_lon = default_lat, default_lon
            zoom = 13
    else:
        center_lat, center_lon = default_lat, default_lon
        zoom = 13

    # ---- Fetch weather for each field center (non-blocking) ----
    field_weather = {}
    try:
        from core.weather_service import WeatherService
        weather_service = WeatherService()
        with st.spinner("正在获取各地块实时天气..."):
            for field in fields:
                if field.center_lon and field.center_lat:
                    wdata = weather_service.get_grid_weather(
                        field.center_lon, field.center_lat
                    )
                    if wdata:
                        field_weather[field.id] = wdata
    except Exception:
        pass  # 天气获取失败不影响地图显示

    # ---- Build the map ----
    try:
        import folium
        from folium.plugins import LocateControl
        from streamlit_folium import st_folium

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # GPS Locate button
        LocateControl(
            auto_start=False,
            strings={"title": "定位到当前位置", "popup": "您在这里"},
            position="topleft",
        ).add_to(m)

        # Satellite basemap
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            name="卫星图",
            overlay=False,
        ).add_to(m)

        # ---- 地块边界与天气信息 ----
        field_group = folium.FeatureGroup(name="地块边界", show=True).add_to(m)
        for field in fields:
            if not (field.coordinates and len(field.coordinates) >= 3):
                continue

            coords_latlng = [[c[1], c[0]] for c in field.coordinates]
            w = field_weather.get(field.id)

            if w:
                popup_html = (
                    f"<div style='font-family:Inter,sans-serif;min-width:180px'>"
                    f"<b style='font-size:15px;color:#141413'>{field.name}</b><br>"
                    f"<span style='color:#6c6a64'>面积: {field.area_mu:.2f}亩"
                )
                if field.current_crop:
                    popup_html += f" | 作物: {field.current_crop}"
                popup_html += "</span><hr style='border-color:#e6dfd8;margin:6px 0'>"
                popup_html += (
                    f"<span style='font-size:26px;font-weight:600;color:#141413'>"
                    f"{w['temp']:.0f}°C</span> "
                    f"<span style='color:#6c6a64'>{w['text']}</span><br>"
                    f"💧 湿度: {w['humidity']}% &nbsp; 🌧 降水: {w['precip']:.1f}mm<br>"
                    f"💨 {w['windDir']} {w['windSpeed']}km/h (级{w['windScale']})<br>"
                    f"☁️ 云量: {w.get('cloud', '-')}% &nbsp;"
                    f"📊 气压: {w['pressure']:.0f}hPa<br>"
                    f"<span style='color:#8e8b82;font-size:11px'>"
                    f"观测: {w.get('obsTime', '')[:19]}</span></div>"
                )
            else:
                popup_html = (
                    f"<b>{field.name}</b><br>面积: {field.area_mu:.2f}亩<br>"
                    f"<span style='color:#c64545'>天气数据获取失败</span>"
                )

            folium.Polygon(
                locations=coords_latlng,
                color="#cc785c",
                weight=2.5,
                fill=True,
                fill_color="#cc785c",
                fill_opacity=0.2,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{field.name} ({w['temp']:.0f}°C)" if w else field.name,
            ).add_to(field_group)

            # 地块中心温度徽标
            if field.center_lat and field.center_lon and w:
                temp = w['temp']
                if temp >= 35:
                    badge_color = "#c64545"
                elif temp >= 25:
                    badge_color = "#e8a55a"
                elif temp <= 0:
                    badge_color = "#5db8a6"
                else:
                    badge_color = "#5db872"

                folium.Marker(
                    location=[field.center_lat, field.center_lon],
                    icon=folium.DivIcon(
                        html=f'<div style="background:{badge_color};color:#fff;'
                        f'font-size:13px;font-weight:600;padding:4px 10px;'
                        f'border-radius:14px;white-space:nowrap;text-align:center;'
                        f'box-shadow:0 1px 4px rgba(0,0,0,0.25)">'
                        f'{w["temp"]:.0f}°C</div>'
                    ),
                ).add_to(field_group)

        # 图层控制
        folium.LayerControl(collapsed=False, position="topright").add_to(m)

        # 显示地图
        st_folium(m, **_map_size(), key="field_overview_map")
        st.caption("图层控制：地块边界 | 卫星图。天气数据来源：和风天气格点实时天气")

        # Weather summary table
        if field_weather:
            st.markdown("#### 各地块实时天气汇总")
            import pandas as pd
            summary_data = []
            for field in fields:
                w = field_weather.get(field.id)
                if w:
                    summary_data.append({
                        "地块": field.name,
                        "作物": field.current_crop or "-",
                        "温度": f"{w['temp']:.1f}°C",
                        "天气": w['text'],
                        "湿度": f"{w['humidity']}%",
                        "降水量": f"{w['precip']:.1f}mm",
                        "风速": f"{w['windDir']} {w['windSpeed']}km/h",
                        "云量": f"{w.get('cloud', '-')}%",
                    })
            if summary_data:
                df = pd.DataFrame(summary_data)
                st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.caption("未能获取天气数据，请检查 WEATHER_API_KEY 配置。")

    except ImportError as e:
        st.warning(f"地图组件不可用: {e}")
    except Exception as e:
        st.warning(f"加载总览地图失败: {e}")


def _render_drawing_interface(fields, map_manager):
    """Render the map drawing interface for adding a new field."""
    st.markdown("### 绘制新地块")
    st.info(
        "**操作步骤**："
        "1. 点击地图右上角定位按钮获取当前位置 → "
        "2. 使用左侧绘制工具（多边形或矩形）绘制地块边界 → "
        "3. 填写地块信息并保存"
    )

    default_lat, default_lon = 39.9, 116.4
    if st.session_state.get("user_region"):
        try:
            from core.map_manager import get_location_from_address
            coords = get_location_from_address(st.session_state["user_region"])
            if coords:
                default_lat, default_lon = coords
        except Exception:
            pass

    existing_shapes = []
    for field in fields:
        if field.coordinates:
            existing_shapes.append({
                "name": field.name,
                "coordinates": field.coordinates
            })

    try:
        from streamlit_folium import st_folium
        from core.map_manager import create_folium_map, extract_polygon_from_map_data

        m = create_folium_map(
            center_lat=default_lat,
            center_lon=default_lon,
            zoom=14,
            drawn_shapes=existing_shapes
        )

        if st.session_state.get("is_mobile", False):
            map_col = st.container()
            form_col = st.container()
        else:
            map_col, form_col = st.columns([3, 1])

        drawn_coordinates = None
        with map_col:
            map_data = st_folium(m, **_map_size(), key="field_draw_map")

            if map_data:
                drawn_coordinates = extract_polygon_from_map_data(map_data)
                if drawn_coordinates:
                    area_m2, area_mu = map_manager.calculate_area(drawn_coordinates)
                    st.success(
                        f"已绘制地块，预估面积: **{area_mu:.2f}亩** "
                        f"({area_m2:.0f}平方米)"
                    )

        with form_col:
            st.markdown("**地块信息**")
            with st.form("save_field_form"):
                field_name = st.text_input(
                    "地块名称 *",
                    value=f"地块{len(fields)+1}",
                    placeholder="如：东地块、小麦田等"
                )
                soil_opts = ["", "壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"]
                default_soil = st.session_state.get("user_soil_type", "")
                soil_idx = soil_opts.index(default_soil) if default_soil in soil_opts else 0
                field_soil = st.selectbox("土壤类型", soil_opts, index=soil_idx)
                field_crop = st.text_input("当前作物", placeholder="如：小麦、玉米等（可选）")

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    submit_field = st.form_submit_button("保存", width='stretch', type="primary")
                with c2:
                    cancel_field = st.form_submit_button("取消", width='stretch')

                if submit_field and not st.session_state.get("_field_save_lock", False):
                    try:
                        st.session_state["_field_save_lock"] = True
                        if not drawn_coordinates:
                            st.session_state["_field_save_lock"] = False
                            st.error("请先在地图上绘制地块边界！")
                        elif not field_name:
                            st.session_state["_field_save_lock"] = False
                            st.error("请输入地块名称！")
                        else:
                            new_field = map_manager.create_field(
                                name=field_name,
                                coordinates=drawn_coordinates,
                                soil_type=field_soil,
                                current_crop=field_crop
                            )
                            st.success(f"地块'{field_name}'保存成功！面积: {new_field.area_mu:.2f}亩")
                            st.session_state.show_add_field = False
                            st.rerun()
                    except Exception as e:
                        st.session_state["_field_save_lock"] = False
                        st.error(f"保存失败: {e}")
                elif submit_field:
                    st.info("正在处理中...")

                if cancel_field:
                    st.session_state.show_add_field = False
                    st.rerun()

    except ImportError as e:
        st.error(f"缺少必要的地图组件: {e}")
        st.info("请安装: pip install folium streamlit-folium")
        if st.button("返回", key="field_back_btn"):
            st.session_state.show_add_field = False
            st.rerun()


def _render_field_comparison(fields, map_manager):
    """多地块对比 + 种植历史"""
    st.markdown("### 地块对比分析")

    data = map_manager.get_comparison_data()

    col_tab, hist_tab = st.tabs(["地块对比", "种植历史"])

    with col_tab:
        if len(data) >= 2:
            import pandas as pd
            df = pd.DataFrame(data)
            df_display = df.rename(columns={
                "name": "地块名称", "area_mu": "面积(亩)",
                "soil_type": "土壤", "current_crop": "当前作物",
                "history_count": "历史记录数",
            })
            st.dataframe(
                df_display[["地块名称", "面积(亩)", "土壤", "当前作物", "历史记录数"]],
                width='stretch', hide_index=True
            )

            # 面积对比图
            import plotly.express as px
            fig = px.bar(
                df_display, x="地块名称", y="面积(亩)", color="当前作物",
                title="各地块面积对比", text="面积(亩)"
            )
            fig.update_traces(texttemplate="%{text:.1f}亩", textposition="outside")
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("需要至少2个地块才能进行对比分析")

    with hist_tab:
        for field in fields:
            history = map_manager.get_field_history(field.id)
            if history:
                st.markdown(f"**{field.name}**")
                for h in history[:5]:
                    st.caption(
                        f"{h.get('season', '')} — {h.get('crop', '')} "
                        f"(产量: {h.get('yield_amount', '-')}kg) — {h.get('notes', '')}"
                    )
            else:
                st.caption(f"{field.name}: 暂无种植历史")

        # 添加历史记录
        with st.expander("添加种植历史"):
            target_field = st.selectbox(
                "选择地块", [f.name for f in fields], key="hist_field"
            )
            hist_crop = st.text_input("作物名称", key="hist_crop")
            hist_season = st.text_input("种植季节/年份", key="hist_season",
                                       placeholder="如: 2026春")
            hist_yield = st.number_input("产量(kg)", min_value=0.0, key="hist_yield")
            hist_note = st.text_input("备注", key="hist_note")
            if st.button("保存历史记录", key="save_field_history"):
                if hist_crop and target_field:
                    target = next((f for f in fields if f.name == target_field), None)
                    if target:
                        map_manager.add_planting_history(
                            target.id, hist_crop, hist_season, hist_yield, hist_note
                        )
                        st.success(f"已添加 {target_field} 的 {hist_crop} 种植记录")
                        st.rerun()
                else:
                    st.warning("请填写作物名称")
