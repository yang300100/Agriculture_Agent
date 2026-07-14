"""设备仪表盘 — 设备状态监控 + 快捷操作 + 待确认管理"""

import json
import base64
import streamlit as st
from datetime import datetime
from app.api_client import api, invalidate_cache

# ── 中文翻译映射 ──────────────────────────────────

# 设备能力
CAP_LABELS = {
    "irrigate":    "灌溉",
    "fertigate":   "施肥",
    "ventilate":   "通风",
    "heat":        "加热",
    "cool":        "降温",
    "shade":       "遮阳",
    "light":       "补光",
    "read_sensor": "传感器读数",
    "capture":     "拍摄",
}

# 传感器字段
SENSOR_LABELS = {
    # 通用
    "power":              "电源",
    "status":             "状态",
    # 灌溉
    "flow_rate":          "流量",
    "total_water_liters": "累计用水(L)",
    "last_duration":      "上次灌溉(分)",
    # 土壤/环境
    "temperature":        "温度(°C)",
    "humidity":           "湿度(%)",
    "soil_moisture":      "土壤湿度(%)",
    "ph":                 "pH值",
    # 通风
    "rpm":                "转速(RPM)",
    # 补光
    "brightness_percent": "亮度(%)",
    # 施肥
    "total_fertilizer_kg": "累计施肥(kg)",
    "last_amount_kg":     "上次施肥量(kg)",
    # 加热
    "target_temp":        "目标温度(°C)",
    "current_temp":       "当前温度(°C)",
    # 错误状态保留字段
    "error_reason":       "错误原因",
    "original_driver":    "原始驱动",
    "mqtt_host":          "MQTT主机",
    "mqtt_port":          "MQTT端口",
    "http_url":           "HTTP地址",
    "modbus_mode":        "Modbus模式",
    "modbus_port":        "Modbus端口",
    "slave_id":           "从站地址",
}

# 驱动类型
DRIVER_LABELS = {
    "simulator": "🖥️ 虚拟模拟器",
    "mqtt":      "📡 MQTT",
    "http":      "🌐 HTTP REST",
    "modbus":    "🔧 Modbus",
    "camera":    "📷 摄像头",
}

# 设备在线状态
STATUS_LABELS = {
    "online":  "在线",
    "offline": "离线",
    "error":   "故障",
    "busy":    "忙碌",
}

# 运行状态
RUN_STATE_LABELS = {
    "powered_off":  "关机",
    "standby":      "待机",
    "running":      "工作中",
    "idle":         "空闲",
    "error":        "故障",
}

# 指令
COMMAND_LABELS = {
    "start":    "启动",
    "stop":     "停止",
    "set_param": "设置参数",
}

# 触发方式
TRIGGER_LABELS = {
    "manual": "手动",
    "api":    "API调用",
    "rule":   "自动规则",
    "agent":  "AI决策",
    "schedule": "定时任务",
}


def _label(key, mapping, fallback=None):
    """安全获取中文标签，无匹配时返回原文或 fallback"""
    v = mapping.get(key)
    return v if v is not None else (fallback if fallback is not None else key)


# ── 传感器常用可选项（注册表单下拉菜单）──
COMMON_SENSOR_OPTIONS = [
    "temperature", "humidity", "soil_moisture", "ph",
    "flow_rate", "total_water_liters",
    "rpm",
    "brightness_percent",
    "total_fertilizer_kg",
    "current_temp",
]

def render_devices_page():
    """渲染设备仪表盘"""
    st.markdown("## 🤖 设备仪表盘")

    # ── 顶部状态概览 ──────────────────────────
    devices = api("/api/devices") or []
    pending = api("/api/actions/pending") or []
    logs = api("/api/actions/log", cache_ttl=15) or []

    online_count = sum(1 for d in devices if d.get("status") == "online")
    offline_count = sum(1 for d in devices if d.get("status") == "offline")
    pending_list = [a for a in pending if a.get("status") == "pending"]
    pending_count = len(pending_list)
    today_actions = sum(1 for l in logs if l.get("timestamp", "").startswith(datetime.now().strftime("%Y-%m-%d")))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🟢 在线设备", online_count)
    with col2:
        st.metric("⚠️ 待确认", pending_count)
    with col3:
        st.metric("🔴 离线", offline_count)
    with col4:
        st.metric("⚡ 今日操作", today_actions)

    st.divider()

    # ── 设备列表 ──────────────────────────────
    st.markdown("### 📡 设备列表")

    # 添加设备按钮
    if "show_add_device" not in st.session_state:
        st.session_state.show_add_device = False

    col_title, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("➕ 添加设备", use_container_width=True):
            st.session_state.show_add_device = not st.session_state.show_add_device
            st.rerun()

    # 添加设备表单（不使用 st.form，确保驱动选择动态更新）
    if st.session_state.show_add_device:
        with st.container():
            st.markdown("#### 注册新设备")

            # ── 基本信息（st.form 内）──
            c1, c2 = st.columns(2)
            with c1:
                new_id = st.text_input("设备ID *", placeholder="如: my_pump_01", key="new_dev_id")
                new_name = st.text_input("设备名称 *", placeholder="如: 我的水泵#1", key="new_dev_name")
            with c2:
                # 从地块列表中选择所属地块
                from core.plot_manager import PlotManager
                pm = PlotManager(st.session_state.get("username", "default"))
                plots = pm.list_plots()
                if plots:
                    plot_options = {f"{p['name']} ({p.get('crop', '未指定')})": p for p in plots}
                    selected_label = st.selectbox(
                        "所属地块 *", list(plot_options.keys()),
                        key="new_dev_plot"
                    )
                    selected_plot = plot_options[selected_label] if selected_label else None
                    new_location = selected_plot["name"] if selected_plot else ""
                    new_plot_id = selected_plot["plot_id"] if selected_plot else ""
                else:
                    st.warning("暂无地块，请先创建地块")
                    new_location = ""
                    new_plot_id = ""
                cap_options = ["irrigate", "fertigate", "ventilate", "heat", "cool", "shade", "light", "read_sensor", "capture"]
                new_caps = st.multiselect(
                    "设备能力", cap_options,
                    format_func=lambda x: _label(x, CAP_LABELS),
                    default=["irrigate"], key="new_dev_caps",
                )
            new_sensors = st.multiselect(
                "传感器",
                COMMON_SENSOR_OPTIONS,
                format_func=lambda x: _label(x, SENSOR_LABELS),
                placeholder="选择设备搭载的传感器（可多选）",
                key="new_dev_sensors",
            )

            st.divider()

            # ── 驱动/协议配置（表单外，可实时切换）──
            st.markdown("**🔌 驱动与连接配置**")
            driver_choice = st.selectbox(
                "驱动类型",
                ["mqtt", "http", "modbus", "camera", "simulator"],
                format_func=lambda x: {
                    "mqtt": "📡 MQTT (通用 IoT 消息协议) [推荐]",
                    "http": "🌐 HTTP REST (智能插座/API 设备)",
                    "modbus": "🔧 Modbus RTU/TCP (工业传感器/PLC)",
                    "camera": "📷 摄像头 (USB/RTSP/ESP32-CAM)",
                    "simulator": "🖥️ 虚拟模拟器 (仅开发测试)",
                }.get(x, x),
                key="new_dev_driver",
            )

            # 根据驱动类型显示不同的连接参数
            mqtt_host = mqtt_port = mqtt_topic = mqtt_state_topic = None
            http_base_url = http_api_key = None
            modbus_mode = modbus_port = None
            modbus_slave = 1

            if driver_choice == "mqtt":
                st.caption("MQTT 连接参数 — 设备通过 MQTT Broker 收发消息")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    mqtt_host = st.text_input("Broker 地址", value="localhost", placeholder="如: 192.168.1.100", key="mqtt_host")
                with mc2:
                    mqtt_port = st.number_input("端口", value=1883, min_value=1, max_value=65535, key="mqtt_port")
                with mc3:
                    mqtt_topic = st.text_input("控制主题", value=f"devices/{new_id}/control" if new_id else "", placeholder="如: greenhouse/pump/control", key="mqtt_topic")
                mqtt_state_topic = st.text_input("状态主题 (可选)", placeholder="如: greenhouse/pump/state", key="mqtt_state_topic")

            elif driver_choice == "http":
                st.caption("HTTP REST 连接参数 — 通过 HTTP 请求控制设备")
                hc1, hc2 = st.columns(2)
                with hc1:
                    http_base_url = st.text_input("设备 Base URL *", placeholder="如: http://192.168.1.101:8080", key="http_base_url")
                with hc2:
                    http_api_key = st.text_input("API Key (可选)", placeholder="Bearer token 或 API Key", key="http_api_key", type="password")
                st.caption("设备需支持: GET {base_url}/state 返回状态, POST {base_url}/command 接收指令")

            elif driver_choice == "modbus":
                st.caption("Modbus 连接参数 — 通过串口或 TCP 连接工业设备")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    modbus_mode = st.selectbox("连接模式", ["rtu", "tcp"], key="modbus_mode")
                with mc2:
                    if modbus_mode == "rtu":
                        modbus_port = st.text_input("串口", value="/dev/ttyUSB0", placeholder="如: COM3 或 /dev/ttyUSB0", key="modbus_port")
                    else:
                        modbus_port = st.text_input("IP:端口", value="192.168.1.200:502", key="modbus_port")
                with mc3:
                    modbus_slave = st.number_input("从站地址", value=1, min_value=1, max_value=247, key="modbus_slave")

            elif driver_choice == "camera":
                st.caption("📷 摄像头连接参数 — 支持 USB / IP(RTSP) / ESP32-CAM")
                cc1, cc2 = st.columns(2)
                with cc1:
                    camera_type = st.selectbox(
                        "摄像头类型",
                        ["usb", "ip", "esp32cam"],
                        format_func=lambda x: {
                            "usb": "🔌 USB 摄像头",
                            "ip": "🌐 IP/RTSP 网络摄像头",
                            "esp32cam": "📡 ESP32-CAM (HTTP MJPEG)",
                        }.get(x, x),
                        key="camera_type",
                    )
                with cc2:
                    if camera_type == "usb":
                        camera_source = st.text_input(
                            "设备编号/路径", value="0",
                            placeholder="0 (默认摄像头) 或 /dev/video0",
                            key="camera_source",
                        )
                    elif camera_type == "ip":
                        camera_source = st.text_input(
                            "RTSP/HTTP 地址 *",
                            placeholder="如: rtsp://192.168.1.100:554/stream",
                            key="camera_source",
                        )
                    else:
                        camera_source = st.text_input(
                            "ESP32-CAM URL *",
                            placeholder="如: http://192.168.1.101/capture",
                            key="camera_source",
                        )
                cc3, cc4 = st.columns(2)
                with cc3:
                    camera_username = st.text_input("用户名 (可选)", placeholder="认证用户名", key="camera_username")
                with cc4:
                    camera_password = st.text_input("密码 (可选)", placeholder="认证密码", type="password", key="camera_password")

            else:  # simulator
                st.caption("💡 虚拟设备在内存中运行，无需额外配置。适合开发测试。")

            st.divider()

            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                if st.button("💾 注册设备", use_container_width=True, type="primary"):
                    if not new_id or not new_name:
                        st.error("设备ID和名称不能为空！")
                    else:
                        sensors_list = list(new_sensors)  # multiselect 已返回列表

                        device_config = {
                            "device_id": new_id,
                            "name": new_name,
                            "capabilities": new_caps,
                            "sensors": sensors_list,
                            "location": new_location,
                            "plot_id": new_plot_id,
                            "driver": driver_choice,
                            "initial_state": {"power": False, "status": "powered_off"},
                        }

                        if driver_choice == "mqtt":
                            device_config["connection"] = {
                                "host": mqtt_host, "port": mqtt_port,
                                "control_topic": mqtt_topic,
                                "state_topic": mqtt_state_topic if mqtt_state_topic else None,
                            }
                        elif driver_choice == "http":
                            if not http_base_url:
                                st.error("HTTP 设备必须填写 Base URL！")
                                st.stop()
                            device_config["connection"] = {
                                "base_url": http_base_url.rstrip("/"),
                                "api_key": http_api_key if http_api_key else None,
                            }
                        elif driver_choice == "modbus":
                            device_config["connection"] = {
                                "mode": modbus_mode, "port": modbus_port,
                                "slave_id": modbus_slave,
                            }
                        elif driver_choice == "camera":
                            if not camera_source:
                                st.error("摄像头必须填写设备地址/URL！")
                                st.stop()
                            # 摄像头自动附带 capture 能力
                            if "capture" not in new_caps:
                                device_config["capabilities"] = new_caps + ["capture"]
                            device_config["connection"] = {
                                "camera_type": camera_type,
                                "source": camera_source,
                                "username": camera_username if camera_username else None,
                                "password": camera_password if camera_password else None,
                            }

                        result = api("/api/devices", method="post", json_data=device_config)
                        if result and result.get("success"):
                            drv_label = _label(driver_choice, DRIVER_LABELS)
                            st.success(f"✅ 设备 '{new_name}' 注册成功！驱动: {drv_label}")
                            st.session_state.show_add_device = False
                            invalidate_cache("/api/devices")
                            st.rerun()
                        else:
                            st.error(f"注册失败: {result.get('error', '未知错误') if result else '无响应'}")
            with c_btn2:
                if st.button("❌ 取消", use_container_width=True):
                    st.session_state.show_add_device = False
                    st.rerun()

    if not devices:
        st.info("暂无设备。请点击「添加设备」注册您的第一个设备～")
    else:
        for dev in devices:
            state = dev.get("state", {})
            status_val = dev.get("status", "online")
            status_icon_map = {"online": "🟢", "offline": "🔴", "error": "⚠️"}
            status_icon = status_icon_map.get(status_val, "⚪")
            status_text = _label(status_val, STATUS_LABELS)

            # 设备运行状态
            device_state_status = state.get("status", "powered_off")
            run_icon_map = {"running": "🟢", "standby": "🟡", "idle": "🟡", "error": "🔴", "powered_off": "⚫"}
            run_icon = run_icon_map.get(device_state_status, "⚪")
            run_text = _label(device_state_status, RUN_STATE_LABELS, fallback=device_state_status)

            driver_name = _label(dev.get('driver', 'unknown'), DRIVER_LABELS)

            plot_name = dev.get('plot_name', '')
            plot_crop = dev.get('plot_crop', '')
            plot_badge = f" | 🌍 {plot_name}" if plot_name else ""
            with st.expander(f"{status_icon} **{dev['name']}** — {dev.get('location', '未分配位置')}{plot_badge} | {status_text} | {run_icon} {run_text}"):
                col_a, col_b = st.columns([2, 1])

                with col_a:
                    st.write(f"**设备ID:** {dev['device_id']}")
                    st.write(f"**驱动:** {driver_name}")
                    if plot_name:
                        crop_str = f" ({plot_crop})" if plot_crop else ""
                        st.write(f"**地块:** 🌍 {plot_name}{crop_str}")
                    # 能力标签 — 用中文展示
                    cap_labels = [_label(c, CAP_LABELS) for c in dev.get('capabilities', [])]
                    st.write(f"**能力:** {', '.join(cap_labels)}")

                    # 错误状态提示
                    if state.get("status") == "error" or state.get("error_reason"):
                        st.error(f"⚠️ {state.get('error_reason', '设备状态异常')}")
                        if state.get("original_driver"):
                            orig_drv = _label(state['original_driver'], DRIVER_LABELS)
                            st.caption(f"原始驱动: {orig_drv}（已降级为模拟器显示）")

                    # 传感器数据 — 字段名翻译为中文
                    if state and not state.get("error"):
                        st.markdown("**传感器读数：**")
                        sensor_items = {k: v for k, v in state.items() if isinstance(v, (int, float, bool))}
                        if sensor_items:
                            cols = st.columns(min(len(sensor_items), 4))
                            for i, (k, v) in enumerate(sensor_items.items()):
                                with cols[i % 4]:
                                    label = _label(k, SENSOR_LABELS)
                                    if isinstance(v, bool):
                                        st.metric(label, "✅ 开启" if v else "⭕ 关闭")
                                    elif isinstance(v, float):
                                        st.metric(label, f"{v:.1f}")
                                    else:
                                        st.metric(label, str(v))

                with col_b:
                    caps = dev.get("capabilities", [])
                    is_custom = not dev.get("device_id", "").startswith("virtual_")

                    if is_custom:
                        st.caption("📝 自定义设备")
                        confirm_key = f"confirm_del_{dev['device_id']}"
                        if st.button("🗑️ 删除设备", key=f"del_dev_{dev['device_id']}"):
                            st.session_state[confirm_key] = True
                            st.rerun()
                        # 二次确认，防止误删
                        if st.session_state.get(confirm_key):
                            st.warning(f"⚠️ 确定要删除设备 **{dev['name']}** 吗？此操作不可撤销！")
                            cc1, cc2 = st.columns(2)
                            with cc1:
                                if st.button("✅ 确认删除", key=f"del_confirm_{dev['device_id']}"):
                                    result = api(f"/api/devices/{dev['device_id']}", method="delete")
                                    if result and result.get("success"):
                                        st.success(f"设备 '{dev['name']}' 已删除")
                                        st.session_state.pop(confirm_key, None)
                                        invalidate_cache("/api/devices")
                                        st.rerun()
                                    else:
                                        st.error(f"删除失败: {result.get('error', '未知错误') if result else '无响应'}")
                            with cc2:
                                if st.button("❌ 取消", key=f"del_cancel_{dev['device_id']}"):
                                    st.session_state.pop(confirm_key, None)
                                    st.rerun()
                        st.divider()

                    # 📷 拍照按钮（摄像头设备）
                    if "capture" in caps:
                        if st.button("📷 拍照", key=f"capture_{dev['device_id']}"):
                            result = api(f"/api/devices/{dev['device_id']}/snapshot")
                            if result and result.get("success"):
                                img_data = result.get("image_base64", "")
                                if img_data:
                                    st.image(
                                        base64.b64decode(img_data),
                                        caption=f"{dev['name']} — 实时快照",
                                        use_container_width=True,
                                    )
                                st.success("📸 拍照成功！")
                                st.caption(
                                    f"分辨率: {result.get('metadata', {}).get('width', '?')}x{result.get('metadata', {}).get('height', '?')} "
                                    f"| 大小: {result.get('metadata', {}).get('size_bytes', 0) // 1024} KB"
                                )
                            else:
                                st.error(f"拍照失败: {result.get('error', '未知错误') if result else '无响应'}")
                        st.divider()

                    # ── 设备控制 ──
                    col_r, col_p = st.columns(2)
                    with col_r:
                        if st.button("🔄 重连并刷新", key=f"reconnect_{dev['device_id']}", use_container_width=True):
                            invalidate_cache("/api/devices", "/api/actions/log")
                            from core.device_registry_factory import invalidate_registry_cache
                            invalidate_registry_cache(st.session_state.get("username", "default"))
                            st.rerun()
                    with col_p:
                        if device_state_status == "powered_off":
                            if st.button("⏻ 通电", key=f"power_on_{dev['device_id']}", use_container_width=True):
                                api(f"/api/devices/{dev['device_id']}/command", method="post",
                                    json_data={"command": "power_on", "params": json.dumps({})})
                                invalidate_cache("/api/devices"); st.rerun()
                        elif device_state_status == "error":
                            if st.button("🔄 复位", key=f"reset_{dev['device_id']}", use_container_width=True):
                                api(f"/api/devices/{dev['device_id']}/command", method="post",
                                    json_data={"command": "reset", "params": json.dumps({})})
                                invalidate_cache("/api/devices"); st.rerun()
                        else:
                            if st.button("⏻ 断电", key=f"power_off_{dev['device_id']}", use_container_width=True):
                                api(f"/api/devices/{dev['device_id']}/command", method="post",
                                    json_data={"command": "power_off", "params": json.dumps({})})
                                invalidate_cache("/api/devices"); st.rerun()

                    # ── 参数化操作（待机状态下可用）──
                    if device_state_status == "standby":
                        if "irrigate" in caps:
                            c_dur, c_btn = st.columns([2, 1])
                            with c_dur:
                                dur = st.number_input("灌溉时长(分)", 1, 120, 30, key=f"irr_dur_{dev['device_id']}", label_visibility="collapsed")
                            with c_btn:
                                if st.button("💧 浇水", key=f"irr_btn_{dev['device_id']}", use_container_width=True):
                                    api(f"/api/devices/{dev['device_id']}/command", method="post",
                                        json_data={"command": "start", "params": json.dumps({"duration": dur})})
                                    invalidate_cache("/api/devices"); st.rerun()
                        if "fertigate" in caps:
                            c_amt, c_btn = st.columns([2, 1])
                            with c_amt:
                                amt = st.number_input("施肥量(kg)", 1, 50, 5, key=f"fert_amt_{dev['device_id']}", label_visibility="collapsed")
                            with c_btn:
                                if st.button("🌱 施肥", key=f"fert_btn_{dev['device_id']}", use_container_width=True):
                                    api(f"/api/devices/{dev['device_id']}/command", method="post",
                                        json_data={"command": "start", "params": json.dumps({"amount_kg": amt})})
                                    invalidate_cache("/api/devices"); st.rerun()
                        if "ventilate" in caps:
                            c_spd, c_btn = st.columns([2, 1])
                            with c_spd:
                                spd = st.slider("转速(%)", 10, 100, 60, key=f"vent_spd_{dev['device_id']}", label_visibility="collapsed")
                            with c_btn:
                                if st.button("🌀 通风", key=f"vent_btn_{dev['device_id']}", use_container_width=True):
                                    api(f"/api/devices/{dev['device_id']}/command", method="post",
                                        json_data={"command": "start", "params": json.dumps({"speed_percent": spd})})
                                    invalidate_cache("/api/devices"); st.rerun()
                        if "light" in caps:
                            c_bri, c_btn = st.columns([2, 1])
                            with c_bri:
                                bri = st.slider("亮度(%)", 10, 100, 80, key=f"light_bri_{dev['device_id']}", label_visibility="collapsed")
                            with c_btn:
                                if st.button("💡 补光", key=f"light_btn_{dev['device_id']}", use_container_width=True):
                                    api(f"/api/devices/{dev['device_id']}/command", method="post",
                                        json_data={"command": "start", "params": json.dumps({"brightness_percent": bri})})
                                    invalidate_cache("/api/devices"); st.rerun()
                        if "heat" in caps:
                            c_tmp, c_btn = st.columns([2, 1])
                            with c_tmp:
                                tmp = st.slider("目标温度(°C)", 10, 40, 25, key=f"heat_tmp_{dev['device_id']}", label_visibility="collapsed")
                            with c_btn:
                                if st.button("🔥 加热", key=f"heat_btn_{dev['device_id']}", use_container_width=True):
                                    api(f"/api/devices/{dev['device_id']}/command", method="post",
                                        json_data={"command": "start", "params": json.dumps({"target_temp": tmp})})
                                    invalidate_cache("/api/devices"); st.rerun()
                        # 通用启动按钮（非特定能力设备）
                        if not any(c in caps for c in ["irrigate", "fertigate", "ventilate", "light", "heat"]):
                            if st.button("▶️ 启动", key=f"start_{dev['device_id']}", use_container_width=True):
                                api(f"/api/devices/{dev['device_id']}/command", method="post",
                                    json_data={"command": "start", "params": json.dumps({})})
                                invalidate_cache("/api/devices"); st.rerun()

                    # ── 运行中：停止按钮 ──
                    if device_state_status == "running":
                        if st.button("⏹️ 停止", key=f"stop_{dev['device_id']}", use_container_width=True):
                            api(f"/api/devices/{dev['device_id']}/command", method="post",
                                json_data={"command": "stop", "params": json.dumps({})})
                            invalidate_cache("/api/devices"); st.rerun()

    st.divider()

    # ── 待确认操作 ──────────────────────────────
    st.markdown("### ⚠️ 待确认操作")

    if not pending_list:
        st.success("暂无待确认操作～")
    else:
        for action in pending_list:
            with st.container():
                cmd_label = _label(action.get('command', ''), COMMAND_LABELS)
                st.warning(f"**{action.get('device_id', '未知设备')}** — {cmd_label}")
                st.caption(f"参数: {action.get('params', {})}")
                st.caption(f"原因: {action.get('reason', '需要用户确认')}")

                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("✅ 确认执行", key=f"confirm_{action['id']}"):
                        result = api(f"/api/actions/{action['id']}/confirm", method="post")
                        if result and result.get("success"):
                            st.success("已执行！")
                            invalidate_cache("/api/actions/pending", "/api/actions/log")
                            st.rerun()
                with c2:
                    if st.button("✏️ 修改参数", key=f"edit_{action['id']}"):
                        st.info("参数编辑功能将在后续版本中支持")
                with c3:
                    if st.button("❌ 拒绝", key=f"reject_{action['id']}"):
                        api(f"/api/actions/{action['id']}/reject", method="post")
                        invalidate_cache("/api/actions/pending")
                        st.rerun()

    st.divider()

    # ── 今日执行日志 ────────────────────────────
    st.markdown("### 📋 今日执行日志")
    today_logs = [l for l in logs if l.get("timestamp", "").startswith(datetime.now().strftime("%Y-%m-%d"))]

    if not today_logs:
        st.caption("今日暂无操作记录")
    else:
        for log in reversed(today_logs[-20:]):
            icon = "✅" if log.get("success") else "❌"
            ts = log.get("timestamp", "").split("T")[1][:8] if log.get("timestamp") and "T" in log.get("timestamp", "") else ""
            cmd = _label(log.get('command', ''), COMMAND_LABELS)
            trigger = _label(log.get('trigger', 'manual'), TRIGGER_LABELS)
            st.caption(
                f"{icon} {ts}  **{log.get('device_id', '')}** → "
                f"{cmd} "
                f"({trigger}) — {str(log.get('message', ''))[:60]}"
            )

    # 刷新按钮
    if st.button("🔄 刷新数据"):
        # 先通知后端清除注册中心缓存，强制驱动重连
        api("/api/devices/refresh", method="post")
        invalidate_cache("/api/devices", "/api/actions/pending", "/api/actions/log")
        st.rerun()
