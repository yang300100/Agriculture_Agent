"""Chat page — 对话页面（仅 UI，后端通过 API 调用）+ 多会话管理"""

import os, json
from datetime import datetime
import streamlit as st
from app.api_client import api


def _render_device_message(content: str) -> bool:
    """检测并渲染设备控制消息为富卡片。返回 True 表示已渲染，False 表示走常规渲染"""
    if any(kw in content for kw in ["指令已执行", "执行失败", "操作预览", "设备控制"]):
        # 用特殊样式渲染设备控制消息
        if "✅" in content:
            st.success(content)
        elif "❌" in content:
            st.error(content)
        elif "⚠️" in content:
            st.warning(content)
        else:
            st.info(content)
        return True
    return False


def _load_profile_from_disk():
    """从磁盘加载用户档案（兼容旧数据）"""
    path = os.path.join("data", "user_profile.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_profile_to_disk():
    profile = {
        "user_region": st.session_state.get("user_region", ""),
        "user_soil_type": st.session_state.get("user_soil_type", ""),
        "user_farm_size": st.session_state.get("user_farm_size", 1.0),
        "user_experience": st.session_state.get("user_experience", ""),
        "user_goals": st.session_state.get("user_goals", []),
        "user_phone": st.session_state.get("user_phone", ""),
    }
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "user_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    # 同时通过 API 保存
    api("/api/profile", "post", profile)


def render_onboarding_form():
    if st.session_state.get("_onboarding_lock"):
        st.session_state["_onboarding_lock"] = False
    if "user_profile_submitted" not in st.session_state:
        st.session_state.user_profile_submitted = False

    if not st.session_state.user_profile_submitted:
        saved = _load_profile_from_disk()
        if saved:
            st.session_state.user_region = saved.get("user_region", "")
            st.session_state.user_soil_type = saved.get("user_soil_type", "")
            st.session_state.user_farm_size = saved.get("user_farm_size", 1.0)
            st.session_state.user_experience = saved.get("user_experience", "")
            st.session_state.user_goals = saved.get("user_goals", [])
            st.session_state.user_phone = saved.get("user_phone", "")
            st.session_state.user_profile_submitted = True

    if st.session_state.user_profile_submitted:
        return True

    with st.container():
        st.markdown("## 欢迎使用智能种植规划助手")
        st.info("请填写以下基础信息，以便我们为您提供更精准的种植建议。")
        with st.form("user_profile_form"):
            c1, c2 = st.columns(2)
            with c1:
                region = st.text_input("所在地区", placeholder="如：华北、山东")
                soil_type = st.selectbox("土壤类型", ["请选择", "壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"])
            with c2:
                farm_size = st.number_input("种植面积（亩）", min_value=0.0, max_value=10000.0, value=0.0, step=0.5)
                experience = st.selectbox("种植经验", ["请选择", "新手（1年以下）", "初级（1-3年）", "中级（3-5年）", "高级（5-10年）", "专家（10年以上）"])
            goals = st.multiselect("种植目标", ["高产", "优质", "省工", "节水", "有机", "多样化种植", "经济效益", "自用为主"])
            phone = st.text_input("手机号码（可选）", placeholder="如：13800138000")
            submitted = st.form_submit_button("开始使用", width='stretch')
            if submitted and not st.session_state.get("_onboarding_lock"):
                try:
                    st.session_state["_onboarding_lock"] = True
                    st.session_state.user_region = region if region else ""
                    st.session_state.user_soil_type = soil_type if soil_type != "请选择" else ""
                    st.session_state.user_farm_size = farm_size if farm_size > 0 else 1.0
                    st.session_state.user_experience = experience if experience != "请选择" else ""
                    st.session_state.user_goals = goals if goals else []
                    st.session_state.user_phone = phone
                    st.session_state.user_profile_submitted = True
                    _save_profile_to_disk()
                    st.success("信息已保存！")
                    st.rerun()
                except Exception:
                    st.session_state["_onboarding_lock"] = False
                    st.error("保存失败，请重试")
    st.stop()
    return False


def render_chat_history():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                if not _render_device_message(msg["content"]):
                    st.markdown(msg["content"])
                if st.button("🔊", key=f"tts_{i}", help="朗读回答"):
                    from core.tts_components import tts_speak
                    tts_speak(msg["content"], key=f"tts_speak_{i}")


def render_image_upload_expander():
    with st.expander("上传农作物图片进行诊断（可选）"):
        uploaded = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"], key="crop_image_uploader")
        if uploaded:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded, caption="预览", width='stretch')
            with col2:
                st.info("图片已加载，可以在下方输入问题或直接发送空消息开始分析。")
            import base64
            st.session_state.uploaded_image_base64 = base64.b64encode(uploaded.getvalue()).decode()
            st.session_state.uploaded_image_mime = uploaded.type


def render_chat_input():
    if st.session_state.get("recording"):
        with st.spinner("正在聆听..."):
            from core.voice_components import voice_input_button
            result = voice_input_button(key="voice_recorder")
            if result:
                st.session_state.recording = False
                if result.startswith("ERROR:"):
                    st.error(f"语音识别失败: {result[6:]}")
                else:
                    st.session_state.voice_text = result
                st.rerun()
    if st.session_state.get("voice_text"):
        text = st.session_state.voice_text
        st.session_state.voice_text = None
        return text

    is_mobile = st.session_state.get("is_mobile", False)
    with st.form("chat_form", clear_on_submit=True):
        if is_mobile:
            text = st.text_input("输入", label_visibility="collapsed", placeholder="请输入...", key="chat_text_input")
            bc = st.columns([1, 1])
            with bc[0]: send = st.form_submit_button("发送", width='stretch')
            with bc[1]: voice = st.form_submit_button("🎤 语音", width='stretch')
        else:
            cols = st.columns([10, 1.2, 0.8])
            with cols[0]:
                text = st.text_input("输入", label_visibility="collapsed", placeholder="请输入您的问题...", key="chat_text_input")
            with cols[1]: send = st.form_submit_button("发送", width='stretch')
            with cols[2]: voice = st.form_submit_button("🎤", width='stretch')

    if voice:
        st.session_state.recording = True
        st.rerun()
    if send and text and text.strip():
        return text.strip()
    return None


def handle_message_submission(user_input):
    if not user_input and not st.session_state.get("uploaded_image_base64"):
        return

    display_content = user_input or "请分析这张农作物图片"
    with st.chat_message("user"):
        st.markdown(display_content)
    st.session_state.chat_history.append({"role": "user", "content": display_content})

    try:
        # 构建请求
        req = {
            "user_question": user_input or "请分析这张农作物图片",
            "user_profile": {
                "region": st.session_state.get("user_region", ""),
                "soil_type": st.session_state.get("user_soil_type", ""),
                "farm_size": st.session_state.get("user_farm_size", 1.0),
                "experience": st.session_state.get("user_experience", ""),
                "goals": st.session_state.get("user_goals", []),
            },
        }
        if st.session_state.get("uploaded_image_base64"):
            req["image_data"] = st.session_state.uploaded_image_base64
            req["image_mime_type"] = st.session_state.uploaded_image_mime
            st.session_state.uploaded_image_base64 = None
            st.session_state.uploaded_image_mime = None

        resp = api("/api/chat", "post", req)
        if resp:
            answer = resp.get("final_answer", "抱歉，后端服务未响应。")
        else:
            answer = "抱歉，后端服务连接失败。请确认 FastAPI 已启动（python app/api_server.py）。"

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # 语音指令自动 TTS 确认
        cmd_prefixes = ("记账：", "记一笔：", "提醒：", "设置提醒：", "添加任务：", "记录进度：", "查天气")
        if user_input and any(user_input.startswith(p) for p in cmd_prefixes):
            from core.tts_components import tts_speak
            tts_speak(answer, key=f"tts_cmd_{len(st.session_state.chat_history)}")

    except Exception as e:
        st.error(f"请求失败：{e}")
