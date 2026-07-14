"""智能种植助手 — Streamlit 纯展示前端"""

import os, sys, logging, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st

from app.api_client import api, invalidate_cache
from app.ui import apply_theme, render_nav_bar, render_common_sidebar
from app.views.chat import (
    render_onboarding_form, render_chat_history, render_image_upload_expander,
    render_chat_input, handle_message_submission,
)
from app.views.profile import render_profile_page
from app.views.fields import render_fields_page
from app.views.finance import render_finance_page
from app.views.calendar import render_calendar_page
from app.views.policy import render_policy_page
from app.views.dashboard import render_dashboard_page
from app.views.encyclopedia import render_encyclopedia_page
from app.views.calculator import render_calculator_page
from app.views.wizard import render_wizard_page
from app.views.devices import render_devices_page
from app.views.rules import render_rules_page
from app.views.docs import render_docs_page

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _render_weather_alert_banner():
    alert_key = "_weather_alert_checked"
    if st.session_state.get(alert_key):
        return
    region = st.session_state.get("user_region", "")
    if not region:
        st.session_state[alert_key] = True
        return
    try:
        data = api(f"/api/weather/alerts/{region}", cache_ttl=1800)
        if data and data.get("has_alert"):
            st.session_state[alert_key] = True
            st.warning(f"⚠️ 天气预警 — {data['region']}：{data['count']}条")
            for a in data.get("alerts", []):
                st.caption(f"- {a.get('type','')}（{a.get('level','')}）: {a.get('desc','')}")
    except Exception:
        st.session_state[alert_key] = True


def _render_proactive_alerts():
    """主动推送横幅：天气预警 + 病虫害风险"""
    key = "_proactive_checked"
    if st.session_state.get(key):
        return
    st.session_state[key] = True

    import json
    alerts = []
    # 天气预警
    wpath = os.path.join("data", "weather_alerts_cache.json")
    if os.path.exists(wpath):
        try:
            with open(wpath, encoding="utf-8") as f:
                w = json.load(f)
            if w.get("has_alert"):
                for a in w.get("alerts", []):
                    alerts.append(f"⚠️ {a.get('type','')}（{a.get('level','')}）: {a.get('desc','')}")
        except Exception:
            pass
    # 病虫害风险
    dpath = os.path.join("data", "disease_risks.json")
    if os.path.exists(dpath):
        try:
            with open(dpath, encoding="utf-8") as f:
                d = json.load(f)
            for r in d.get("risks", [])[:3]:
                if r.get("risk") in ("高", "中"):
                    alerts.append(f"🦠 {r['crop']} {r['disease']} 风险{r['risk']}（{r.get('advice','')[:40]}）")
        except Exception:
            pass
    if alerts:
        with st.container():
            st.warning("**⚠️ 主动预警**")
            for a in alerts[:5]:
                st.caption(a)


def _init_mobile_detection():
    if "is_mobile" in st.session_state:
        return
    from app.agent.config import MOBILE_BREAKPOINT
    from streamlit_javascript import st_javascript as st_js
    raw = st_js("window.innerWidth", key="detect_mobile_width")
    try:
        width = int(raw) if raw else 0
    except (ValueError, TypeError):
        width = 0
    if width > 0:
        st.session_state.is_mobile = width < MOBILE_BREAKPOINT


def _restore_session_context():
    if st.session_state.get("_session_restored"):
        return
    st.session_state["_session_restored"] = True
    # 尝试从 API 加载档案
    try:
        profile = api("/api/profile")
        if profile and profile.get("user_region"):
            st.session_state.user_region = profile.get("user_region", "")
            st.session_state.user_soil_type = profile.get("user_soil_type", "")
            st.session_state.user_farm_size = profile.get("user_farm_size", 1.0)
            st.session_state.user_experience = profile.get("user_experience", "")
            st.session_state.user_goals = profile.get("user_goals", [])
            st.session_state.user_phone = profile.get("user_phone", "")
            st.session_state.user_profile_submitted = True
    except Exception:
        pass


def _load_users():
    # 优先从数据库加载
    try:
        from core.database.repository.users import UserRepository
        repo = UserRepository()
        all_users = repo.get_all()
        if all_users:
            return {u.username: u.password_hash for u in all_users}
    except Exception:
        pass
    # JSON兜底
    path = os.path.join("data", "users.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}

def _save_users(users):
    # 写入数据库
    try:
        from core.database.engine import init_db
        from core.database.repository.users import UserRepository
        init_db()
        repo = UserRepository()
        for username, password in users.items():
            existing = repo.get_by_username(username)
            if not existing:
                repo.create(username=username, password_hash=password if isinstance(password, str) else password.get("password", ""))
    except Exception:
        pass
    # JSON兜底
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "users.json"), "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def _ensure_user_data_dir(username):
    os.makedirs(os.path.join("data", username), exist_ok=True)

def _user_data_dir():
    return os.path.join("data", st.session_state.get("username", "default"))

def _render_auth():
    if "username" in st.session_state:
        return True
    st.markdown("## 🌾 智能种植规划助手")
    tab1, tab2 = st.tabs(["登录", "注册"])
    with tab1:
        user = st.text_input("用户名", key="login_user")
        pwd = st.text_input("密码", type="password", key="login_pwd")
        if st.button("登录", width='stretch'):
            users = _load_users()
            if user in users and users[user] == pwd:
                st.session_state.username = user
                _ensure_user_data_dir(user)
                st.rerun()
            else:
                st.error("用户名或密码错误")
    with tab2:
        new_user = st.text_input("用户名", key="reg_user")
        new_pwd = st.text_input("密码", type="password", key="reg_pwd")
        new_pwd2 = st.text_input("确认密码", type="password", key="reg_pwd2")
        if st.button("注册", width='stretch'):
            if not new_user or not new_pwd:
                st.error("用户名和密码不能为空")
            elif new_pwd != new_pwd2:
                st.error("两次密码不一致")
            else:
                users = _load_users()
                if new_user in users:
                    st.error("用户名已存在")
                else:
                    users[new_user] = new_pwd
                    _save_users(users)
                    _ensure_user_data_dir(new_user)
                    st.session_state.username = new_user
                    st.success("注册成功！")
                    st.rerun()
    st.stop()
    return False


def main():
    st.set_page_config(page_title="智能种植规划助手", page_icon="🌾", layout="wide", initial_sidebar_state="auto")
    if not _render_auth():
        return

    _init_mobile_detection()
    apply_theme()

    _is_mobile = st.session_state.get("is_mobile", False)
    if not _is_mobile:
        st.title("智能种植规划助手")
        st.markdown("### 为您提供作物选择、种植时间规划、农事提醒等全周期种植服务")
    else:
        st.title("🌾 智能种植规划助手")

    render_nav_bar()
    current_page = st.session_state.get("current_page", "dashboard")

    # 初始化会话状态
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _restore_session_context()
    if "autonomy_level" not in st.session_state:
        st.session_state.autonomy_level = "medium"
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = None

    # 页面路由
    if current_page in ("dashboard", "chat"):
        _render_proactive_alerts()

    if current_page == "dashboard":
        st.divider()
        render_dashboard_page()
    elif current_page == "fields":
        st.divider()
        render_fields_page()
    elif current_page == "finance":
        st.divider()
        render_finance_page()
    elif current_page == "profile":
        st.divider()
        render_profile_page()
    elif current_page == "calendar":
        st.divider()
        render_calendar_page()
    elif current_page == "policy":
        st.divider()
        render_policy_page()
    elif current_page == "encyclopedia":
        st.divider()
        render_encyclopedia_page()
    elif current_page == "calculator":
        st.divider()
        render_calculator_page()
    elif current_page == "wizard":
        st.divider()
        render_wizard_page()
    elif current_page == "devices":
        st.divider()
        render_devices_page()
    elif current_page == "rules":
        st.divider()
        render_rules_page()
    elif current_page == "docs":
        st.divider()
        render_docs_page()
    else:  # chat
        st.divider()
        if not render_onboarding_form():
            return
        st.success("基础信息已设置，您可以在「基本信息」页面随时修改。")
        _render_weather_alert_banner()
        render_chat_history()
        render_image_upload_expander()
        user_input = render_chat_input()
        if user_input or st.session_state.get("uploaded_image_base64"):
            handle_message_submission(user_input)

    render_common_sidebar()


if __name__ == "__main__":
    main()
