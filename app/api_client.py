"""共享 API 客户端 — 所有 Streamlit 视图共用"""

import os
import logging
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE", "http://localhost:18001")
logger = logging.getLogger(__name__)

# 会话级缓存（减少重复 API 调用）
_CACHE_TTL = {"default": 10, "dashboard": 30, "encyclopedia": 300, "solar-terms": 300}


@st.cache_resource
def _get_http_session():
    """复用后端 HTTP 连接，避免每次切页都重新建立 TCP 连接。"""
    return requests.Session()


def api(path, method="get", json_data=None, cache_ttl=None):
    """调用 FastAPI 后端，自动带 username，支持会话级缓存"""
    method = method.lower()
    username = st.session_state.get("username", "default")
    sep = "&" if "?" in path else "?"
    url = f"{API_BASE}{path}{sep}username={username}"

    # 会话级缓存
    cache_key = f"_api_cache_{url}"
    # 只有只读请求可以缓存，写请求必须实时到达后端。
    if method == "get" and cache_ttl is None:
        cache_ttl = _CACHE_TTL["default"]
        # 从路径推断更合适的缓存时间
        for prefix, ttl in _CACHE_TTL.items():
            if prefix != "default" and prefix in path:
                cache_ttl = ttl
                break
    use_cache = method == "get" and bool(cache_ttl)
    if use_cache:
        cached = st.session_state.get(cache_key)
        if cached and (datetime.now() - cached["ts"]).total_seconds() < cache_ttl:
            return cached["data"]

    try:
        session = _get_http_session()
        headers = {}
        auth_token = st.session_state.get("auth_token", "")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if method == "get":
            r = session.get(url, timeout=30, headers=headers)
        elif method == "delete":
            r = session.delete(url, timeout=30, headers=headers)
        else:
            # 创建副本，避免修改调用方的原始 dict
            data = dict(json_data) if json_data else {}
            data.setdefault("username", username)
            # 聊天请求超时设为 600s（Agent 流水线包含多次 LLM 调用，可能很长）
            r = session.request(
                method,
                url,
                json=data,
                headers=headers,
                timeout=600 if "chat" in path else 30,
            )
        if 200 <= r.status_code < 300:
            result = r.json()
        else:
            try:
                error_body = r.json()
                detail = error_body.get("detail") or error_body.get("error")
            except (ValueError, AttributeError):
                detail = ""
            detail = detail or f"后端返回 HTTP {r.status_code}"
            st.error(f"请求失败：{detail}")
            logger.warning(
                "API 请求失败: method=%s path=%s status=%s",
                method,
                path,
                r.status_code,
            )
            result = None
    except requests.exceptions.Timeout:
        st.error("⏰ 请求超时：后端处理时间过长，请稍后重试或简化问题。")
        result = None
    except requests.exceptions.ConnectionError:
        st.error("🔌 无法连接后端，请确认 FastAPI 已启动（python app/api_server.py）。")
        result = None
    except Exception:
        logger.exception("API 请求异常: method=%s path=%s", method, path)
        st.error("请求处理失败，请稍后重试。")
        result = None

    if use_cache and result is not None:
        st.session_state[cache_key] = {"ts": datetime.now(), "data": result}
    return result


def invalidate_cache(*prefixes):
    """清除特定路径的缓存"""
    keys_to_del = []
    for prefix in prefixes:
        cache_key = f"_api_cache_{API_BASE}{prefix}"
        for k in list(st.session_state.keys()):
            if k.startswith(cache_key):
                keys_to_del.append(k)
    for k in keys_to_del:
        del st.session_state[k]
