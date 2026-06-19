"""共享 API 客户端 — 所有 Streamlit 视图共用"""

import os, requests, streamlit as st
from datetime import datetime, timedelta

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# 会话级缓存（减少重复 API 调用）
_CACHE_TTL = {"default": 10, "dashboard": 30, "encyclopedia": 300, "solar-terms": 300}


def api(path, method="get", json_data=None, cache_ttl=None):
    """调用 FastAPI 后端，自动带 username，支持会话级缓存"""
    username = st.session_state.get("username", "default")
    sep = "&" if "?" in path else "?"
    url = f"{API_BASE}{path}{sep}username={username}"

    # 会话级缓存
    cache_key = f"_api_cache_{url}"
    if cache_ttl is None:
        # 从路径推断缓存时间
        for prefix, ttl in _CACHE_TTL.items():
            if prefix in path:
                cache_ttl = ttl
                break
    if cache_ttl:
        cached = st.session_state.get(cache_key)
        if cached and (datetime.now() - cached["ts"]).total_seconds() < cache_ttl:
            return cached["data"]

    try:
        if method == "get":
            r = requests.get(url, timeout=15)
        elif method == "delete":
            r = requests.delete(url, timeout=15)
        else:
            # 创建副本，避免修改调用方的原始 dict
            data = dict(json_data) if json_data else {}
            data["username"] = username
            r = requests.post(url, json=data, timeout=60 if "chat" in path else 15)
        if r.status_code == 200:
            result = r.json()
        else:
            result = None
    except Exception:
        result = None

    if cache_ttl and result is not None:
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
