"""Streamlit API 客户端的缓存与请求行为测试。"""

from app import api_client


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.get_calls = []
        self.request_calls = []

    def get(self, url, timeout, headers=None):
        self.get_calls.append((url, timeout))
        return _FakeResponse({"call": len(self.get_calls)})

    def request(self, method, url, json, timeout, headers=None):
        self.request_calls.append((method, url, json, timeout, headers or {}))
        return _FakeResponse({"success": True})

    def delete(self, url, timeout, headers=None):
        return _FakeResponse({"success": True})


def _prepare(monkeypatch):
    state = {"username": "哥哥"}
    session = _FakeSession()
    monkeypatch.setattr(api_client.st, "session_state", state)
    monkeypatch.setattr(api_client, "_get_http_session", lambda: session)
    return state, session


def test_get请求使用默认缓存(monkeypatch):
    _, session = _prepare(monkeypatch)

    first = api_client.api("/api/tasks")
    second = api_client.api("/api/tasks")

    assert first == second == {"call": 1}
    assert len(session.get_calls) == 1


def test_写请求不进入缓存(monkeypatch):
    state, session = _prepare(monkeypatch)

    api_client.api("/api/tasks", "post", {"title": "浇水"})
    api_client.api("/api/tasks", "post", {"title": "施肥"})

    assert len(session.request_calls) == 2
    assert not any(key.startswith("_api_cache_") for key in state)
    assert session.request_calls[0][2]["username"] == "哥哥"


def test_认证请求保留表单中的用户名(monkeypatch):
    _, session = _prepare(monkeypatch)

    api_client.api(
        "/api/auth/login", "post",
        {"username": "新用户", "password": "secret"},
    )

    assert session.request_calls[0][2]["username"] == "新用户"


def test_登录令牌会发送给后端(monkeypatch):
    state, session = _prepare(monkeypatch)
    state["auth_token"] = "signed-token"

    api_client.api("/api/profile", "post", {})

    assert session.request_calls[0][4]["Authorization"] == "Bearer signed-token"


def test_可按接口前缀清除缓存(monkeypatch):
    state, _ = _prepare(monkeypatch)
    api_client.api("/api/tasks")
    api_client.api("/api/profile")

    api_client.invalidate_cache("/api/tasks")

    assert not any("/api/tasks" in key for key in state)
    assert any("/api/profile" in key for key in state)


def test_后端错误会显示明确原因(monkeypatch):
    state = {"username": "哥哥"}
    errors = []

    class ErrorResponse:
        status_code = 503

        def json(self):
            return {"detail": "设备服务暂不可用"}

    class ErrorSession:
        def get(self, url, timeout, headers=None):
            return ErrorResponse()

    monkeypatch.setattr(api_client.st, "session_state", state)
    monkeypatch.setattr(api_client.st, "error", errors.append)
    monkeypatch.setattr(api_client, "_get_http_session", lambda: ErrorSession())

    result = api_client.api("/api/devices", cache_ttl=0)

    assert result is None
    assert errors == ["请求失败：设备服务暂不可用"]
