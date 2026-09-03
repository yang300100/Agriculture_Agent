"""启动编排的配置与健康检查测试。"""

import urllib.error

from app import start


def test_后端地址支持自定义端口(monkeypatch):
    monkeypatch.delenv("API_BASE", raising=False)
    monkeypatch.setenv("PORT", "9123")

    assert start._backend_base_url() == "http://localhost:9123"


def test_后端地址默认使用农业专用端口(monkeypatch):
    monkeypatch.delenv("API_BASE", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    assert start._backend_base_url() == "http://localhost:18001"


def test_显式api地址优先且移除末尾斜杠(monkeypatch):
    monkeypatch.setenv("API_BASE", "https://farm.example.com/")
    monkeypatch.setenv("PORT", "9123")

    assert start._backend_base_url() == "https://farm.example.com"


def test_健康检查使用公开端点(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def read(self):
            return b'{"service":"agriculture-agent","status":"ok"}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(url, timeout):
        captured.update(url=url, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(start.urllib.request, "urlopen", fake_urlopen)

    assert start._backend_is_ready("http://localhost:9123") is True
    assert captured == {
        "url": "http://localhost:9123/api/health",
        "timeout": 3,
    }


def test_健康检查拒绝其他服务的成功响应(monkeypatch):
    class FakeResponse:
        status = 200

        def read(self):
            return b'{"service":"virtual-world-core","status":"ok"}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(start.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    assert start._backend_is_ready("http://localhost:8000") is False


def test_健康检查网络失败时返回false(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(start.urllib.request, "urlopen", fail)

    assert start._backend_is_ready("http://localhost:9123") is False
