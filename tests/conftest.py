"""测试环境总入口：在收集测试前隔离数据库和运行时文件。"""

import os
import shutil
import socket
import tempfile
from ipaddress import ip_address
from pathlib import Path

import pytest

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="agriculture-agent-tests-"))
_TEST_DATA = _TEST_ROOT / "data"
_TEST_DATA.mkdir(parents=True, exist_ok=True)

# 必须在任何业务模块导入数据库引擎前设置，避免测试写入真实 data/。
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}"
os.environ["DATA_STORAGE_DIR"] = str(_TEST_DATA)
os.environ["ENABLE_SCHEDULER"] = "false"
os.environ["REQUIRE_AUTH"] = "false"
# 哥哥，测试不读取真实密钥，也不允许 dotenv 反向覆盖临时数据目录。
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["LLM_API_KEY"] = "test-key-not-real"
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
os.environ["WEATHER_API_KEY"] = ""
os.environ["DL_MODELS_DIR"] = str(_TEST_ROOT / "models")


@pytest.fixture(autouse=True)
def isolate_external_network(monkeypatch):
    """哥哥，保留本机测试通信，拦截真实天气、LLM 和远端硬件连接。"""
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo

    def is_local(host):
        if host is None or host == "localhost":
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii")
        try:
            return ip_address(host).is_loopback
        except ValueError:
            return False

    def connect(sock, address):
        if isinstance(address, tuple) and is_local(address[0]):
            return original_connect(sock, address)
        raise OSError("离线测试禁止连接外部网络")

    def getaddrinfo(host, *args, **kwargs):
        if is_local(host):
            return original_getaddrinfo(host, *args, **kwargs)
        raise socket.gaierror("离线测试禁止解析外部网络地址")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def pytest_sessionfinish(session, exitstatus):
    """测试结束后回收临时数据，不影响项目真实数据库。"""
    try:
        # Windows 下 SQLite 仍被连接池持有时，临时数据库无法删除。
        from core.database import engine as db_engine
        db_engine._engine.dispose()
    except (ImportError, AttributeError):
        pass
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
