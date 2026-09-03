"""测试环境总入口：在收集测试前隔离数据库和运行时文件。"""

import os
import shutil
import tempfile
from pathlib import Path


_TEST_ROOT = Path(tempfile.mkdtemp(prefix="agriculture-agent-tests-"))
_TEST_DATA = _TEST_ROOT / "data"
_TEST_DATA.mkdir(parents=True, exist_ok=True)

# 必须在任何业务模块导入数据库引擎前设置，避免测试写入真实 data/。
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}"
os.environ["DATA_STORAGE_DIR"] = str(_TEST_DATA)
os.environ["ENABLE_SCHEDULER"] = "false"
os.environ["REQUIRE_AUTH"] = "false"


def pytest_sessionfinish(session, exitstatus):
    """测试结束后回收临时数据，不影响项目真实数据库。"""
    try:
        # Windows 下 SQLite 仍被连接池持有时，临时数据库无法删除。
        from core.database import engine as db_engine
        db_engine._engine.dispose()
    except (ImportError, AttributeError):
        pass
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
