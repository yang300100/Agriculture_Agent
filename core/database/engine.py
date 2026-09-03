"""SQLAlchemy 引擎与 Session 工厂"""
import os
import threading
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SASession

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "agriculture.db")
DB_URL = os.getenv("DATABASE_URL") or f"sqlite:///{os.path.abspath(DB_PATH)}"
# 部分托管平台仍会下发旧式 postgres://，SQLAlchemy 需要 postgresql://。
if DB_URL.startswith("postgres://"):
    DB_URL = "postgresql://" + DB_URL[len("postgres://"):]

_engine_options = {
    "echo": False,
    "pool_pre_ping": True,
    "pool_size": 20,
    "max_overflow": 30,
}
if DB_URL.startswith("sqlite"):
    _engine_options["connect_args"] = {"check_same_thread": False}

_engine = create_engine(DB_URL, **_engine_options)
Session = sessionmaker(bind=_engine)
_init_lock = threading.Lock()
_initialized = False


def get_session() -> SASession:
    """获取一个已完成建表与迁移的数据库会话。"""
    init_db()
    return Session()


def init_db():
    """初始化数据库并执行可重复的版本化迁移。"""
    global _initialized
    if _initialized:
        return
    from core.database.models import Base
    from core.database.migrations import apply_migrations
    with _init_lock:
        if _initialized:
            return
        Base.metadata.create_all(_engine)
        apply_migrations(_engine)
        _initialized = True
