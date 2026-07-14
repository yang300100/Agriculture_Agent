"""SQLAlchemy 引擎与 Session 工厂"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SASession

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "agriculture.db")
DB_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

_engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=_engine)


def get_session() -> SASession:
    """获取一个新的数据库会话"""
    return Session()


def init_db():
    """初始化数据库，创建所有表"""
    from core.database.models import Base
    Base.metadata.create_all(_engine)
