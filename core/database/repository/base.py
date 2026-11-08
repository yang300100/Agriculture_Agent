"""通用 Repository 基类"""
import datetime
from contextlib import contextmanager
from typing import Optional, List, TypeVar, Generic, Type
from sqlalchemy import Date, DateTime
from sqlalchemy.orm import Session, selectinload
from core.database.engine import get_session

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """泛型 CRUD 基类"""

    def __init__(self, model: Type[T], session: Optional[Session] = None):
        self.model = model
        self._session = session
        self._owns_session = session is None
        self._session_depth = 0
        # 预热 Date 列名缓存
        self._date_columns = {
            c.name for c in self.model.__table__.columns
            if isinstance(c.type, Date) and not isinstance(c.type, DateTime)
        }
        self._datetime_columns = {
            c.name for c in self.model.__table__.columns
            if isinstance(c.type, DateTime)
        }

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = get_session()
        return self._session

    @contextmanager
    def session_scope(self):
        """哥哥，嵌套操作共用会话，最外层只回收仓储自己创建的会话。"""
        session = self.session
        self._session_depth += 1
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_depth -= 1
            if self._owns_session and self._session_depth == 0:
                try:
                    session.close()
                finally:
                    self._session = None

    def _query(self, session: Session):
        """哥哥，关闭会话前加载关联，返回的对话仍能读取消息。"""
        return session.query(self.model).options(selectinload("*"))

    def _refresh_for_return(self, session: Session, obj: T):
        """哥哥，提交后的记录先恢复属性，再归还连接。"""
        session.refresh(obj)
        for relation in self.model.__mapper__.relationships:
            getattr(obj, relation.key)

    def _coerce_dates(self, kwargs: dict) -> dict:
        """将字符串日期时间转换为 SQLAlchemy 可接受的 Python 对象。"""
        for key in self._datetime_columns:
            val = kwargs.get(key)
            if isinstance(val, datetime.datetime):
                continue
            if isinstance(val, datetime.date):
                kwargs[key] = datetime.datetime.combine(val, datetime.time())
            elif isinstance(val, str):
                text = val.strip()
                if not text:
                    kwargs[key] = None
                else:
                    try:
                        kwargs[key] = datetime.datetime.fromisoformat(
                            text.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        kwargs[key] = None
            elif val is not None:
                kwargs[key] = None
        for key in self._date_columns:
            val = kwargs.get(key)
            if isinstance(val, str):
                if not val.strip():
                    kwargs[key] = None
                else:
                    try:
                        kwargs[key] = datetime.date.fromisoformat(val[:10])
                    except (ValueError, TypeError):
                        kwargs[key] = None  # 非法日期→None
            elif val is not None and not isinstance(val, datetime.date):
                kwargs[key] = None
        return kwargs

    def get_by_id(self, id: int) -> Optional[T]:
        with self.session_scope() as session:
            return self._query(session).filter(self.model.id == id).first()

    def get_all(self, user_id: Optional[int] = None) -> List[T]:
        with self.session_scope() as session:
            q = self._query(session)
            if user_id is not None and hasattr(self.model, "user_id"):
                q = q.filter(self.model.user_id == user_id)
            return q.all()

    def find_by(self, **filters) -> List[T]:
        with self.session_scope() as session:
            return self._query(session).filter_by(**filters).all()

    def find_one(self, **filters) -> Optional[T]:
        with self.session_scope() as session:
            return self._query(session).filter_by(**filters).first()

    def create(self, **kwargs) -> T:
        kwargs = self._coerce_dates(kwargs)
        obj = self.model(**kwargs)
        with self.session_scope() as session:
            session.add(obj)
            session.commit()
            self._refresh_for_return(session, obj)
            return obj

    def update(self, id: int, **kwargs) -> Optional[T]:
        kwargs = self._coerce_dates(kwargs)
        with self.session_scope() as session:
            obj = self.get_by_id(id)
            if obj is None:
                return None
            for key, value in kwargs.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            session.commit()
            self._refresh_for_return(session, obj)
            return obj

    def delete(self, id: int) -> bool:
        with self.session_scope() as session:
            obj = self.get_by_id(id)
            if obj is None:
                return False
            session.delete(obj)
            session.commit()
            return True

    def count(self, user_id: Optional[int] = None) -> int:
        with self.session_scope() as session:
            q = session.query(self.model)
            if user_id is not None and hasattr(self.model, "user_id"):
                q = q.filter(self.model.user_id == user_id)
            return q.count()

    def bulk_create(self, items: List[dict]) -> List[T]:
        items = [self._coerce_dates(item) for item in items]
        objs = [self.model(**item) for item in items]
        with self.session_scope() as session:
            session.add_all(objs)
            session.commit()
            for obj in objs:
                self._refresh_for_return(session, obj)
            return objs

    def replace_all_for_user(self, user_id: int, items: List[dict]):
        """原子替换某用户的所有记录（先删后插）"""
        with self.session_scope() as session:
            session.query(self.model).filter(self.model.user_id == user_id).delete()
            items = [self._coerce_dates(item) for item in items]
            objs = [self.model(user_id=user_id, **item) for item in items]
            session.add_all(objs)
            session.commit()
