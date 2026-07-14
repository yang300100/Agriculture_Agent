"""通用 Repository 基类"""
from typing import Optional, List, TypeVar, Generic, Type
from sqlalchemy.orm import Session
from core.database.engine import get_session

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """泛型 CRUD 基类"""

    def __init__(self, model: Type[T], session: Optional[Session] = None):
        self.model = model
        self._session = session

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = get_session()
        return self._session

    def get_by_id(self, id: int) -> Optional[T]:
        return self.session.query(self.model).filter(self.model.id == id).first()

    def get_all(self, user_id: Optional[int] = None) -> List[T]:
        q = self.session.query(self.model)
        if user_id is not None and hasattr(self.model, "user_id"):
            q = q.filter(self.model.user_id == user_id)
        return q.all()

    def find_by(self, **filters) -> List[T]:
        return self.session.query(self.model).filter_by(**filters).all()

    def find_one(self, **filters) -> Optional[T]:
        return self.session.query(self.model).filter_by(**filters).first()

    def create(self, **kwargs) -> T:
        obj = self.model(**kwargs)
        self.session.add(obj)
        self.session.commit()
        self.session.refresh(obj)
        return obj

    def update(self, id: int, **kwargs) -> Optional[T]:
        obj = self.get_by_id(id)
        if obj is None:
            return None
        for key, value in kwargs.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        self.session.commit()
        return obj

    def delete(self, id: int) -> bool:
        obj = self.get_by_id(id)
        if obj is None:
            return False
        self.session.delete(obj)
        self.session.commit()
        return True

    def count(self, user_id: Optional[int] = None) -> int:
        q = self.session.query(self.model)
        if user_id is not None and hasattr(self.model, "user_id"):
            q = q.filter(self.model.user_id == user_id)
        return q.count()

    def bulk_create(self, items: List[dict]) -> List[T]:
        objs = [self.model(**item) for item in items]
        self.session.add_all(objs)
        self.session.commit()
        return objs

    def replace_all_for_user(self, user_id: int, items: List[dict]):
        """原子替换某用户的所有记录（先删后插）"""
        self.session.query(self.model).filter(self.model.user_id == user_id).delete()
        objs = [self.model(user_id=user_id, **item) for item in items]
        self.session.add_all(objs)
        self.session.commit()
