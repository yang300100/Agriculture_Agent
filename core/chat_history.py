"""对话历史持久化模块 — SQLite 数据库存储"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChatHistoryStore:
    """对话历史持久化存储（纯SQLite）"""

    def __init__(self, storage_dir: str = None):
        # storage_dir 参数保留以兼容旧调用方，不再使用
        from core.database.engine import init_db
        init_db()
        self._ensure_user()

    def _ensure_user(self):
        from core.database.repository.users import UserRepository
        repo = UserRepository()
        self._user = repo.get_by_username("default")
        if not self._user:
            self._user = repo.create(username="default", password_hash="")

    @property
    def _uid(self) -> int:
        return self._user.id

    # ── 内部 DB 操作 ──────────────────────────────

    def _load(self) -> Dict:
        """从DB加载所有会话"""
        from core.database.repository.chat import ChatSessionRepository
        repo = ChatSessionRepository()
        db_sessions = repo.find_by(user_id=self._uid)
        sessions = []
        for s in db_sessions:
            sessions.append({
                "id": str(s.id),
                "title": s.title or "未命名",
                "created_at": s.created_at.isoformat() if s.created_at else "",
                "updated_at": s.updated_at.isoformat() if s.updated_at else "",
                "message_count": len(s.messages) if s.messages else 0,
                "messages": [{"role": m.role, "content": m.content} for m in (s.messages or [])],
            })
        return {"sessions": sessions}

    def _save(self, data: Dict):
        """全量替换DB中的会话"""
        from core.database.repository.chat import ChatSessionRepository, ChatMessageRepository
        session_repo = ChatSessionRepository()
        msg_repo = ChatMessageRepository()
        # 删旧写新
        for s in session_repo.find_by(user_id=self._uid):
            session_repo.delete(s.id)
        for s in data.get("sessions", []):
            sid = session_repo.create(
                user_id=self._uid,
                title=s.get("title", "未命名"),
            ).id
            for m in s.get("messages", []):
                msg_repo.create(
                    session_id=sid,
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                )

    # ── 公共接口 ──────────────────────────────────

    def save_session(self, session_id: str, messages: List[Dict], title: str = ""):
        """保存一个会话（增量更新：先删旧，再插新）"""
        from core.database.repository.chat import ChatSessionRepository, ChatMessageRepository
        session_repo = ChatSessionRepository()
        msg_repo = ChatMessageRepository()

        if not title and messages:
            for m in messages:
                if m.get("role") == "user":
                    title = (m.get("content") or "")[:30]
                    break

        # 查找并删除已有同ID会话
        for s in session_repo.find_by(user_id=self._uid):
            if str(s.id) == session_id:
                session_repo.delete(s.id)
                break

        # 创建新会话
        now = datetime.now()
        sid = session_repo.create(
            user_id=self._uid,
            title=title or "新对话",
            created_at=now,
            updated_at=now,
        ).id
        for m in messages:
            msg_repo.create(
                session_id=sid,
                role=m.get("role", "user"),
                content=m.get("content", ""),
            )

    def load_session(self, session_id: str) -> Optional[List[Dict]]:
        data = self._load()
        for session in data["sessions"]:
            if session.get("id") == session_id:
                return session.get("messages", [])
        return None

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        data = self._load()
        sessions = data.get("sessions", [])
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return [{
            "id": s.get("id"),
            "title": s.get("title", "未命名"),
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at"),
            "message_count": s.get("message_count", 0),
        } for s in sessions[:limit]]

    def delete_session(self, session_id: str) -> bool:
        from core.database.repository.chat import ChatSessionRepository
        repo = ChatSessionRepository()
        for s in repo.find_by(user_id=self._uid):
            if str(s.id) == session_id:
                repo.delete(s.id)
                return True
        return False

    def get_latest_session_id(self) -> Optional[str]:
        sessions = self.list_sessions(limit=1)
        return sessions[0]["id"] if sessions else None
