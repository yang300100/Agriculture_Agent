"""对话历史持久化模块 — JSON 文件存储，支持多会话浏览"""

import os
import json
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import dotenv

dotenv.load_dotenv()
DEFAULT_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR", "data")

logger = logging.getLogger(__name__)


class ChatHistoryStore:
    """对话历史持久化存储"""

    def __init__(self, storage_dir: str = None):
        self.storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self.store_file = os.path.join(self.storage_dir, "chat_history.json")
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        if not os.path.exists(self.store_file):
            with open(self.store_file, 'w', encoding='utf-8') as f:
                json.dump({"sessions": []}, f, ensure_ascii=False, indent=2)

    def _load(self) -> Dict:
        # 优先从数据库加载
        try:
            db_data = self._load_from_db()
            if db_data is not None:
                return db_data
        except Exception as e:
            logger.debug("数据库加载对话失败，回退JSON: %s", e)

        if not os.path.exists(self.store_file):
            return {"sessions": []}
        try:
            with open(self.store_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict) or "sessions" not in data:
                raise ValueError("chat_history.json 格式无效")
            return data
        except json.JSONDecodeError:
            if os.path.exists(self.store_file):
                backup_path = f"{self.store_file}.corrupted.{datetime.now().strftime('%Y%m%d%H%M%S')}"
                try:
                    shutil.copy2(self.store_file, backup_path)
                    logger.error("chat_history.json JSON 解析失败，已备份至 %s", backup_path)
                except Exception:
                    logger.error("备份损坏的 chat_history.json 失败")
            return {"sessions": []}
        except (OSError, IOError) as e:
            logger.error("chat_history.json 读取 IO 错误: %s", e)
            raise

    def _save(self, data: Dict):
        # 原子写入JSON
        tmp_file = self.store_file + ".tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self.store_file)
        # 同步到数据库
        try:
            self._save_to_db(data)
        except Exception as e:
            logger.debug("数据库同步对话失败: %s", e)

    def _load_from_db(self) -> Optional[Dict]:
        from core.database.repository.chat import ChatSessionRepository
        from core.database.repository.users import UserRepository
        user_repo = UserRepository()
        user = user_repo.get_by_username("default")
        if not user:
            return None
        session_repo = ChatSessionRepository()
        db_sessions = session_repo.find_by(user_id=user.id)
        if not db_sessions:
            return None
        sessions = []
        for s in db_sessions:
            sessions.append({
                "id": str(s.id),
                "title": s.title or "未命名",
                "created_at": s.created_at.isoformat() if s.created_at else "",
                "updated_at": s.updated_at.isoformat() if s.updated_at else "",
                "message_count": len(s.messages) if s.messages else 0,
                "messages": [{
                    "role": m.role,
                    "content": m.content,
                } for m in (s.messages or [])],
            })
        return {"sessions": sessions}

    def _save_to_db(self, data: Dict):
        from core.database.engine import init_db
        from core.database.repository.chat import ChatSessionRepository, ChatMessageRepository
        from core.database.repository.users import UserRepository
        init_db()
        user_repo = UserRepository()
        user = user_repo.get_by_username("default")
        if not user:
            user = user_repo.create(username="default", password_hash="")
        session_repo = ChatSessionRepository()
        msg_repo = ChatMessageRepository()
        # 删除旧会话（替换为最新数据）
        existing = session_repo.find_by(user_id=user.id)
        for s in existing:
            session_repo.delete(s.id)
        for s in data.get("sessions", []):
            sid = session_repo.create(
                user_id=user.id,
                title=s.get("title", "未命名"),
            ).id
            for m in s.get("messages", []):
                msg_repo.create(
                    session_id=sid,
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                )

    def save_session(self, session_id: str, messages: List[Dict], title: str = ""):
        """保存一个会话"""
        data = self._load()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 生成标题（用第一条用户消息）
        if not title and messages:
            for m in messages:
                if m.get("role") == "user":
                    title = (m.get("content") or "")[:30]
                    break

        # 查找现有会话
        for session in data["sessions"]:
            if session.get("id") == session_id:
                session["messages"] = messages
                session["updated_at"] = now
                session["title"] = title or session.get("title", "未命名")
                session["message_count"] = len(messages)
                self._save(data)
                return

        # 新会话
        data["sessions"].append({
            "id": session_id,
            "title": title or "新对话",
            "created_at": now,
            "updated_at": now,
            "message_count": len(messages),
            "messages": messages,
        })
        self._save(data)

    def load_session(self, session_id: str) -> Optional[List[Dict]]:
        """加载指定会话"""
        data = self._load()
        for session in data["sessions"]:
            if session.get("id") == session_id:
                return session.get("messages", [])
        return None

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """列出所有会话摘要（不含完整消息）"""
        data = self._load()
        sessions = data.get("sessions", [])
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        summaries = []
        for s in sessions[:limit]:
            summaries.append({
                "id": s.get("id"),
                "title": s.get("title", "未命名"),
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at"),
                "message_count": s.get("message_count", 0),
            })
        return summaries

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话"""
        data = self._load()
        original_count = len(data["sessions"])
        data["sessions"] = [s for s in data["sessions"] if s.get("id") != session_id]
        if len(data["sessions"]) < original_count:
            self._save(data)
            return True
        return False

    def get_latest_session_id(self) -> Optional[str]:
        """获取最近会话的ID"""
        sessions = self.list_sessions(limit=1)
        return sessions[0]["id"] if sessions else None
