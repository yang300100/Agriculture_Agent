from core.database.models import ChatSession, ChatMessage
from core.database.repository.base import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    def __init__(self, session=None):
        super().__init__(ChatSession, session)


class ChatMessageRepository(BaseRepository[ChatMessage]):
    def __init__(self, session=None):
        super().__init__(ChatMessage, session)
