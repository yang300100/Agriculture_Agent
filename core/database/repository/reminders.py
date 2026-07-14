from core.database.models import Reminder
from core.database.repository.base import BaseRepository


class ReminderRepository(BaseRepository[Reminder]):
    def __init__(self, session=None):
        super().__init__(Reminder, session)

    def get_active(self, user_id: int):
        return self.find_by(user_id=user_id, status="active")
