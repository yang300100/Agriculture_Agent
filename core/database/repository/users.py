from core.database.models import User, UserProfile
from core.database.repository.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, session=None):
        super().__init__(User, session)

    def get_by_username(self, username: str):
        return self.find_one(username=username)


class UserProfileRepository(BaseRepository[UserProfile]):
    def __init__(self, session=None):
        super().__init__(UserProfile, session)

    def get_by_user_id(self, user_id: int):
        return self.find_one(user_id=user_id)
