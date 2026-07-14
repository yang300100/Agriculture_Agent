from core.database.models import Field
from core.database.repository.base import BaseRepository


class FieldRepository(BaseRepository[Field]):
    def __init__(self, session=None):
        super().__init__(Field, session)
