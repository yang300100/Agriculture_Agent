"""地块分区数据访问。"""

from core.database.models import FieldZone
from core.database.repository.base import BaseRepository


class FieldZoneRepository(BaseRepository[FieldZone]):
    def __init__(self, session=None):
        super().__init__(FieldZone, session)
