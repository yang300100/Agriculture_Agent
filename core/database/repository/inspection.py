from core.database.models import InspectionReport
from core.database.repository.base import BaseRepository


class InspectionRepository(BaseRepository[InspectionReport]):
    def __init__(self, session=None):
        super().__init__(InspectionReport, session)
