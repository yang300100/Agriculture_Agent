from core.database.models import DiseaseRisk
from core.database.repository.base import BaseRepository


class DiseaseRiskRepository(BaseRepository[DiseaseRisk]):
    def __init__(self, session=None):
        super().__init__(DiseaseRisk, session)
