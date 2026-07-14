from core.database.models import PlantingPlan, PlantingTask
from core.database.repository.base import BaseRepository


class PlantingPlanRepository(BaseRepository[PlantingPlan]):
    def __init__(self, session=None):
        super().__init__(PlantingPlan, session)


class PlantingTaskRepository(BaseRepository[PlantingTask]):
    def __init__(self, session=None):
        super().__init__(PlantingTask, session)

    def get_by_plan(self, plan_id: int):
        return self.find_by(plan_id=plan_id)
