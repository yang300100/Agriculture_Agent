from core.database.models import FinanceRecord
from core.database.repository.base import BaseRepository


class FinanceRepository(BaseRepository[FinanceRecord]):
    def __init__(self, session=None):
        super().__init__(FinanceRecord, session)

    def get_by_date_range(self, user_id: int, start_date, end_date):
        return self.session.query(FinanceRecord).filter(
            FinanceRecord.user_id == user_id,
            FinanceRecord.date >= start_date,
            FinanceRecord.date <= end_date,
        ).all()
