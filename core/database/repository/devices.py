from core.database.models import DeviceConfig, DeviceRule, DeviceActionLog
from core.database.repository.base import BaseRepository


class DeviceConfigRepository(BaseRepository[DeviceConfig]):
    def __init__(self, session=None):
        super().__init__(DeviceConfig, session)

    def get_by_device_id(self, device_id: str):
        return self.find_one(device_id=device_id)


class DeviceRuleRepository(BaseRepository[DeviceRule]):
    def __init__(self, session=None):
        super().__init__(DeviceRule, session)


class DeviceLogRepository(BaseRepository[DeviceActionLog]):
    def __init__(self, session=None):
        super().__init__(DeviceActionLog, session)

    def get_recent(self, user_id: int, limit: int = 100):
        return self.session.query(DeviceActionLog).filter(
            DeviceActionLog.user_id == user_id
        ).order_by(DeviceActionLog.created_at.desc()).limit(limit).all()
