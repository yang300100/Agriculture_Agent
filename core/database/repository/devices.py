from core.database.models import (
    DeviceActionLog,
    DeviceConfig,
    DeviceRule,
    DeviceSafetyPolicy,
)
from core.database.repository.base import BaseRepository


class DeviceConfigRepository(BaseRepository[DeviceConfig]):
    def __init__(self, session=None):
        super().__init__(DeviceConfig, session)

    def get_by_device_id(self, device_id: str):
        return self.find_one(device_id=device_id)


class DeviceRuleRepository(BaseRepository[DeviceRule]):
    def __init__(self, session=None):
        super().__init__(DeviceRule, session)

    def sync_for_user(self, user_id: int, items: list[dict]):
        """按规则 ID 原子同步，避免全删重建导致 ID 与日志外键失效。"""
        existing_rows = self.find_by(user_id=user_id)
        existing = {row.id: row for row in existing_rows}
        kept_ids = set()
        synced_rows = []
        try:
            for item in items:
                payload = dict(item)
                raw_id = payload.pop("id", None)
                try:
                    rule_id = int(raw_id) if raw_id not in (None, "") else None
                except (TypeError, ValueError):
                    rule_id = None

                row = existing.get(rule_id)
                if row is None:
                    row = DeviceRule(user_id=user_id, **payload)
                    self.session.add(row)
                    self.session.flush()
                else:
                    for key, value in payload.items():
                        if hasattr(row, key):
                            setattr(row, key, value)
                kept_ids.add(row.id)
                synced_rows.append(row)

            for rule_id, row in existing.items():
                if rule_id not in kept_ids:
                    self.session.delete(row)
            self.session.commit()
            return synced_rows
        except Exception:
            self.session.rollback()
            raise


class DeviceSafetyPolicyRepository(BaseRepository[DeviceSafetyPolicy]):
    def __init__(self, session=None):
        super().__init__(DeviceSafetyPolicy, session)


class DeviceLogRepository(BaseRepository[DeviceActionLog]):
    def __init__(self, session=None):
        super().__init__(DeviceActionLog, session)

    def get_recent(self, user_id: int, limit: int = 100):
        return self.session.query(DeviceActionLog).filter(
            DeviceActionLog.user_id == user_id
        ).order_by(DeviceActionLog.created_at.desc()).limit(limit).all()
