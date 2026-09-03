"""测试规则引擎"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime
from types import SimpleNamespace
from core.device_rule_engine import RuleEngine, RuleDecision


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine(username="test_user")
        self.engine.rules = []

    def _make_irrigation_rule(self, extra=None):
        rule = {
            "id": "rule_test_irrigation",
            "name": "测试灌溉规则",
            "enabled": True,
            "trigger": {
                "logic": "AND",
                "conditions": [
                    {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30},
                    {"type": "time", "field": "", "op": "between", "value": ["06:00", "20:00"]},
                ]
            },
            "action": {
                "device_id": "irrigation_valve_01",
                "command": "start",
                "params": {"duration": 30},
            },
            "constraints": {
                "max_duration_per_use": 60,
                "max_duration_per_day": 180,
                "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5],
            },
        }
        if extra:
            rule.update(extra)
        return rule

    def test_add_and_list_rules(self):
        rule = self._make_irrigation_rule()
        self.engine.add_rule(rule)
        assert len(self.engine.list_rules()) == 1

    def test_delete_rule(self):
        rule = self._make_irrigation_rule()
        rule_id = self.engine.add_rule(rule)
        assert self.engine.delete_rule(rule_id)
        assert len(self.engine.list_rules()) == 0

    def test_toggle_rule(self):
        rule = self._make_irrigation_rule()
        rule_id = self.engine.add_rule(rule)
        self.engine.toggle_rule(rule_id, False)
        r = self.engine.get_rule(rule_id)
        assert r["enabled"] is False

    def test_trigger_match_and(self):
        rule = self._make_irrigation_rule()
        self.engine.add_rule(rule)
        context = {"sensor_data": {"soil_moisture": 25}}
        matched = self.engine.find_matching_rules(context)
        hour = datetime.now().hour
        if 6 <= hour < 20:
            assert len(matched) == 1
        else:
            assert len(matched) == 0

    def test_trigger_match_or(self):
        rule = self._make_irrigation_rule()
        rule["trigger"]["logic"] = "OR"
        self.engine.add_rule(rule)
        context = {"sensor_data": {"soil_moisture": 25}}
        matched = self.engine.find_matching_rules(context)
        assert len(matched) == 1

    def test_evaluate_auto_execute(self):
        rule = self._make_irrigation_rule()
        # 清除 forbidden_hours 避免当前时间在禁止时段导致失败
        rule["constraints"]["forbidden_hours"] = []
        proposed = {"duration": 30}
        context = {}
        decision, reason, params = self.engine.evaluate_action(rule, proposed, context)
        assert decision == RuleDecision.AUTO_EXECUTE

    def test_evaluate_hard_limit_rejected(self):
        rule = self._make_irrigation_rule()
        proposed = {"duration": 150}
        context = {}
        decision, reason, params = self.engine.evaluate_action(rule, proposed, context)
        assert decision == RuleDecision.REJECTED

    def test_evaluate_constraint_need_confirm(self):
        rule = self._make_irrigation_rule()
        rule["constraints"]["max_duration_per_use"] = 40
        proposed = {"duration": 50}
        context = {}
        decision, reason, params = self.engine.evaluate_action(rule, proposed, context)
        assert decision == RuleDecision.NEED_CONFIRM

    def test_ai_enhance_clamping(self):
        rule = self._make_irrigation_rule()
        rule["ai_enhance"] = {
            "enabled": True,
            "can_adjust": ["duration"],
            "adjust_range": {"duration": [-10, 10]},
        }
        proposed = {"duration": 50}
        proposed = self.engine._apply_ai_enhance(rule["ai_enhance"], proposed, {"duration": 30})
        assert proposed["duration"] == 40


def test_规则扩展字段保存后重新加载不丢失(monkeypatch):
    """模拟数据库往返，验证新增的规则元数据确实写入持久层。"""
    from core.database.repository import devices as device_repositories
    from core.database.repository import users as user_repositories

    class FakeUserRepository:
        user = SimpleNamespace(id=7, username="persistence_user")

        def get_by_username(self, username):
            return self.user if username == self.user.username else None

        def create(self, username, password_hash):
            self.user = SimpleNamespace(id=7, username=username)
            return self.user

    class FakeDeviceRuleRepository:
        rows = []

        def find_by(self, **filters):
            return list(self.rows) if filters.get("user_id") == 7 else []

        def sync_for_user(self, user_id, items):
            assert user_id == 7
            previous = {row.id: row for row in self.__class__.rows}
            rows = []
            for index, item in enumerate(items, start=1):
                payload = dict(item)
                raw_id = payload.pop("id", None)
                row_id = int(raw_id) if raw_id not in (None, "") else index
                row = previous.get(row_id)
                if row is None:
                    row = SimpleNamespace(
                        id=row_id,
                        user_id=user_id,
                        created_at=datetime(2026, 8, 4, 12, 0, 0),
                        **payload,
                    )
                else:
                    for key, value in payload.items():
                        setattr(row, key, value)
                rows.append(row)
            self.__class__.rows = rows
            return rows

    monkeypatch.setattr(user_repositories, "UserRepository", FakeUserRepository)
    monkeypatch.setattr(
        device_repositories, "DeviceRuleRepository", FakeDeviceRuleRepository
    )

    engine = RuleEngine(username="persistence_user")
    persisted_id = engine.add_rule({
        "name": "持久化测试",
        "enabled": True,
        "trigger": {
            "logic": "OR",
            "conditions": [
                {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30},
                {"type": "sensor", "field": "temperature", "op": ">", "value": 35},
            ],
        },
        "action": {
            "device_id": "irrigation_valve_01",
            "capability": "irrigate",
            "command": "start",
            "params": {"duration": 20},
        },
        "constraints": {},
        "ai_enhance": {
            "enabled": True,
            "can_adjust": ["duration"],
            "absolute_range": {"duration": [10, 30]},
        },
        "execution_mode": "confirm",
    })
    original_row = FakeDeviceRuleRepository.rows[0]
    assert engine.update_rule(persisted_id, {"name": "持久化测试（已更新）"})

    reloaded = RuleEngine(username="persistence_user").list_rules()[0]

    assert str(reloaded["id"]) == persisted_id
    assert FakeDeviceRuleRepository.rows[0] is original_row
    assert reloaded["name"] == "持久化测试（已更新）"
    assert reloaded["trigger"]["logic"] == "OR"
    assert reloaded["ai_enhance"] == {
        "enabled": True,
        "can_adjust": ["duration"],
        "absolute_range": {"duration": [10, 30]},
    }
    assert reloaded["execution_mode"] == "confirm"
