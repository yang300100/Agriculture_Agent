"""测试规则引擎"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime
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
        self.engine.add_rule(rule)
        assert self.engine.delete_rule("rule_test_irrigation")
        assert len(self.engine.list_rules()) == 0

    def test_toggle_rule(self):
        rule = self._make_irrigation_rule()
        self.engine.add_rule(rule)
        self.engine.toggle_rule("rule_test_irrigation", False)
        r = self.engine.get_rule("rule_test_irrigation")
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
