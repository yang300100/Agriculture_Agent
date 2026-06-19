"""自主决策编排器 单元测试"""
import os, sys, json, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.autonomous_farm_manager import (
    CameraView, FarmState, DecisionPlan, ActionResult, CycleReport,
    AutonomousFarmManager,
)


class TestDataStructures:
    """数据结构序列化/反序列化测试"""

    def test_camera_view_defaults(self):
        cv = CameraView(device_id="cam_01", location="大棚A")
        assert cv.image_base64 is None
        assert cv.vision_analysis is None
        assert cv.error is None

    def test_camera_view_with_error(self):
        cv = CameraView(device_id="cam_02", location="大棚B", error="设备离线")
        assert cv.error == "设备离线"
        assert cv.image_base64 is None

    def test_farm_state_empty(self):
        state = FarmState(region="大棚A", username="test")
        assert state.camera_views == []
        assert state.sensor_readings == {}
        assert state.current_weather is None

    def test_farm_state_with_data(self):
        state = FarmState(
            region="大棚A", username="test",
            camera_views=[CameraView(device_id="c1", location="大棚A")],
            sensor_readings={"soil_moisture": 28.5},
            current_weather={"temperature": 22},
        )
        assert len(state.camera_views) == 1
        assert state.sensor_readings["soil_moisture"] == 28.5

    def test_decision_plan_from_dict(self):
        data = {
            "region": "大棚A",
            "overall_assessment": "土壤偏干",
            "actions": [{"action": "irrigate", "params": {"duration": 25}}],
            "follow_up": "3天后复查",
        }
        plan = DecisionPlan(**data)
        assert plan.region == "大棚A"
        assert len(plan.actions) == 1

    def test_decision_plan_empty_actions(self):
        plan = DecisionPlan(region="大棚A", overall_assessment="一切正常")
        assert plan.actions == []

    def test_action_result_success(self):
        ar = ActionResult(action="irrigate", device_id="dev_01", success=True,
                          message="执行成功", executed_params={"duration": 25})
        assert ar.success is True

    def test_cycle_report_fields(self):
        report = CycleReport(
            cycle_id="cycle_001", username="test", region="大棚A",
            timestamp="2026-06-19T14:00:00", summary="完成",
        )
        assert report.cycle_id == "cycle_001"
        assert report.fallback_used is False
        assert report.duration_ms == 0


class TestAutonomousFarmManager:
    """编排器核心测试"""

    def test_init(self):
        mgr = AutonomousFarmManager()
        assert mgr is not None
        assert hasattr(mgr, 'hard_limits')

    def test_discover_regions_from_devices(self):
        """区域发现：从设备列表按 location 分组"""
        class MockDevice:
            def __init__(self, location):
                self.location = location
                self.device_id = f"dev_{location}"
                self.capabilities = []
                self.status = type('S', (), {'value': 'online'})()

        devices = [MockDevice("大棚A"), MockDevice("大棚A"), MockDevice("大棚B")]
        regions = AutonomousFarmManager._group_by_region(devices)
        assert set(regions.keys()) == {"大棚A", "大棚B"}
        assert len(regions["大棚A"]) == 2
        assert len(regions["大棚B"]) == 1

    def test_should_skip_night_irrigation(self):
        """夜间灌溉跳过逻辑（NIGHT_MODE=silent）"""
        mgr = AutonomousFarmManager()
        from core.device_rule_engine import RuleDecision
        result = mgr._check_night_constraint("irrigate", night_mode="silent", hour=23)
        assert result == RuleDecision.REJECTED

    def test_night_mode_full_allows_all(self):
        """NIGHT_MODE=full 不限制"""
        mgr = AutonomousFarmManager()
        from core.device_rule_engine import RuleDecision
        result = mgr._check_night_constraint("irrigate", night_mode="full", hour=23)
        assert result is None
