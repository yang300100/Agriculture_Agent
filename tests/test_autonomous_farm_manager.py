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
        mgr = AutonomousFarmManager()
        regions = mgr._group_by_region(devices)
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


class TestDecisionEngine:
    """LLM 决策引擎测试"""

    @pytest.fixture
    def sample_state(self):
        return FarmState(
            region="大棚A", username="test",
            timestamp="2026-06-19T14:00:00",
            camera_views=[
                CameraView(device_id="cam01", location="大棚A",
                          vision_analysis={
                              "crop_type": "番茄", "growth_stage": "fruiting",
                              "health_assessment": {"overall": "fair",
                                  "water_status": "drought-stressed"},
                              "recommended_actions": [
                                  {"action": "irrigate", "urgency": "today",
                                   "detail": "土壤偏干需要灌溉"}
                              ],
                          }),
            ],
            sensor_readings={"sensor01.soil_moisture": 28.5},
            current_weather={"temperature": 28, "humidity": 45,
                           "weather_desc": "晴"},
            weather_forecast=[
                {"date": "2026-06-20", "weather_desc": "晴",
                 "temperature_high": 30, "temperature_low": 20, "humidity": 40},
            ],
            active_crops=[
                {"crop": "番茄", "stage": "结果期", "stage_number": 4,
                 "total_stages": 6, "status": "进行中", "progress_percent": 65},
            ],
        )

    def test_build_prompt_contains_key_fields(self, sample_state):
        mgr = AutonomousFarmManager()
        prompt = mgr.build_decision_prompt(sample_state)
        assert "大棚A" in prompt
        assert "番茄" in prompt
        assert "28.5" in prompt
        assert "soil_moisture" in prompt

    def test_parse_valid_json(self):
        mgr = AutonomousFarmManager()
        content = '```json\n{"region":"大棚A","overall_assessment":"测试","actions":[],"follow_up":""}\n```'
        result = mgr._parse_decision(content)
        assert result["region"] == "大棚A"
        assert result["overall_assessment"] == "测试"

    def test_parse_json_without_code_block(self):
        mgr = AutonomousFarmManager()
        content = '{"region":"大棚A","overall_assessment":"OK","actions":[],"follow_up":""}'
        result = mgr._parse_decision(content)
        assert result["region"] == "大棚A"

    def test_parse_truncated_json_recovers(self):
        mgr = AutonomousFarmManager()
        content = '{"region":"大棚A","overall_assessment":"一切正常","actions":[{"action":"irrigate","params":{"duration":25'
        result = mgr._parse_decision(content)
        assert result is not None
        assert result["region"] == "大棚A"

    def test_parse_completely_invalid_returns_none(self):
        mgr = AutonomousFarmManager()
        content = "这是一段不是JSON的回复文本"
        result = mgr._parse_decision(content)
        assert result is None

    def test_validate_plan_accepts_valid_actions(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "irrigate", "device_id": "pump1", "params": {"duration": 30}, "urgency": "today", "reason": "土壤缺水"},
            {"action": "alert", "urgency": "this_week", "reason": "需注意病害"},
        ])
        plan = mgr.validate_plan(plan, available_capabilities={"irrigate"})
        assert len(plan.actions) == 2

    def test_validate_plan_rejects_unknown_action(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "fly_to_moon", "params": {}, "urgency": "today", "reason": "?"},
        ])
        plan = mgr.validate_plan(plan, available_capabilities=set())
        assert len(plan.actions) == 0

    def test_validate_plan_clips_exceeded_params(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "irrigate", "device_id": "pump1", "params": {"duration": 999}, "urgency": "today", "reason": "测试"},
        ])
        plan = mgr.validate_plan(plan, available_capabilities={"irrigate"})
        assert plan.actions[0]["params"]["duration"] == 120

    def test_validate_plan_limits_max_actions(self):
        mgr = AutonomousFarmManager()
        actions = [
            {"action": "irrigate", "device_id": f"pump{i}", "params": {"duration": 10}, "urgency": "today", "reason": f"测试{i}"}
            for i in range(10)
        ]
        plan = DecisionPlan(region="大棚A", actions=actions)
        plan = mgr.validate_plan(plan, available_capabilities={"irrigate"}, max_actions=5)
        assert len(plan.actions) == 5

    def test_validate_plan_dedup_same_device(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "irrigate", "device_id": "pump1", "params": {"duration": 10}, "urgency": "today", "reason": "a"},
            {"action": "irrigate", "device_id": "pump1", "params": {"duration": 20}, "urgency": "today", "reason": "b"},
        ])
        plan = mgr.validate_plan(plan, available_capabilities={"irrigate"})
        assert len(plan.actions) == 1


class TestExecutionAndCycle:
    """执行和编排测试"""

    def test_run_cycle_returns_report_even_on_failure(self):
        """即使采集失败，run_cycle 也应返回报告"""
        mgr = AutonomousFarmManager()
        report = mgr.run_cycle("nonexistent_user", "不存在的区域")
        assert report is not None
        assert report.cycle_id != ""
        assert report.summary != ""

    def test_cycle_report_has_duration(self):
        mgr = AutonomousFarmManager()
        report = mgr.run_cycle("test_user", "test_region")
        assert report.duration_ms > 0

    def test_fallback_rule_engine_empty_state(self):
        mgr = AutonomousFarmManager()
        state = FarmState(region="test", username="test")
        plan = mgr._fallback_rule_engine(state, "test")
        # 空状态可能匹配不到规则，返回 None 或空 actions
        assert plan is None or plan.actions == []

    def test_summarize_successful_report(self):
        mgr = AutonomousFarmManager()
        state = FarmState(region="大棚A", username="test")
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "irrigate", "params": {"duration": 25}, "urgency": "today", "reason": "缺水"},
        ])
        results = [
            ActionResult(action="irrigate", device_id="pump1", success=True,
                        message="执行成功", executed_params={"duration": 25}),
        ]
        report = CycleReport(
            cycle_id="c1", username="test", region="大棚A",
            timestamp="2026-06-19T14:00:00",
            farm_state=state, decision_plan=plan,
            execution_results=results, summary="",
        )
        summary = mgr._summarize(report)
        assert "irrigate" in summary or "大棚A" in summary
