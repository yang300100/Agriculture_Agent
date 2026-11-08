"""自主决策编排器 单元测试"""
import os, sys, json, pytest
from datetime import datetime
from pathlib import Path
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

    def test_build_prompt_contains_exact_actionable_device_id(self, sample_state):
        sample_state.available_devices = [{
            "device_id": "irrigation_pump_01",
            "name": "温室灌溉泵",
            "capabilities": ["irrigate"],
            "status": "online",
            "location": "大棚A",
        }]
        prompt = AutonomousFarmManager().build_decision_prompt(sample_state)
        assert "device_id=irrigation_pump_01" in prompt
        assert "不得编造 device_id" in prompt

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

    def test_validate_plan_rejects_unknown_device(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[{
            "action": "irrigate", "device_id": "fake_pump",
            "params": {"duration": 10}, "reason": "测试",
        }])
        devices = [{
            "device_id": "real_pump", "capabilities": ["irrigate"],
            "status": "online",
        }]
        validated = mgr.validate_plan(plan, available_devices=devices)
        assert validated.actions == []

    def test_validate_plan_rejects_capability_mismatch(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[{
            "action": "fertigate", "device_id": "pump1",
            "params": {"amount_kg": 2}, "reason": "测试",
        }])
        devices = [{
            "device_id": "pump1", "capabilities": ["irrigate"],
            "status": "online",
        }]
        validated = mgr.validate_plan(plan, available_devices=devices)
        assert validated.actions == []

    def test_validate_plan_accepts_matching_online_device(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[{
            "action": "irrigate", "device_id": "pump1",
            "params": {"duration": 10}, "reason": "土壤偏干",
        }])
        devices = [{
            "device_id": "pump1", "capabilities": ["irrigate"],
            "status": "online",
        }]
        validated = mgr.validate_plan(plan, available_devices=devices)
        assert len(validated.actions) == 1

    def test_select_region_devices_accepts_plot_name_alias(self, monkeypatch):
        from core import device_registry_factory
        from core import plot_manager

        class MockDevice:
            device_id = "sensor_01"
            location = "温室入口"

        monkeypatch.setattr(
            device_registry_factory,
            "load_custom_devices",
            lambda username: [{
                "device_id": "sensor_01",
                "plot_id": "hebei",
                "location": "温室入口",
            }],
        )
        monkeypatch.setattr(
            plot_manager.PlotManager,
            "get_plot",
            lambda self, plot_id: {
                "plot_id": "1", "name": "hebei",
            } if plot_id == "1" else None,
        )

        selected = AutonomousFarmManager()._select_region_devices(
            [MockDevice()], "123", "1")
        assert len(selected) == 1


class TestExecutionAndCycle:
    """执行和编排测试"""

    def test_run_cycle_returns_report_even_on_failure(self, monkeypatch):
        """即使采集失败，run_cycle 也应返回报告"""
        mgr = AutonomousFarmManager()
        monkeypatch.setattr(mgr, "_collect_weather", lambda *args, **kwargs: (None, []))
        report = mgr.run_cycle("nonexistent_user", "不存在的区域")
        assert report is not None
        assert report.cycle_id != ""
        assert report.summary != ""
        report_path = (
            Path(os.environ["DATA_STORAGE_DIR"])
            / "nonexistent_user" / "autonomous_reports"
            / f"{report.cycle_id}.json"
        )
        assert report_path.is_file()

    def test_cycle_report_has_duration(self, monkeypatch):
        mgr = AutonomousFarmManager()
        monkeypatch.setattr(mgr, "_collect_weather", lambda *args, **kwargs: (None, []))
        report = mgr.run_cycle("test_user", "test_region")
        assert report.duration_ms > 0
        report_path = (
            Path(os.environ["DATA_STORAGE_DIR"])
            / "test_user" / "autonomous_reports"
            / f"{report.cycle_id}.json"
        )
        assert report_path.is_file()
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        assert persisted["duration_ms"] == report.duration_ms

    @pytest.mark.parametrize("outcome, summary_fragment", [
        ("success", "巡检完成"),
        ("decision_unavailable", "LLM 和规则引擎均不可用"),
        ("collection_failure", "周期异常"),
    ])
    def test_cycle_always_persists_final_report_with_elapsed_duration(
            self, monkeypatch, outcome, summary_fragment):
        """哥哥，无论正常、兜底失败还是异常，报告都要完整落盘。"""
        import core.autonomous_farm_manager as manager_module

        times = iter([
            datetime(2026, 11, 5, 12, 0, 0),
            datetime(2026, 11, 5, 12, 0, 0, 250000),
        ])

        class CycleClock:
            @staticmethod
            def now():
                return next(times)

        monkeypatch.setattr(manager_module, "datetime", CycleClock)
        mgr = AutonomousFarmManager()
        username = f"cycle_report_{outcome}"
        state = FarmState(
            region="报告区", username=username,
            sensor_readings={"soil_moisture": 30},
        )

        def collect_state(*args, **kwargs):
            if outcome == "collection_failure":
                raise RuntimeError("采集失败")
            return state

        monkeypatch.setattr(mgr, "collect_farm_state", collect_state)
        monkeypatch.setattr(
            mgr, "request_decision",
            lambda prompt: DecisionPlan(region="报告区", actions=[])
            if outcome == "success" else None,
        )
        monkeypatch.setattr(mgr, "_fallback_rule_engine", lambda *args: None)

        report = mgr.run_cycle(username, "报告区")

        assert summary_fragment in report.summary
        assert report.duration_ms == 250
        report_path = (
            Path(os.environ["DATA_STORAGE_DIR"])
            / username / "autonomous_reports" / f"{report.cycle_id}.json"
        )
        assert report_path.is_file()
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        assert persisted["summary"] == report.summary
        assert persisted["duration_ms"] == 250
        assert persisted["cycle_id"] == report.cycle_id

    def test_weather_only_state_still_runs_decision(self, monkeypatch):
        mgr = AutonomousFarmManager()
        state = FarmState(
            region="大棚A", username="test",
            current_weather={"temperature": 30, "weather_desc": "晴"},
        )
        monkeypatch.setattr(
            mgr, "collect_farm_state",
            lambda *args, **kwargs: state,
        )
        monkeypatch.setattr(
            mgr, "request_decision",
            lambda prompt: DecisionPlan(
                region="大棚A", overall_assessment="天气偏热", actions=[]),
        )
        monkeypatch.setattr(mgr, "_save_report", lambda report: None)

        report = mgr.run_cycle("test", "大棚A")
        assert report.decision_plan is not None
        assert "天气偏热" in report.summary

    def test_cycle_uses_configured_max_actions(self, monkeypatch):
        from app.agent import config

        mgr = AutonomousFarmManager()
        state = FarmState(
            region="大棚A", username="test",
            current_weather={"temperature": 30, "weather_desc": "晴"},
        )
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "alert", "reason": "提醒1"},
            {"action": "alert", "reason": "提醒2"},
        ])
        captured = {}
        monkeypatch.setattr(config, "AUTO_DECISION_MAX_ACTIONS", 1)
        monkeypatch.setattr(
            mgr, "collect_farm_state",
            lambda *args, **kwargs: state,
        )
        monkeypatch.setattr(mgr, "request_decision", lambda prompt: plan)

        def _capture_plan(checked_plan, username):
            captured["actions"] = checked_plan.actions
            return []

        monkeypatch.setattr(
            mgr, "execute_plan",
            _capture_plan,
        )
        monkeypatch.setattr(mgr, "_save_report", lambda report: None)

        mgr.run_cycle("test", "大棚A")
        assert len(captured["actions"]) == 1

    def test_fallback_rule_engine_empty_state(self):
        mgr = AutonomousFarmManager()
        state = FarmState(region="test", username="test")
        plan = mgr._fallback_rule_engine(state, "test")
        # 空状态可能匹配不到规则，返回 None 或空 actions
        assert plan is None or plan.actions == []

    def test_fallback_rule_uses_short_sensor_name_and_capability(
            self, monkeypatch):
        from core import device_rule_engine

        captured = {}

        class FakeRuleEngine:
            def __init__(self, username):
                pass

            def find_matching_rules(self, context):
                captured["sensor_data"] = context["sensor_data"]
                return [{
                    "id": "dry_soil",
                    "name": "土壤过干灌溉",
                    "action": {
                        "device_id": "irrigation_pump_01",
                        "command": "start",
                        "params": {"duration": 20},
                    },
                }]

            def evaluate_action(self, rule, params, context):
                return (
                    device_rule_engine.RuleDecision.AUTO_EXECUTE,
                    "规则通过",
                    params,
                )

            def _infer_capability(self, action):
                return "irrigate"

        monkeypatch.setattr(device_rule_engine, "RuleEngine", FakeRuleEngine)
        state = FarmState(
            region="大棚A", username="test",
            sensor_readings={"sensor_01.soil_moisture": 22.0},
        )

        mgr = AutonomousFarmManager()
        monkeypatch.setattr(mgr, "_check_night_constraint",
                            lambda *args, **kwargs: None)
        plan = mgr._fallback_rule_engine(state, "test")
        assert captured["sensor_data"]["soil_moisture"] == 22.0
        assert plan.actions[0]["action"] == "irrigate"
        assert plan.actions[0]["device_id"] == "irrigation_pump_01"

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

    @pytest.mark.parametrize(
        "command,mode,constraints,initial_status,expected_status,pending_count,night_blocked",
        [
            ("stop", "auto", {}, "running", "standby", 0, False),
            ("stop", "auto", {}, "running", "standby", 0, True),
            ("set_param", "auto", {}, "standby", "standby", 0, False),
            ("start", "notify", {}, "standby", "standby", 0, False),
            ("start", "confirm", {}, "standby", "standby", 1, False),
            ("start", "auto", {"max_duration_per_use": 1}, "standby", "standby", 1, False),
        ],
        ids=["stop", "night-stop", "set-param", "notify", "confirm", "constraint-confirm"],
    )
    def test_fallback_rule_keeps_command_and_execution_mode(
            self, monkeypatch, command, mode, constraints, initial_status,
            expected_status, pending_count, night_blocked):
        """哥哥，模型离线时也要按原规则停止、改参数、提醒或等待确认。"""
        import uuid
        from core.device_executor import DeviceExecutor
        from core.device_registry_factory import load_custom_devices, save_custom_devices
        from core.device_rule_engine import RuleDecision, RuleEngine
        from devices.registry import DeviceDriverRegistry

        username = f"fallback_{uuid.uuid4().hex}"
        monkeypatch.setenv("AUTONOMY_LEVEL", "medium")
        save_custom_devices(username, [{
            "device_id": "fallback_pump", "name": "兜底测试水泵",
            "driver": "simulator", "capabilities": ["irrigate"],
            "initial_state": {"power": True, "status": initial_status, "flow_rate": 3},
        }])
        params = {"flow_rate": 5} if command == "set_param" else {"duration": 2}
        rule_id = RuleEngine(username).add_rule({
            "name": "离线兜底规则", "enabled": True, "execution_mode": mode,
            "trigger": {"conditions": [{
                "type": "sensor", "field": "soil_moisture", "op": ">", "value": 70,
            }]},
            "action": {"device_id": "fallback_pump", "capability": "irrigate",
                       "command": command, "params": params},
            "constraints": constraints,
        })
        state = FarmState(
            region="测试区", username=username,
            sensor_readings={"sensor.soil_moisture": 90},
            available_devices=[{
                "device_id": "fallback_pump", "capabilities": ["irrigate"], "status": "online",
            }],
        )
        mgr = AutonomousFarmManager()
        monkeypatch.setattr(
            mgr, "_check_night_constraint",
            lambda *args: RuleDecision.REJECTED if night_blocked else None,
        )

        plan = mgr._fallback_rule_engine(state, username)
        assert plan is not None
        plan = mgr.validate_plan(plan, available_devices=state.available_devices)
        results = mgr.execute_plan(plan, username)

        persisted = load_custom_devices(username)[0]["initial_state"]
        assert persisted["status"] == expected_status
        if command == "set_param":
            assert persisted["flow_rate"] == 5
        executor = DeviceExecutor(DeviceDriverRegistry(), username)
        pending = executor.list_pending()
        assert len(pending) == pending_count
        if pending_count:
            assert pending[0]["command"] == "start"
            assert str(pending[0]["rule_id"]) == str(rule_id)
        elif mode == "notify":
            assert results[0].action == "alert"
            assert "离线兜底规则" in results[0].message
        else:
            assert results[0].success is True

    def test_offline_cycle_fallback_keeps_sensor_readings_in_target_plot(self, monkeypatch):
        """哥哥，完整离线巡检只给缺水地块浇水，湿润地块要保持待机。"""
        import uuid
        from core.device_registry_factory import load_custom_devices, save_custom_devices
        from core.device_rule_engine import RuleEngine

        username = f"fallback_scope_{uuid.uuid4().hex}"
        monkeypatch.setenv("AUTONOMY_LEVEL", "medium")
        configs = []
        for suffix, plot_id in [("a", 1), ("b", 2)]:
            configs.extend([
                {"device_id": f"pump_{suffix}", "name": f"水泵{suffix}",
                 "driver": "simulator", "capabilities": ["irrigate"], "plot_id": plot_id,
                 "initial_state": {"power": True, "status": "standby"}},
                {"device_id": f"sensor_{suffix}", "name": f"土壤传感器{suffix}",
                 "driver": "simulator", "capabilities": ["read_sensor"], "plot_id": plot_id,
                 "initial_state": {"soil_moisture": 22 if suffix == "a" else 70}},
            ])
        save_custom_devices(username, configs)
        engine = RuleEngine(username)
        for suffix in ["a", "b"]:
            engine.add_rule({
                "name": f"地块{suffix}缺水灌溉", "enabled": True,
                "trigger": {"conditions": [{
                    "type": "sensor", "field": "soil_moisture", "op": "<", "value": 30,
                }]},
                "action": {"device_id": f"pump_{suffix}", "capability": "irrigate",
                           "command": "start", "params": {"duration": 2}},
            })
        state = FarmState(
            region="测试区", username=username,
            sensor_readings={"sensor_a.soil_moisture": 22, "sensor_b.soil_moisture": 70},
            available_devices=[
                {"device_id": f"pump_{suffix}", "name": f"水泵{suffix}",
                 "capabilities": ["irrigate"], "status": "online"}
                for suffix in ["a", "b"]
            ],
        )
        mgr = AutonomousFarmManager()
        monkeypatch.setattr(mgr, "collect_farm_state", lambda *args, **kwargs: state)
        monkeypatch.setattr(mgr, "request_decision", lambda prompt: None)
        monkeypatch.setattr(mgr, "_check_night_constraint", lambda *args: None)

        report = mgr.run_cycle(username, "测试区")

        states = {device["device_id"]: device["initial_state"]
                  for device in load_custom_devices(username)}
        assert report.fallback_used is True
        assert states["pump_a"]["status"] == "running"
        assert states["pump_b"]["status"] == "standby"
        assert [result.device_id for result in report.execution_results] == ["pump_a"]
