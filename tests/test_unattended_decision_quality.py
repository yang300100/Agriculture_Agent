from datetime import datetime, timedelta
import math

from core.autonomous_farm_manager import AutonomousFarmManager, DecisionPlan


def test_observations_reject_stale_errors_units_and_nonfinite():
    from core.observation_quality import validate_sensor_state
    now = datetime.now()
    valid, problems = validate_sensor_state({"soil_moisture": 25, "temperature": math.nan,
        "_observed_at": now.isoformat(), "_quality": "good", "_units": {"soil_moisture": "%", "temperature": "°C"}}, now=now)
    assert valid == {"soil_moisture": 25}
    assert problems
    for extra in ({"_quality": "error"}, {"_error": "offline"},
                  {"_observed_at": (now-timedelta(seconds=301)).isoformat()},
                  {"_units": {"soil_moisture": "F"}}):
        state = {"soil_moisture": 25, "_observed_at": now.isoformat(), "_quality": "good", "_units": {"soil_moisture": "%"}}
        state.update(extra)
        assert validate_sensor_state(state, now=now)[0] == {}


def test_opposite_temperature_actions_are_filtered():
    plan = DecisionPlan(region="1", actions=[{"action": "heat", "device_id": "h"}, {"action": "cool", "device_id": "c"}])
    assert [a["action"] for a in AutonomousFarmManager().validate_plan(plan).actions] == ["heat"]


def test_crop_info_uses_only_explicit_plot(monkeypatch):
    from core.plot_manager import PlotManager
    from core.planting_tracker import PlantingTracker
    monkeypatch.setattr(PlotManager, "get_plot", lambda self, plot_id: {"plot_id": plot_id, "crop": "小麦"})
    monkeypatch.setattr(PlantingTracker, "get_progress_cards", lambda self, limit=None: [
        {"plot_id": "2", "crop": "番茄", "status": "进行中"},
        {"plot_id": None, "crop": "玉米", "status": "进行中"}])
    crops, risks = AutonomousFarmManager()._collect_crop_info("test", "1")
    assert [c["crop"] for c in crops] == ["小麦"]
    assert risks == []


def test_deferred_action_and_follow_up_are_persisted_without_execution(monkeypatch):
    import sys, types
    from core import device_registry_factory
    queued = []
    monkeypatch.setitem(sys.modules, "core.autonomy_runtime", types.SimpleNamespace(
        schedule_action=lambda user, region, action, due_at=None: queued.append((user, region, action, due_at)),
        raise_alert=lambda *a, **k: None, queue_effect_check=lambda *a: None))
    class Loop:
        def run_until_complete(self, coroutine):
            coroutine.close()
            return []
    class Registry:
        async def discover_all(self): return []
    monkeypatch.setattr(device_registry_factory, "setup_registry", lambda user: (Registry(), Loop()))
    monkeypatch.setattr(device_registry_factory, "close_registry", lambda *a: None)
    plan = DecisionPlan(region="1", actions=[{"action":"irrigate", "device_id":"pump", "urgency":"today"}], follow_up="检查土壤")
    result = AutonomousFarmManager().execute_plan(plan, "u")
    assert {item[2]["action"] for item in queued} == {"irrigate", "review"}
    assert all(item[3] is None for item in queued)
    assert not result[0].success


def test_failed_dependency_is_not_dispatched(monkeypatch):
    import sys, types
    from core import device_registry_factory
    alerts=[]
    monkeypatch.setitem(sys.modules, "core.autonomy_runtime", types.SimpleNamespace(
        schedule_action=lambda *a, **k: None, raise_alert=lambda *a, **k: alerts.append(a),
        queue_effect_check=lambda *a: None))
    class Loop:
        def run_until_complete(self, coroutine):
            coroutine.close()
            return []
    class Registry:
        async def discover_all(self): return []
    monkeypatch.setattr(device_registry_factory, "setup_registry", lambda user: (Registry(), Loop()))
    monkeypatch.setattr(device_registry_factory, "close_registry", lambda *a: None)
    plan=DecisionPlan(region="1", actions=[{"id":"first", "action":"heat", "device_id":"missing", "urgency":"immediate"},
        {"id":"second", "depends_on":"first", "action":"alert", "reason":"should not send", "urgency":"immediate"}])
    result=AutonomousFarmManager().execute_plan(plan,"u")
    assert len(result)==2 and not result[1].success and not alerts


def test_low_confidence_does_not_confirm_or_recommend_actions(monkeypatch):
    from app.agent.agents.crop_monitor_agent import CropMonitorAgent
    from core import model_registry_factory, model_executor
    from types import SimpleNamespace
    monkeypatch.setattr(model_registry_factory, "resolve_inference_model", lambda *a: (None, "x"))
    monkeypatch.setattr(model_executor.ModelExecutor, "infer_sync", lambda *a: SimpleNamespace(
        success=True, predictions=[SimpleNamespace(class_name="disease", confidence=0.2)], inference_time_ms=1))
    agent=CropMonitorAgent()
    monkeypatch.setattr(agent, "_invoke_llm_structured", lambda _: {"health_assessment": {"disease_presence": "confirmed"}, "recommended_actions": [{"action":"irrigate"}]})
    result=agent._call_dl_model("YQ==", "image/jpeg")
    assert result["health_assessment"]["disease_presence"] != "confirmed"
    assert result["recommended_actions"] == []


def test_stale_camera_image_is_not_analyzed(monkeypatch):
    from types import SimpleNamespace
    from app.agent.agents.crop_monitor_agent import CropMonitorAgent
    calls=[]
    monkeypatch.setattr(CropMonitorAgent, "analyze_image", lambda *a, **k: calls.append(1))
    class Loop:
        def run_until_complete(self, coroutine):
            coroutine.close()
            return SimpleNamespace(success=True, raw_response={"image_bytes": b"jpg", "metadata": {"timestamp": (datetime.now()-timedelta(hours=1)).isoformat()}})
    class Registry:
        async def execute(self, *args): pass
    views=AutonomousFarmManager()._capture_and_analyze(Registry(), Loop(), [SimpleNamespace(device_id="camera", location="x")], "u", "1", [{"crop":"wheat"}])
    assert not calls and views[0].vision_analysis is None and views[0].error


def test_sensor_collection_keeps_quality_and_rejects_missing_timestamp():
    from types import SimpleNamespace
    class Loop:
        def run_until_complete(self, coroutine):
            coroutine.close()
            return {"soil_moisture": 20, "_quality": "good"}
    class Registry:
        async def read_state(self, device_id): pass
    quality={}
    values=AutonomousFarmManager()._collect_sensors(Registry(), Loop(), [SimpleNamespace(device_id="s")], quality)
    assert not values and quality["s"]["issues"]


def test_plot_risk_requires_matching_plot_and_fresh_timestamp(monkeypatch, tmp_path):
    import json
    import core.autonomous_farm_manager as module
    from core.plot_manager import PlotManager
    from core.planting_tracker import PlantingTracker
    monkeypatch.setattr(module, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(PlotManager, "get_plot", lambda self, pid: {"plot_id": pid, "crop": "wheat"})
    monkeypatch.setattr(PlantingTracker, "get_progress_cards", lambda self, limit=None: [])
    path=tmp_path/"u"/"disease_risks"/"1.json"
    path.parent.mkdir(parents=True)
    manager=AutonomousFarmManager()
    for plot, when, expected in [("1", datetime.now(), [{"name":"rust"}]), ("2", datetime.now(), []), ("1", datetime.now()-timedelta(days=2), [])]:
        path.write_text(json.dumps({"plot_id":plot,"updated":when.isoformat(),"quality":"good","risks":[{"name":"rust"}]}),encoding="utf-8")
        assert manager._collect_crop_info("u", "1")[1] == expected


def test_device_plot_binding_takes_precedence_over_shared_location(monkeypatch):
    from types import SimpleNamespace
    from core import device_registry_factory, plot_manager
    monkeypatch.setattr(device_registry_factory, "load_custom_devices", lambda user: [{"device_id":"s", "plot_id":"2", "location":"shared"}])
    monkeypatch.setattr(plot_manager.PlotManager, "get_plot", lambda self, pid: {"plot_id":pid, "name":"shared"})
    selected=AutonomousFarmManager()._select_region_devices([SimpleNamespace(device_id="s", location="shared")], "u", "1")
    assert selected == []


def test_visual_diagnosis_rejects_other_crop(monkeypatch):
    from app.agent.agents.crop_monitor_agent import CropMonitorAgent
    agent=CropMonitorAgent()
    monkeypatch.setattr(agent, "_call_dl_model", lambda *a: {"crop_type":"苹果", "health_assessment":{"disease_presence":"confirmed"}, "recommended_actions":[{"action":"alert"}], "dl_predictions":[{"class_name":"苹果-黑星病", "confidence":0.99}]})
    result=agent.analyze_image("YQ==", user_context={"crop":"番茄", "plot_id":"1"})["analysis"]
    assert result["health_assessment"]["disease_presence"] != "confirmed"
    assert result["recommended_actions"] == []


def test_invalid_heat_does_not_suppress_valid_cooling():
    plan=DecisionPlan(region="1", actions=[{"action":"heat", "device_id":"missing"}, {"action":"cool", "device_id":"c"}])
    devices=[{"device_id":"c", "capabilities":["cool"], "status":"online"}]
    assert [a["action"] for a in AutonomousFarmManager().validate_plan(plan, available_devices=devices).actions] == ["cool"]


def test_acknowledged_start_preserves_execution_id_and_defers_dependency(monkeypatch):
    import sys, types
    from core import device_registry_factory, device_executor, device_rule_engine
    from app.agent import config
    queued, commands = [], []
    monkeypatch.setitem(sys.modules, "core.autonomy_runtime", types.SimpleNamespace(
        schedule_action=lambda user, region, action, due_at=None: queued.append((action.copy(), due_at)),
        raise_alert=lambda *a, **k: None, queue_effect_check=lambda *a: None))
    class Loop:
        def run_until_complete(self, coro):
            coro.close()
            return [types.SimpleNamespace(device_id=did, capabilities=[types.SimpleNamespace(value=cap)], status=types.SimpleNamespace(value="online")) for did, cap in [("pump", "irrigate"), ("fert", "fertigate")]]
    class Registry:
        async def discover_all(self): pass
    class Engine:
        def __init__(self, username): pass
        def context_for_device(self, device_id, context): return context
        def evaluate_action(self, rule, params, context): return device_rule_engine.RuleDecision.AUTO_EXECUTE, "ok", params
        def record_execution(self, *args): pass
    class Executor:
        def __init__(self, *args, **kwargs): pass
        def execute_sync(self, device_id, command, **kwargs):
            commands.append((device_id, command.execution_id))
            return {"success": True, "result": types.SimpleNamespace(message="started", acknowledged=True, execution_id=command.execution_id)}
    monkeypatch.setattr(device_registry_factory, "setup_registry", lambda user: (Registry(), Loop()))
    monkeypatch.setattr(device_registry_factory, "close_registry", lambda *a: None)
    monkeypatch.setattr(device_rule_engine, "RuleEngine", Engine)
    monkeypatch.setattr(device_executor, "DeviceExecutor", Executor)
    monkeypatch.setattr(config, "get_autonomy_level", lambda user: "high")
    mgr=AutonomousFarmManager()
    monkeypatch.setattr(mgr, "_check_night_constraint", lambda *a: None)
    plan=DecisionPlan(region="1", actions=[
        {"id":"water", "action":"irrigate", "device_id":"pump", "urgency":"immediate", "execution_id":"run-123", "params":{"duration":10}},
        {"id":"feed", "depends_on":"water", "action":"fertigate", "device_id":"fert", "urgency":"immediate", "params":{"duration":10}}])
    results=mgr.execute_plan(plan,"u")
    assert commands == [("pump", "run-123")]
    assert not results[1].success
    assert len(queued) == 1 and isinstance(queued[0][1], datetime)
    assert queued[0][0]["depends_on"] == "water"
    assert queued[0][0]["dependency_runs"] == [{"device_id": "pump", "execution_id": "run-123"}]


def test_unavailable_disease_risk_is_distinct_from_evaluated_empty(monkeypatch, tmp_path):
    import json
    import core.autonomous_farm_manager as module
    from core.plot_manager import PlotManager
    from core.planting_tracker import PlantingTracker
    monkeypatch.setattr(module, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(PlotManager, "get_plot", lambda self, pid: {"plot_id":pid, "crop":"wheat"})
    monkeypatch.setattr(PlantingTracker, "get_progress_cards", lambda self, limit=None: [])
    path=tmp_path/"u"/"disease_risks"/"1.json"
    path.parent.mkdir(parents=True)
    manager=AutonomousFarmManager()
    for quality in ("unavailable", "good"):
        path.write_text(json.dumps({"plot_id":"1", "updated":datetime.now().isoformat(), "quality":quality, "risks":[]}), encoding="utf-8")
        status={}
        _, risks=manager._collect_crop_info("u", "1", status)
        assert risks == [] and status["quality"] == quality


def test_dependency_never_overrides_later_requested_schedule(monkeypatch):
    import sys, types
    from core import device_registry_factory, device_executor, device_rule_engine
    from app.agent import config
    scheduled=[]
    monkeypatch.setitem(sys.modules, "core.autonomy_runtime", types.SimpleNamespace(schedule_action=lambda user, region, action, due_at=None: scheduled.append(due_at), raise_alert=lambda *a, **k: None, queue_effect_check=lambda *a: None))
    class Loop:
        def run_until_complete(self, coro):
            coro.close()
            return [types.SimpleNamespace(device_id="pump", capabilities=[types.SimpleNamespace(value="irrigate")], status=types.SimpleNamespace(value="online"))]
    class Registry:
        async def discover_all(self): pass
    class Engine:
        def __init__(self, username): pass
        def context_for_device(self, device_id, context): return context
        def evaluate_action(self, rule, params, context): return device_rule_engine.RuleDecision.AUTO_EXECUTE, "ok", params
        def record_execution(self, *args): pass
    class Executor:
        def __init__(self, *a, **k): pass
        def execute_sync(self, device_id, command, **kwargs): return {"success":True, "result":types.SimpleNamespace(message="ok", acknowledged=True, execution_id=command.execution_id)}
    monkeypatch.setattr(device_registry_factory,"setup_registry",lambda user:(Registry(),Loop()))
    monkeypatch.setattr(device_registry_factory,"close_registry",lambda *a:None)
    monkeypatch.setattr(device_rule_engine,"RuleEngine",Engine)
    monkeypatch.setattr(device_executor,"DeviceExecutor",Executor)
    monkeypatch.setattr(config,"get_autonomy_level",lambda user:"high")
    manager=AutonomousFarmManager()
    monkeypatch.setattr(manager,"_check_night_constraint",lambda *a:None)
    later=datetime.now()+timedelta(days=1)
    manager.execute_plan(DecisionPlan(region="1",actions=[
        {"id":"water","action":"irrigate","device_id":"pump","urgency":"immediate","params":{"duration":1}},
        {"action":"fertigate","device_id":"fert","urgency":"today","depends_on":"water","scheduled_at":later.isoformat()}]),"u")
    assert scheduled == [later]


def test_天气仅使用真实观测时间且不伪造预报质量(monkeypatch):
    from types import SimpleNamespace
    from core import weather_service
    observed = datetime.now()
    records = []
    class Service:
        provider = 'qweather'
        def get_grid_weather(self, lon, lat):
            records.append((lon, lat))
            return {'temp': 22, 'humidity': 50, 'precip': 0, 'obsTime': observed.isoformat()}
        def get_forecast_by_coords(self, lat, lon, region, days):
            return [{'date': '2099-01-01', 'temperature_high': 30}]
    monkeypatch.setattr(weather_service, 'WeatherService', Service)
    manager = AutonomousFarmManager()
    current, forecast = manager._collect_weather('1', lat=39.9, lon=116.4)
    assert current['temperature'] == 22 and current['_observed_at'] == observed.isoformat()
    assert forecast == [] and records == [(116.4, 39.9)]
    observed -= timedelta(hours=2)
    current, forecast = manager._collect_weather('1', lat=39.9, lon=116.4)
    assert current is None and forecast == []


def test_缺少地块坐标不回退地区查询(monkeypatch):
    from core import weather_service
    calls = []
    class Service:
        def __init__(self): calls.append(True)
    monkeypatch.setattr(weather_service, 'WeatherService', Service)
    assert AutonomousFarmManager()._collect_weather('1') == (None, [])
    assert not calls


def test_持续天气不复用全局共享历史(monkeypatch):
    from core import weather_history
    monkeypatch.setattr(weather_history, 'check_persistence', lambda: [{'type': '其他地区暴雨'}])
    assert AutonomousFarmManager()._collect_persistence() == []


def test_采集天气使用当前用户真实地块坐标而非调用参数(monkeypatch):
    from core import device_registry_factory, plot_manager
    seen=[]
    class Loop:
        def run_until_complete(self, coroutine):
            coroutine.close()
            return []
    class Registry:
        async def discover_all(self): pass
    monkeypatch.setattr(device_registry_factory,'setup_registry',lambda user:(Registry(),Loop()))
    monkeypatch.setattr(device_registry_factory,'close_registry',lambda *args:None)
    monkeypatch.setattr(plot_manager.PlotManager,'get_plot',lambda self, region: {'plot_id':region, 'lat':39.1, 'lon':114.1} if region == '1' else None)
    mgr=AutonomousFarmManager()
    monkeypatch.setattr(mgr,'_collect_crop_info',lambda *args: ([], []))
    monkeypatch.setattr(mgr,'_collect_recent_actions',lambda *args: [])
    monkeypatch.setattr(mgr,'_collect_weather',lambda region, lat=None, lon=None: (seen.append((lat,lon)) or None, []))
    mgr.collect_farm_state('owner','1',plot_lat=10,plot_lon=20)
    mgr.collect_farm_state('owner','unknown',plot_lat=10,plot_lon=20)
    assert seen == [(39.1,114.1),(None,None)]


def test_巡检天气按用户地块隔离(monkeypatch):
    from core import device_registry_factory, plot_manager
    calls=[]
    class Loop:
        def run_until_complete(self, coroutine):
            coroutine.close()
            return []
    class Registry:
        async def discover_all(self): pass
    monkeypatch.setattr(device_registry_factory, 'setup_registry', lambda user:(Registry(),Loop()))
    monkeypatch.setattr(device_registry_factory, 'close_registry', lambda *args:None)
    monkeypatch.setattr(plot_manager.PlotManager,'get_plot',lambda self, region: {'plot_id': region, 'lat':39.0 if self.username == 'a' else 29.0, 'lon':114.0})
    manager=AutonomousFarmManager()
    monkeypatch.setattr(manager,'_collect_crop_info',lambda *args:([], []))
    monkeypatch.setattr(manager,'_collect_recent_actions',lambda *args:[])
    monkeypatch.setattr(manager,'_collect_weather',lambda region,lat=None,lon=None:(calls.append((lat,lon)) or None, []))
    manager.collect_farm_state('a','1')
    manager.collect_farm_state('b','1')
    assert calls == [(39.0,114.0),(29.0,114.0)]
