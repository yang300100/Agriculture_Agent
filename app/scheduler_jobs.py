"""APScheduler 定时任务 — 提醒检查 / 天气预警 / 病害风险评估"""

import logging
import os
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)


def check_reminders_job():
    """每 5 分钟检查到期提醒并发送 SMS"""
    try:
        from core.reminder_scheduler import ReminderScheduler
        import os, json

        sched = ReminderScheduler()
        # 从 user_profile 读取手机号
        phone = ""
        profile_path = os.path.join("data", "user_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, encoding="utf-8") as f:
                profile = json.load(f)
                phone = profile.get("user_phone", "")

        fired = sched.check_and_fire(phone=phone)
        if fired:
            logger.info("提醒调度: 触发 %d 条", len(fired))
    except Exception as e:
        logger.warning("提醒检查失败: %s", e)


def check_weather_job():
    """每 30 分钟：天气预警 + 历史记录 + 持续异常检测 + SMS 推送"""
    try:
        import os, json
        from core.weather_alerts import check_weather_alert_for_region
        from core.weather_service import WeatherService
        from core.weather_history import record_today, check_persistence
        from core.sms_service import SMSService
        from core.planting_tracker import PlantingTracker

        region, phone = _get_profile()
        ws = WeatherService()

        # 1. 天气预警
        result = check_weather_alert_for_region(region)
        if result:
            os.makedirs("data", exist_ok=True)
            with open(os.path.join("data", "weather_alerts_cache.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if result.get("has_alert"):
                logger.info("天气预警: %s %d条", region, result.get("count", 0))

        # 2. 记录今日天气（持续异常检测用）
        current = ws.get_current_weather(region)
        if current:
            record_today({
                "rain": any(kw in current.weather_desc for kw in ["雨", "rain", "shower", "drizzle"]),
                "temp_high": current.temperature_high,
                "temp_low": current.temperature_low,
                "humidity": current.humidity,
                "desc": current.weather_desc,
            })

            # 3. 持续异常检测
            tracker = PlantingTracker()
            active = tracker.get_progress()
            active_crops = list(set(p.crop for p in active if p.status == "进行中"))
            persistence = check_persistence(active_crops)
            if persistence:
                os.makedirs("data", exist_ok=True)
                with open(os.path.join("data", "weather_persistence.json"), "w", encoding="utf-8") as f:
                    json.dump({"updated": datetime.now().isoformat(), "alerts": persistence},
                              f, ensure_ascii=False, indent=2)

                # 4. SMS 推送
                if phone:
                    sms = SMSService()
                    if sms.is_configured:
                        for alert in persistence:
                            msg = f"【{alert['type']}】已持续{alert['days']}天（{alert['period']}）。{alert['advice'][:200]}"
                            sms.send_sms([phone], [alert['type'], str(alert['days']), msg[:100], ""])
                            logger.info("持续天气 SMS 已发送: %s", alert['type'])
    except Exception as e:
        logger.warning("天气综合检查失败: %s", e)


def _get_profile():
    """读取用户档案中的地区+手机号"""
    import os, json
    path = os.path.join("data", "user_profile.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                p = json.load(f)
                return p.get("user_region", "北京") or "北京", p.get("user_phone", "")
        except Exception:
            pass
    return "北京", ""


def check_disease_job():
    """每 6 小时评估病虫害风险（使用量化阈值引擎）"""
    try:
        import os, json
        from core.weather_service import WeatherService
        from core.disease_risk import assess_all_active_crops

        # 获取当前天气
        ws = WeatherService()
        current = ws.get_current_weather("北京")
        wdata = {
            "temperature": current.temperature if current else 20,
            "humidity": current.humidity if current else 60,
            "rain_24h": getattr(current, "precipitation", 0) or 0,
            "forecast_rain": False,
        }
        # 检查未来预报是否有雨
        try:
            forecast = ws.get_forecast("北京", 2)
            for w in forecast:
                if any(kw in w.weather_desc for kw in ["雨", "rain", "shower"]):
                    wdata["forecast_rain"] = True
                    break
        except Exception:
            pass

        risks = assess_all_active_crops(wdata)
        if risks:
            os.makedirs("data", exist_ok=True)
            with open(os.path.join("data", "disease_risks.json"), "w", encoding="utf-8") as f:
                json.dump({"updated": datetime.now().isoformat(), "risks": risks}, f, ensure_ascii=False)
            logger.info("病害风险: %d条", len(risks))
    except Exception as e:
        logger.warning("病害检查失败: %s", e)


def check_device_rules_job():
    """每 5 分钟检查自动规则并触发设备操作"""
    try:
        import os
        import asyncio as _asyncio
        from core.device_rule_engine import RuleEngine
        from devices.registry import DeviceDriverRegistry
        from devices.simulator_driver import SimulatorDriver
        from devices.base import DeviceCommand
        from core.device_executor import DeviceExecutor

        # 获取当前传感器数据
        sim = SimulatorDriver(simulated_latency_ms=50)
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(sim.connect())
            sensor_data = loop.run_until_complete(sim.read_state("virtual_soil_sensor_01"))

            # 遍历所有有规则的用户
            data_dir = os.path.join("data")
            usernames = ["default"]
            if os.path.exists(data_dir):
                for d in os.listdir(data_dir):
                    user_path = os.path.join(data_dir, d)
                    if os.path.isdir(user_path) and os.path.exists(os.path.join(user_path, "device_rules.json")):
                        usernames.append(d)

            for username in set(usernames):
                engine = RuleEngine(username=username)
                context = {"sensor_data": sensor_data, "weather": {}}
                matched = engine.find_matching_rules(context)

                if not matched:
                    continue

                registry = DeviceDriverRegistry()
                registry.register("simulator", sim)
                loop.run_until_complete(registry.discover_all())
                executor = DeviceExecutor(registry, username=username)

                for rule in matched:
                    action = rule.get("action", {})
                    proposed = action.get("params", {})
                    decision, reason, final_params = engine.evaluate_action(
                        rule, proposed, {"device_id": action.get("device_id", "")})

                    if decision == "auto_execute":
                        cmd = DeviceCommand(
                            command=action.get("command", "start"),
                            params=final_params,
                        )
                        result = executor.execute_sync(
                            action.get("device_id"), cmd,
                            trigger="rule", rule_id=rule["id"], decision=decision)

                        if result["success"]:
                            engine.record_execution(action.get("device_id"), final_params)
                            logger.info("自动规则触发: %s → %s", rule.get("name", "未命名"), action.get("device_id"))
        finally:
            loop.close()
            try:
                _asyncio.set_event_loop(None)
            except Exception:
                pass
    except Exception as e:
        logger.warning("设备规则轮询失败: %s", e)


def check_task_execution_job():
    """每 3 分钟检查待执行的设备任务，匹配启用规则后自动执行"""
    try:
        import os
        from datetime import datetime
        from core.planting_tracker import PlantingTracker
        from core.device_rule_engine import RuleEngine
        from devices.base import DeviceCommand
        from core.device_executor import DeviceExecutor
        from core.device_registry_factory import setup_registry, close_registry

        data_dir = os.path.join("data")
        usernames = ["default"]
        if os.path.exists(data_dir):
            for d in os.listdir(data_dir):
                user_path = os.path.join(data_dir, d)
                if os.path.isdir(user_path) and os.path.exists(os.path.join(user_path, "planting_tasks.json")):
                    usernames.append(d)

        for username in set(usernames):
            sd = os.path.join("data", username)
            tracker = PlantingTracker(sd)
            pending_tasks = tracker.get_pending_device_tasks()

            if not pending_tasks:
                continue

            engine = RuleEngine(username=username)

            for task in pending_tasks:
                # 检查是否有匹配规则且 evaluate_action 为 auto_execute
                matched = engine.find_matching_rules({})
                matching_rule = None
                for rule in matched:
                    if not rule.get("enabled", True):
                        continue
                    rule_action = rule.get("action", {})
                    decision, reason, final_params = engine.evaluate_action(
                        rule, task.device_params or {},
                        {"device_id": task.device_id}
                    )
                    if decision == "auto_execute":
                        matching_rule = rule
                        break

                if not matching_rule:
                    continue

                # 检查截止日期是否已过
                if task.end_date:
                    try:
                        end_date = datetime.strptime(task.end_date, "%Y-%m-%d")
                        if datetime.now().date() > end_date.date():
                            tracker.update_task_status(task.id, "已逾期")
                            logger.info("任务 %s 已逾期，自动标记", task.title)
                            continue
                    except (ValueError, TypeError):
                        pass

                # 执行任务
                try:
                    registry, loop = setup_registry(username)
                    try:
                        loop.run_until_complete(registry.discover_all())
                        cmd = DeviceCommand(
                            command=task.device_command,
                            params=task.device_params or {},
                        )
                        executor = DeviceExecutor(registry, username=username)
                        tracker.update_task_status(task.id, "进行中", progress=10)
                        result = executor.execute_sync(
                            task.device_id, cmd,
                            trigger="task_scheduler",
                            rule_id=matching_rule.get("id"),
                        )
                        if result["success"]:
                            engine.record_execution(task.device_id, task.device_params or {})
                            tracker.update_task_status(task.id, "已完成", progress=100)
                            logger.info("自动执行任务成功: %s → %s", task.title, task.device_id)
                        else:
                            tracker.update_task_status(task.id, "待办", progress=0)
                            logger.warning("自动执行任务失败: %s → %s", task.title, task.device_id)
                    finally:
                        close_registry(loop)
                except Exception as e:
                    logger.warning("自动执行任务异常: %s → %s", task.title, e)
                    tracker.update_task_status(task.id, "待办", progress=0)

    except Exception as e:
        logger.warning("任务自动执行检查失败: %s", e)


# ── 摄像头定时巡检 ──────────────────────────────────

def _get_int_env(key: str, default: int) -> int:
    """安全读取整数环境变量，解析失败时返回默认值"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning("环境变量 %s 值非法，使用默认值 %d", key, default)
        return default

CHECK_CAMERA_INTERVAL_MINUTES = _get_int_env("CAMERA_CHECK_INTERVAL", 30)


def check_camera_capture_job():
    """每 N 分钟自动拍照 + Vision 分析 + 自主执行

    流程:
    1. 遍历所有用户，发现摄像头设备
    2. 逐个摄像头拍照
    3. 调用 CropMonitorAgent 分析照片
    4. 如有推荐操作，通过 RuleEngine + DeviceExecutor 执行
    """
    try:
        import os
        import json
        import base64
        from datetime import datetime
        from core.device_registry_factory import setup_registry, close_registry, DEFAULT_DATA_DIR
        from devices.base import DeviceCommand

        data_dir = DEFAULT_DATA_DIR
        usernames = ["default"]
        if os.path.exists(data_dir):
            for d in os.listdir(data_dir):
                user_path = os.path.join(data_dir, d)
                if os.path.isdir(user_path) and os.path.exists(
                    os.path.join(user_path, "custom_devices.json")
                ):
                    usernames.append(d)

        for username in set(usernames):
            registry, loop = setup_registry(username)
            try:
                devices = loop.run_until_complete(registry.discover_all())
                cameras = [
                    d for d in devices
                    if "capture" in [c.value for c in d.capabilities]
                    and d.status.value in ("online", "offline")
                ]
                # 记录故障摄像头（有 capture 能力但状态异常）
                error_cameras = [
                    d for d in devices
                    if "capture" in [c.value for c in d.capabilities]
                    and d.status.value == "error"
                ]
                for ec in error_cameras:
                    logger.warning("摄像头故障，跳过巡检: %s/%s", username, ec.device_id)

                if not cameras:
                    continue

                for cam in cameras:
                    try:
                        # 1. 拍摄照片
                        cmd = DeviceCommand(command="capture", params={}, timeout_ms=15000)
                        result = loop.run_until_complete(
                            registry.execute(cam.device_id, cmd)
                        )
                        if not result.success:
                            logger.warning(
                                "摄像头拍照失败: %s → %s", cam.device_id, result.message
                            )
                            continue

                        image_bytes = result.raw_response.get("image_bytes")
                        if not image_bytes:
                            continue

                        # 2. 保存照片到磁盘
                        photo_dir = os.path.join(DEFAULT_DATA_DIR, username, "photos", cam.device_id)
                        os.makedirs(photo_dir, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        photo_path = os.path.join(photo_dir, f"capture_{ts}.jpg")
                        with open(photo_path, "wb") as f:
                            f.write(image_bytes)
                        logger.info("照片已保存: %s", photo_path)

                        # 3. base64 编码 → Vision 分析
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                        from app.agent.agents.crop_monitor_agent import CropMonitorAgent
                        monitor = CropMonitorAgent()
                        analysis_result = monitor.analyze_image(
                            image_b64, "image/jpeg",
                            user_context={
                                "username": username,
                                "device_id": cam.device_id,
                                "location": cam.location,
                            },
                        )

                        # 4. 存储分析结果
                        analysis_path = os.path.join(
                            photo_dir, f"analysis_{ts}.json"
                        )
                        with open(analysis_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "timestamp": datetime.now().isoformat(),
                                "device_id": cam.device_id,
                                "photo_path": photo_path,
                                "analysis": analysis_result.get("analysis", {}),
                                "error": analysis_result.get("error"),
                            }, f, ensure_ascii=False, indent=2)

                        # 5. 如有推荐操作 → 自动执行
                        if analysis_result.get("success"):
                            analysis = analysis_result.get("analysis", {})
                            actions = analysis.get("recommended_actions", [])
                            if actions:
                                _process_camera_actions(
                                    actions, username, registry, loop, cam.device_id
                                )

                    except Exception as e:
                        logger.warning(
                            "摄像头巡检异常 [%s/%s]: %s", username, cam.device_id, e
                        )
            finally:
                close_registry(loop)

    except Exception as e:
        logger.warning("摄像头定时巡检失败: %s", e)


def _process_camera_actions(actions: list, username: str, registry, loop, camera_id: str):
    """处理 Vision 分析推荐的设备操作

    通过 RuleEngine 评估，尊重自主权设置，满足条件则自动执行。
    """
    try:
        from core.device_rule_engine import RuleEngine, RuleDecision, apply_autonomy
        from devices.base import DeviceCommand
        from core.device_executor import DeviceExecutor
        from app.agent.config import get_autonomy_level

        engine = RuleEngine(username=username)
        executor = DeviceExecutor(registry, username=username)
        autonomy = get_autonomy_level()

        # 发现可用的执行设备
        all_devices = loop.run_until_complete(registry.discover_all())

        for action in actions:
            action_type = action.get("action", "")
            urgency = action.get("urgency", "routine")
            detail = action.get("detail", "")

            if action_type in ("none",):
                continue
            if action_type == "alert":
                logger.warning(
                    "摄像头告警 [%s/%s]: %s (紧急度: %s)", username, camera_id, detail, urgency
                )
                continue

            # 映射 action → 能力
            capability_map = {"irrigate": "irrigate", "fertigate": "fertigate"}
            capability = capability_map.get(action_type)
            if not capability:
                continue

            # 找到匹配的执行设备
            target_device = None
            for d in all_devices:
                if capability in [c.value for c in d.capabilities]:
                    target_device = d.device_id
                    break
            if not target_device:
                logger.info("未找到 %s 类型执行设备，跳过", action_type)
                continue

            # 通过 RuleEngine 评估 — 收集真实传感器数据
            sensor_data = {}
            for d in all_devices:
                try:
                    state = loop.run_until_complete(registry.read_state(d.device_id))
                    if state and not state.get("error"):
                        for k, v in state.items():
                            if isinstance(v, (int, float)) and not k.startswith("_"):
                                sensor_data[k] = v
                except Exception:
                    pass  # 单个设备读取失败不影响整体
            context = {"sensor_data": sensor_data, "weather": {}}
            matched = engine.find_matching_rules(context)
            duration = 15  # 默认时长

            if matched:
                rule = matched[0]
                proposed = rule.get("action", {}).get("params", {"duration": duration})
                decision, reason, final_params = engine.evaluate_action(
                    rule, proposed, context
                )
                decision = apply_autonomy(decision, autonomy)

                if decision == RuleDecision.AUTO_EXECUTE:
                    cmd = DeviceCommand(command="start", params=final_params)
                    result = executor.execute_sync(
                        target_device, cmd, trigger="camera", rule_id=rule["id"]
                    )
                    if result["success"]:
                        engine.record_execution(target_device, final_params)
                        logger.info(
                            "摄像头触发操作: %s → %s (%s, 自主权=%s)",
                            target_device, action_type, detail, autonomy,
                        )
                    else:
                        logger.warning("摄像头触发操作失败: %s", target_device)
            else:
                # 无匹配规则 → 仅在 high 自主权下执行
                if autonomy == "high" and urgency in ("immediate", "today"):
                    cmd = DeviceCommand(command="start", params={"duration": duration})
                    result = executor.execute_sync(
                        target_device, cmd, trigger="camera"
                    )
                    if result["success"]:
                        logger.info(
                            "摄像头自主操作 (high): %s → %s", target_device, action_type
                        )

    except Exception as e:
        logger.warning("处理摄像头推荐操作失败: %s", e)


# ── 自主决策巡检（替代原 check_camera_capture_job）──

def _get_active_usernames() -> List[str]:
    """发现所有活跃用户"""
    usernames = ["default"]
    data_dir = os.path.join("data")
    if os.path.exists(data_dir):
        for d in os.listdir(data_dir):
            user_path = os.path.join(data_dir, d)
            if os.path.isdir(user_path):
                if os.path.exists(os.path.join(user_path, "custom_devices.json")) or \
                   os.path.exists(os.path.join(user_path, "planting_progress.json")):
                    usernames.append(d)
    return list(set(usernames))


def check_autonomous_cycle_job():
    """自主决策定时巡检：发现区域 → 数据采集 → LLM决策 → 执行

    替代原有的 check_camera_capture_job，实现完整的感知→决策→执行闭环。
    """
    try:
        from core.autonomous_farm_manager import AutonomousFarmManager
        from app.agent.config import (
            AUTO_DECISION_REGIONS, AUTO_DECISION_MIN_INTERVAL,
        )

        usernames = _get_active_usernames()
        configured_regions = [r.strip() for r in AUTO_DECISION_REGIONS.split(",") if r.strip()]

        for username in usernames:
            manager = AutonomousFarmManager()

            # 发现该用户的区域
            try:
                from core.device_registry_factory import setup_registry, close_registry
                registry, loop = setup_registry(username)
                try:
                    devices = loop.run_until_complete(registry.discover_all())
                    all_regions = set()
                    for d in devices:
                        loc = getattr(d, 'location', '') or '默认区域'
                        all_regions.add(loc)

                    if configured_regions:
                        regions = sorted(all_regions & set(configured_regions))
                    else:
                        regions = sorted(all_regions)
                finally:
                    close_registry(loop)
            except Exception as e:
                logger.warning("区域发现失败 [%s]: %s", username, e)
                continue

            if not regions:
                logger.debug("用户 %s 无可用区域", username)
                continue

            for region in regions:
                # 检查最小间隔
                last = manager._last_run.get(region)
                if last and (datetime.now() - last).total_seconds() < AUTO_DECISION_MIN_INTERVAL * 60:
                    logger.debug("区域 %s 距上次巡检不足%d分钟，跳过", region, AUTO_DECISION_MIN_INTERVAL)
                    continue

                try:
                    logger.info("自主决策巡检: %s/%s", username, region)
                    report = manager.run_cycle(username, region)
                    logger.info("巡检完成: %s/%s — %s", username, region,
                               report.summary.replace('\n', ' | ')[:200])
                except Exception as e:
                    logger.exception("巡检异常 [%s/%s]: %s", username, region, e)

    except Exception as e:
        logger.exception("自主决策调度失败: %s", e)
