"""哥哥，无人值守只接受已接线的结构化设备安全契约。"""
import math


def unattended_contract_error(config: dict, command: str, params: dict) -> str | None:
    if command != "start":
        return None
    driver = str(config.get("driver", config.get("driver_name", ""))).lower()
    if driver == "simulator":
        return None
    if driver not in {"http", "coap", "mqtt"}:
        return f"{driver or '未知'} 驱动未接线无人值守执行确认与去重"
    connection = config.get("connection")
    autonomy = connection.get("autonomy") if isinstance(connection, dict) else None
    if not isinstance(autonomy, dict):
        return "缺少结构化无人值守安全契约"
    timed = autonomy.get("local_timed_stop")
    if not isinstance(timed, dict) or timed.get("duration_param") != "duration" or timed.get("unit") != "minutes":
        return "缺少已接线的端侧定时停止配置"
    watchdog = autonomy.get("link_loss_stop")
    if not isinstance(watchdog, dict) or watchdog.get("mode") != "firmware_watchdog":
        return "缺少固件失联保护配置"
    timeout = watchdog.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        return "失联保护超时必须为正有限数"
    eid = autonomy.get("execution_id")
    if not isinstance(eid, dict) or eid.get("field") != "execution_id" or eid.get("deduplicates") is not True:
        return "缺少端侧执行编号去重配置"
    ack = autonomy.get("ack")
    if not isinstance(ack, dict) or ack.get("execution_id_field") != "execution_id" or ack.get("success_field") != "success":
        return "缺少匹配执行编号的设备回执配置"
    if driver == "mqtt" and (not isinstance(ack.get("topic"), str) or not ack["topic"].strip() or "+" in ack["topic"] or "#" in ack["topic"]):
        return "MQTT 缺少具体回执主题"
    return None
