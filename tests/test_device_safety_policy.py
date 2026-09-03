"""统一设备安全策略与动作参数目录测试。"""

import pytest

from core.device_action_schema import normalize_action
from core.device_safety_policy import (
    AUTO_EXECUTE,
    NEED_CONFIRM,
    REJECTED,
    SafetyPolicyService,
)


def _policy(policy_id=1, **updates):
    """构造不访问数据库的安全策略。"""
    policy = {
        "id": policy_id,
        "name": f"测试策略{policy_id}",
        "enabled": True,
        "scope_type": "capability",
        "capability": "irrigate",
        "device_id": None,
        "plot_id": None,
        "zone_id": None,
        "limits": {},
        "violation_action": "reject",
    }
    policy.update(updates)
    return policy


def test_物理时长上限不可被普通策略突破():
    service = SafetyPolicyService(policies=[])

    result = service.evaluate(
        "valve_01", "irrigate", {"duration": 121}, command="start"
    )

    assert result.decision == REJECTED
    assert "物理上限" in result.reason


@pytest.mark.parametrize(
    ("violation_action", "expected"),
    [("reject", REJECTED), ("confirm", NEED_CONFIRM)],
)
def test_用户单次限制可选择拒绝或请求确认(violation_action, expected):
    service = SafetyPolicyService(policies=[_policy(
        limits={"max_duration_per_use_minutes": 30},
        violation_action=violation_action,
    )])

    result = service.evaluate("valve_01", "irrigate", {"duration": 40})

    assert result.decision == expected
    assert result.matched_policy_ids == [1]


def test_设备与地块范围只约束匹配目标():
    policies = [
        _policy(
            11,
            name="指定设备策略",
            scope_type="device",
            device_id="valve_01",
            limits={"max_duration_per_use_minutes": 20},
        ),
        _policy(
            12,
            name="二号地块策略",
            scope_type="plot",
            plot_id=2,
            limits={"max_duration_per_use_minutes": 10},
        ),
    ]
    service = SafetyPolicyService(policies=policies)

    target = service.evaluate(
        "valve_01", "irrigate", {"duration": 25}, context={"plot_id": 1}
    )
    other = service.evaluate(
        "valve_02", "irrigate", {"duration": 25}, context={"plot_id": 1}
    )
    plot = service.evaluate(
        "valve_02", "irrigate", {"duration": 15}, context={"plot_id": 2}
    )

    assert target.decision == REJECTED
    assert target.matched_policy_ids == [11]
    assert other.decision == AUTO_EXECUTE
    assert other.matched_policy_ids == []
    assert plot.decision == REJECTED
    assert plot.matched_policy_ids == [12]


def test_标称流量乘持续时间可用于水量限制():
    service = SafetyPolicyService(policies=[_policy(
        limits={
            "rated_flow_lpm": 10,
            "max_volume_per_use_liters": 100,
        }
    )])

    result = service.evaluate("valve_01", "irrigate", {"duration": 15})

    assert result.decision == REJECTED
    assert result.calculated_volume_liters == 150
    assert "单次水量" in result.reason


def test_要求传感器数据时缺失会阻止自动执行():
    service = SafetyPolicyService(policies=[_policy(
        limits={"require_sensor_data": True},
        violation_action="confirm",
    )])

    missing = service.evaluate("valve_01", "irrigate", {"duration": 10})
    present = service.evaluate(
        "valve_01",
        "irrigate",
        {"duration": 10},
        context={"sensor_data": {"soil_moisture": 25}},
    )

    assert missing.decision == NEED_CONFIRM
    assert "缺少传感器数据" in missing.reason
    assert present.decision == AUTO_EXECUTE


def test_停止操作作为减险动作直接放行():
    service = SafetyPolicyService(policies=[_policy(
        limits={"max_duration_per_use_minutes": 1},
    )])

    result = service.evaluate(
        "valve_01", "irrigate", {"duration": 9999}, command="stop"
    )

    assert result.decision == AUTO_EXECUTE
    assert "直接放行" in result.reason


def test_策略配置不能超过设备物理绝对上限():
    service = SafetyPolicyService(policies=[])

    with pytest.raises(ValueError, match="物理上限"):
        service.validate_policy({
            "name": "错误上限",
            "scope_type": "capability",
            "capability": "irrigate",
            "limits": {"max_duration_per_use_minutes": 121},
        })


def test_临时策略没有数据库ID也不会导致评估崩溃():
    service = SafetyPolicyService(policies=[_policy(
        policy_id=None,
        limits={"max_duration_per_use_minutes": 5},
    )])

    result = service.evaluate("valve_01", "irrigate", {"duration": 10})

    assert result.decision == REJECTED
    assert result.matched_policy_ids == []
    assert service.get_policy(1) is None


def test_动作参数支持启动停止与单参数设置():
    assert normalize_action(
        "irrigate", "start", {"duration": "30", "unknown": 1}
    ) == {"duration": 30}
    assert normalize_action(
        "irrigate", "stop", {"duration": 9999}
    ) == {}
    assert normalize_action(
        "irrigate", "set_param", {"flow_rate": 12.5}
    ) == {"flow_rate": 12.5}


def test_动作参数拒绝越界值和多参数设置():
    with pytest.raises(ValueError, match="必须在"):
        normalize_action("irrigate", "start", {"duration": 121})
    with pytest.raises(ValueError, match="必须且只能"):
        normalize_action(
            "irrigate", "set_param", {"duration": 10, "flow_rate": 20}
        )
