"""哥哥，动作与安全策略都要拒绝无穷大和无效数值。"""

import json
import pytest

from core.device_action_schema import normalize_action
from core.device_safety_policy import REJECTED, SafetyPolicyService


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", float("nan"), float("inf")])
def test_动作目录拒绝非有限运行参数(value):
    with pytest.raises(ValueError):
        normalize_action("irrigate", "start", {"duration": value})


@pytest.mark.parametrize("key,value", [
    ("duration", "NaN"), ("duration", "Infinity"), ("amount_kg", "abc"),
    ("brightness_percent", float("nan")), ("target_temp", "Infinity"),
])
def test_安全门禁拒绝无效数字且结果可序列化(key, value):
    result = SafetyPolicyService(policies=[]).evaluate("demo_pump", "irrigate", {key: value})
    assert result.decision == REJECTED
    json.dumps(result.to_dict(), allow_nan=False)


def test_安全策略上限不能保存NaN():
    with pytest.raises(ValueError):
        SafetyPolicyService(policies=[]).validate_policy({
            "capability": "irrigate", "limits": {"max_duration_per_use_minutes": "NaN"},
        })
