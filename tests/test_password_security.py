"""部署场景下的密码存储安全测试。"""

from core.password_security import hash_password, verify_password


def test_新密码不会以明文保存():
    stored = hash_password("哥哥的密码")

    assert "哥哥的密码" not in stored
    assert verify_password("哥哥的密码", stored) == (True, False)
    assert verify_password("错误密码", stored) == (False, False)


def test_旧版明文密码可登录并标记升级():
    assert verify_password("legacy", "legacy") == (True, True)
    assert verify_password("wrong", "legacy") == (False, True)
