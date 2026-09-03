"""登录令牌签名与过期验证测试。"""

from core.auth_token import create_token, verify_token


def test_签名令牌可验证且不能被篡改():
    token = create_token("哥哥", "strong-secret")

    assert verify_token(token, "strong-secret") == "哥哥"
    assert verify_token(token + "x", "strong-secret") is None
    assert verify_token(token, "wrong-secret") is None


def test_过期令牌会被拒绝():
    token = create_token("哥哥", "strong-secret", ttl_seconds=-1)

    assert verify_token(token, "strong-secret") is None
