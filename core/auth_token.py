"""无额外依赖的短期登录令牌。"""

import base64
import hashlib
import hmac
import json
import time


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def create_token(username: str, secret: str, ttl_seconds: int = 604800) -> str:
    """创建包含用户和过期时间的 HMAC 签名令牌。"""
    payload = _encode(json.dumps(
        {"sub": username, "exp": int(time.time()) + ttl_seconds},
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))
    signature = _encode(hmac.new(
        secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256
    ).digest())
    return f"{payload}.{signature}"


def verify_token(token: str, secret: str) -> str | None:
    """验证签名和有效期，成功时返回用户名。"""
    try:
        payload, signature = token.split(".", 1)
        expected = _encode(hmac.new(
            secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256
        ).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        data = json.loads(_decode(payload).decode("utf-8"))
        if int(data.get("exp", 0)) < int(time.time()):
            return None
        return str(data.get("sub", "")) or None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
