"""密码散列与旧数据兼容验证。"""

import base64
import hashlib
import hmac
import secrets


SCHEME = "pbkdf2_sha256"
ITERATIONS = 390_000


def hash_password(password: str) -> str:
    """使用带随机盐的 PBKDF2-SHA256 保存密码。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return "$".join((
        SCHEME,
        str(ITERATIONS),
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    ))


def verify_password(password: str, stored: str) -> tuple[bool, bool]:
    """返回（是否匹配，是否为需要升级的旧版明文）。"""
    if not stored.startswith(f"{SCHEME}$"):
        return hmac.compare_digest(password, stored), True
    try:
        _, iterations, salt_text, digest_text = stored.split("$", 3)
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        )
        return hmac.compare_digest(actual, expected), False
    except (ValueError, TypeError):
        return False, False
