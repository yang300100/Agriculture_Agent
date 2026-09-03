"""从中国政府网检索真实农业政策。"""

import base64
import html
import re
from typing import Dict, List
from urllib.parse import quote

import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding


_PUBLIC_KEY_DER = (
    "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCSMhMJQ+XLI7oW0k9Bwufur4Ag40tc"
    "srzT7WZf6Ao0O/hyY1gZtCSYFxkxIZUXjW46j27XSW8IDX1rTJoHaMxHCWsOpTi2W5s"
    "tybGYZytsY5on8gd8AIaS1d52h9eaS2TFydtJJtE50xHmT0WmoyoinWCuVCOkdCLhh9"
    "b9jSdeSQIDAQAB"
)
_APP_KEY_TEXT = b"a46884b2013e4d189f2a8e2d49a23525"
_APP_NAME = "%E5%9B%BD%E7%BD%91%E6%90%9C%E7%B4%A2"
_SEARCH_URL = (
    "https://sousuoht.www.gov.cn/athena/forward/"
    "2B22E8E39E850E17F95A016A74FCB6B673336FA8B6FEC0E2955907EF9AEE06BE"
)


def _strip_markup(value) -> str:
    """移除搜索高亮标签，保留可读纯文本。"""
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _official_headers() -> Dict[str, str]:
    """生成中国政府网站内搜索要求的公开调用标识。"""
    public_key = serialization.load_der_public_key(
        base64.b64decode(_PUBLIC_KEY_DER)
    )
    encrypted = public_key.encrypt(_APP_KEY_TEXT, padding.PKCS1v15())
    return {
        "User-Agent": "Qinghe-Smart-Farm/1.0",
        "athenaAppName": _APP_NAME,
        "athenaAppKey": quote(base64.b64encode(encrypted).decode(), safe=""),
    }


def search_official_policies(query: str, limit: int = 8) -> List[Dict]:
    """检索中国政府网政策库并返回稳定的前端字段。"""
    keyword = str(query or "").strip()
    if not keyword:
        return []
    safe_limit = max(1, min(int(limit), 20))
    payload = {
        "code": "17da70961a7",
        "searchWord": keyword,
        "dataTypeId": "14",
        "orderBy": "time",
        "searchBy": "all",
        "appendixType": "",
        "granularity": "ALL",
        "trackTotalHits": True,
        "beginDateTime": "",
        "endDateTime": "",
        "isSearchForced": 0,
        "filters": [],
        "pageNo": 1,
        "pageSize": safe_limit,
        "customFilter": {"operator": "and", "properties": []},
    }
    response = requests.post(
        _SEARCH_URL,
        headers=_official_headers(),
        json=payload,
        timeout=12,
    )
    response.raise_for_status()
    body = response.json()
    result_code = body.get("resultCode", {}).get("code")
    if result_code != 200:
        raise RuntimeError("中国政府网政策检索暂时不可用")
    data = body.get("result", {}).get("data") or {}
    rows = data.get("middle", {}).get("list", []) or []

    results = []
    for row in rows:
        title = _strip_markup(row.get("title_no_tag") or row.get("title"))
        url = str(row.get("url") or "").strip()
        if not title or not url.startswith(("https://", "http://")):
            continue
        results.append(
            {
                "title": title,
                "summary": _strip_markup(row.get("summary") or row.get("content")),
                "url": url,
                "source": "中国政府网",
                "published_at": str(row.get("time") or ""),
            }
        )
    return results[:safe_limit]
