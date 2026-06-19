"""企业微信/钉钉群机器人通知 — Webhook 推送农事提醒"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional

import dotenv
import requests

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

WECHAT_WEBHOOK_URL = os.getenv("WECHAT_WEBHOOK_URL", "")


def is_configured() -> bool:
    return bool(WECHAT_WEBHOOK_URL)


def send_text(content: str) -> Dict:
    """发送纯文本消息"""
    if not is_configured():
        return {"success": False, "error": "企业微信 Webhook 未配置"}

    try:
        resp = requests.post(
            WECHAT_WEBHOOK_URL,
            json={"msgtype": "text", "text": {"content": content}},
            timeout=10,
        )
        data = resp.json()
        ok = data.get("errcode") == 0
        if not ok:
            logger.warning("企微通知发送失败: %s", data.get("errmsg", ""))
        return {"success": ok, "errcode": data.get("errcode"), "errmsg": data.get("errmsg", "")}
    except Exception as e:
        logger.warning("企微通知请求异常: %s", e)
        return {"success": False, "error": str(e)}


def send_markdown(title: str, content: str) -> Dict:
    """发送 Markdown 格式消息"""
    if not is_configured():
        return {"success": False, "error": "企业微信 Webhook 未配置"}

    md = f"## {title}\n{content}"
    try:
        resp = requests.post(
            WECHAT_WEBHOOK_URL,
            json={"msgtype": "markdown", "markdown": {"content": md}},
            timeout=10,
        )
        data = resp.json()
        ok = data.get("errcode") == 0
        if not ok:
            logger.warning("企微 Markdown 通知发送失败: %s", data.get("errmsg", ""))
        return {"success": ok, "errcode": data.get("errcode"), "errmsg": data.get("errmsg", "")}
    except Exception as e:
        logger.warning("企微 Markdown 通知请求异常: %s", e)
        return {"success": False, "error": str(e)}


def send_reminder(crop: str, task_type: str, task_desc: str = "",
                  time_info: str = "", alert_type: str = "农事提醒") -> Dict:
    """发送农事提醒（Markdown 格式，信息更丰富）"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"**作物**：{crop}",
        f"**任务**：{task_type}",
    ]
    if task_desc:
        lines.append(f"**说明**：{task_desc[:80]}")
    if time_info:
        lines.append(f"**时间**：{time_info}")
    lines.append(f"**发送时间**：{now}")
    return send_markdown(f"🌾 {alert_type}", "\n".join(lines))
