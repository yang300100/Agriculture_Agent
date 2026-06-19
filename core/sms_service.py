"""腾讯云短信服务 — 发送农事提醒短信"""

import os
import json
import logging
from typing import List, Dict, Optional

import dotenv
from tencentcloud.common import credential
from tencentcloud.sms.v20210111 import sms_client, models
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# 短信配置
SMS_SECRET_ID = os.getenv("SMS_SECRET_ID", "")
SMS_SECRET_KEY = os.getenv("SMS_SECRET_KEY", "")
SMS_SDK_APP_ID = os.getenv("SMS_SDK_APP_ID", "")
SMS_SIGN_NAME = os.getenv("SMS_SIGN_NAME", "")
SMS_TEMPLATE_ID = os.getenv("SMS_TEMPLATE_ID", "")
SMS_REGION = os.getenv("SMS_REGION", "ap-guangzhou")


class SMSService:
    """腾讯云短信服务"""

    def __init__(self, secret_id: str = None, secret_key: str = None,
                 sdk_app_id: str = None, sign_name: str = None,
                 template_id: str = None, region: str = None):
        self.secret_id = secret_id or SMS_SECRET_ID
        self.secret_key = secret_key or SMS_SECRET_KEY
        self.sdk_app_id = sdk_app_id or SMS_SDK_APP_ID
        self.sign_name = sign_name or SMS_SIGN_NAME
        self.template_id = template_id or SMS_TEMPLATE_ID
        self.region = region or SMS_REGION

    @property
    def is_configured(self) -> bool:
        return bool(self.secret_id and self.secret_key and
                   self.sdk_app_id and self.sign_name and self.template_id)

    def _get_client(self) -> sms_client.SmsClient:
        cred = credential.Credential(self.secret_id, self.secret_key)
        return sms_client.SmsClient(cred, self.region)

    def send_sms(self, phone_numbers: List[str], template_params: List[str] = None,
                 session_context: str = "", extend_code: str = "") -> Dict:
        """
        发送短信

        Args:
            phone_numbers: 手机号列表，支持 11位国内号码或 E.164 格式
            template_params: 模板参数列表，与模板变量一一对应
            session_context: 用户上下文（可选，最大512字节）
            extend_code: 短信码号扩展号（可选）

        Returns:
            {"success": bool, "results": [...], "request_id": str, "error": str}
        """
        if not self.is_configured:
            return {"success": False, "error": "短信服务未配置，请在 .env 中设置 SMS_SECRET_ID 等参数"}

        try:
            client = self._get_client()
            req = models.SendSmsRequest()

            # 格式化手机号为 E.164
            formatted = [_normalize_phone(p) for p in phone_numbers]
            req.PhoneNumberSet = formatted
            req.SmsSdkAppId = self.sdk_app_id
            req.SignName = self.sign_name
            req.TemplateId = self.template_id

            if template_params:
                req.TemplateParamSet = template_params
            if session_context:
                req.SessionContext = session_context[:512]
            if extend_code:
                req.ExtendCode = extend_code

            resp = client.SendSms(req)
            results = []
            for status in resp.SendStatusSet:
                results.append({
                    "phone": status.PhoneNumber,
                    "code": status.Code,
                    "message": status.Message,
                    "serial": status.SerialNo,
                    "fee": status.Fee,
                })

            all_ok = all(s.get("code") == "Ok" for s in results)
            logger.info(f"SMS sent to {len(formatted)} numbers, all_ok={all_ok}")

            return {
                "success": all_ok,
                "results": results,
                "request_id": resp.RequestId,
                "error": "" if all_ok else "部分号码发送失败",
            }

        except TencentCloudSDKException as e:
            logger.error(f"SMS SDK error: {e}")
            return {"success": False, "error": str(e), "results": [], "request_id": ""}
        except Exception as e:
            logger.error(f"SMS send error: {e}")
            return {"success": False, "error": str(e), "results": [], "request_id": ""}

    def send_reminder(self, phone: str, crop: str, task_type: str, task_desc: str = "",
                      time_info: str = "") -> Dict:
        """
        发送农事提醒短信（使用配置的模板）

        Args:
            phone: 手机号码
            crop: 作物名称
            task_type: 任务类型（浇水/施肥/打药等）
            task_desc: 任务描述
            time_info: 时间信息

        Returns:
            {"success": bool, ...}
        """
        # 模板参数顺序取决于腾讯云审核通过的模板
        # 默认模板：{1}作物 {2}任务类型 {3}任务描述 {4}时间
        params = [crop, task_type, task_desc[:50] if task_desc else "", time_info]

        context = json.dumps({
            "crop": crop,
            "task_type": task_type,
            "reminded_at": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False)

        return self.send_sms(
            phone_numbers=[phone],
            template_params=params,
            session_context=context,
        )


def _normalize_phone(phone: str) -> str:
    """将手机号规范化为 E.164 格式"""
    phone = phone.strip()
    if phone.startswith("+"):
        return phone
    if phone.startswith("0086"):
        return "+86" + phone[4:]
    if phone.startswith("86") and len(phone) == 13:
        return "+" + phone
    if len(phone) == 11 and phone.isdigit():
        return "+86" + phone
    return phone


# 便捷函数
def get_sms_service() -> SMSService:
    return SMSService()
