"""AgentState 数据模型"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    user_question: Optional[str] = None
    intent_type: Optional[Literal[
        "greeting", "thanks", "farewell", "identity", "function",
        "crop_selection", "planting_schedule", "planting_method",
        "reminder_setup", "progress_tracking", "disease_prevention",
        "harvest_planning", "image_analysis", "weather_query",
        "finance_query", "field_management",
        "device_control", "crop_monitoring", "unclear"
    ]] = None
    short_term_facts: Dict[str, Any] = Field(default_factory=dict)
    long_term_profile: Dict[str, Any] = Field(default_factory=lambda: {
        "summary": "",
        "conversation_round": 0,
        "user_profile": {}
    })
    need_rag: bool = False
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    need_clarification: bool = False
    refuse_answer: bool = False
    final_answer: Optional[str] = None

    planting_plan: Dict[str, Any] = Field(default_factory=lambda: {
        "crops": [],
        "schedule": {},
        "methods": {},
        "progress": {},
        "created_at": None
    })

    reminders: List[Dict[str, Any]] = Field(default_factory=list)

    user_profile: Dict[str, Any] = Field(default_factory=lambda: {
        "region": "",
        "climate": "",
        "soil_type": "",
        "farm_size": 0,
        "experience": "",
        "goals": []
    })

    image_data: Optional[str] = None
    image_mime_type: Optional[str] = None
    image_analysis_result: Optional[Dict[str, Any]] = Field(default_factory=dict)
    has_image: bool = False

    fields_data: List[Dict[str, Any]] = Field(default_factory=list)
    current_field_id: Optional[str] = None

    # 设备控制相关（新增）
    username: str = "default"                                   # 当前用户名
    device_command: Optional[Dict[str, Any]] = None       # 待执行的设备指令
    device_result: Optional[Dict[str, Any]] = None        # 执行结果
    pending_action: Optional[Dict[str, Any]] = None       # 待用户确认的操作
    matched_rules: List[str] = Field(default_factory=list)  # 命中的规则ID列表
