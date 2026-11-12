"""SQLAlchemy ORM 模型定义 — 13张表"""
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    region = Column(String(100))
    soil_type = Column(String(50))
    farm_size = Column(Float)
    experience_level = Column(String(20))
    goals = Column(Text)  # JSON array
    phone = Column(String(20))
    autonomy_level = Column(String(10), nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(200))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    cards_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    session = relationship("ChatSession", back_populates="messages")


class PlantingPlan(Base):
    __tablename__ = "planting_plans"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    crop = Column(String(50), nullable=False)
    plot_id = Column(Integer, ForeignKey("fields.id"))
    stage = Column(String(50))
    stage_number = Column(Integer)
    total_stages = Column(Integer)
    start_date = Column(Date)
    expected_end_date = Column(Date)
    actual_end_date = Column(Date)
    progress_percent = Column(Float, default=0)
    status = Column(String(20), default="active")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PlantingTask(Base):
    __tablename__ = "planting_tasks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_id = Column(Integer, ForeignKey("planting_plans.id"))
    crop = Column(String(50))
    task_type = Column(String(50))
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(String(20), default="pending")
    priority = Column(String(10), default="normal")
    progress_percent = Column(Integer, default=0)
    start_date = Column(Date)
    end_date = Column(Date)
    completed_date = Column(Date)
    device_id = Column(String(100))
    device_command = Column(String(100))
    device_params = Column(Text)  # JSON
    execution_state = Column(Text)  # 哥哥，保存本轮设备运行标识和真实起止信息。
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Reminder(Base):
    __tablename__ = "reminders"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    crop = Column(String(50))
    reminder_type = Column(String(50), nullable=False)
    task_description = Column(Text)
    growth_stage = Column(String(50))
    start_date = Column(Date, nullable=True)
    task_id = Column(Integer, nullable=True)
    progress_id = Column(Integer, nullable=True)
    frequency = Column(String(20), default="once")
    interval_days = Column(Integer)
    time_of_day = Column(String(10))
    advance_hours = Column(Integer, default=0)
    channels = Column(Text)  # JSON array
    status = Column(String(20), default="active")
    last_triggered = Column(DateTime)
    next_trigger = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)


class Field(Base):
    __tablename__ = "fields"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(100), nullable=False)
    coordinates = Column(Text, nullable=False)  # JSON
    center_lat = Column(Float)
    center_lon = Column(Float)
    area_mu = Column(Float)
    area_m2 = Column(Float)
    soil_type = Column(String(50))
    current_crop = Column(String(50))
    planting_history = Column(Text)  # JSON array
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class FinanceRecord(Base):
    __tablename__ = "finance_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date, nullable=False)
    crop = Column(String(50))
    plot = Column(String(100))
    record_type = Column(String(10), nullable=False)  # income / cost
    category = Column(String(50))
    item_name = Column(String(200), nullable=False)
    quantity = Column(Float)
    unit = Column(String(20))
    unit_price = Column(Float)
    total_amount = Column(Float, nullable=False)
    buyer = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)


class DeviceConfig(Base):
    __tablename__ = "device_configs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    device_id = Column(String(100), nullable=False)
    name = Column(String(200))
    driver = Column(String(50), nullable=False)
    capabilities = Column(Text)   # JSON array
    sensors = Column(Text)        # JSON array
    connection = Column(Text)     # JSON
    location = Column(String(200))
    plot_id = Column(Integer)
    zone_id = Column(String(100))
    initial_state = Column(Text)  # JSON
    created_at = Column(DateTime, default=datetime.now)


class DeviceRule(Base):
    __tablename__ = "device_rules"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(200), nullable=False)
    enabled = Column(Integer, default=1)
    conditions = Column(Text, nullable=False)   # JSON
    actions = Column(Text, nullable=False)       # JSON
    constraints = Column(Text)                    # JSON
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class DeviceSafetyPolicy(Base):
    """用户可配置的设备安全策略。

    设备物理绝对上限仍由代码维护；本表保存用户在物理上限以内设置的
    地块、设备或能力级运行边界。
    """
    __tablename__ = "device_safety_policies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(200), nullable=False)
    enabled = Column(Integer, default=1)
    scope_type = Column(String(20), default="capability")
    capability = Column(String(50))
    device_id = Column(String(100))
    plot_id = Column(Integer)
    zone_id = Column(String(100))
    limits = Column(Text, nullable=False)  # JSON
    violation_action = Column(String(20), default="reject")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class FieldZone(Base):
    """地块内可独立观测或控制的作业分区。"""
    __tablename__ = "field_zones"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    field_id = Column(Integer, ForeignKey("fields.id", ondelete="CASCADE"), nullable=False)
    zone_id = Column(String(100), nullable=False)
    name = Column(String(200), nullable=False)
    zone_type = Column(String(30), default="operation")
    coordinates = Column(Text, default="[]")
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "field_id",
            "zone_id",
            name="uq_field_zone_user_field_zone",
        ),
    )


class AutonomousCycleLease(Base):
    """跨进程巡检租约，同一用户和区域只有一条记录。"""
    __tablename__ = "autonomous_cycle_leases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)
    region = Column(String(200), nullable=False)
    claimed_at = Column(DateTime, nullable=False)
    lease_until = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("username", "region", name="uq_cycle_lease_user_region"),
    )


class DeviceActionLog(Base):
    __tablename__ = "device_action_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    device_id = Column(String(100), nullable=False)
    command = Column(String(100), nullable=False)
    params = Column(Text)        # JSON
    trigger = Column(String(50))
    rule_id = Column(Integer, ForeignKey("device_rules.id"))
    decision = Column(String(20), default="auto")
    status = Column(String(20), default="pending")
    success = Column(Integer, default=1)
    attempts = Column(Integer, default=1)
    message = Column(Text)
    error_code = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)


class DiseaseRisk(Base):
    __tablename__ = "disease_risks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    crop = Column(String(50), nullable=False)
    disease = Column(String(100), nullable=False)
    risk_level = Column(String(20), nullable=False)
    score = Column(Float)
    matched_conditions = Column(Text)  # JSON
    advice = Column(Text)
    assessed_at = Column(DateTime, default=datetime.now)


class InspectionReport(Base):
    __tablename__ = "inspection_reports"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    device_id = Column(String(100))
    cycle_id = Column(String(100))
    photo_path = Column(String(500))
    analysis_result = Column(Text)  # JSON
    crop_type = Column(String(50))
    health_status = Column(String(50))
    issues_found = Column(Text)     # JSON
    actions_taken = Column(Text)    # JSON
    duration_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)


class AutonomyJob(Base):
    """哥哥，延后动作和效果复查使用同一份可恢复任务账本。"""
    __tablename__ = "autonomy_jobs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), nullable=False, index=True)
    region = Column(String(200), nullable=False)
    kind = Column(String(50), nullable=False)
    dedup_key = Column(String(128), nullable=False)
    payload = Column(Text, nullable=False)
    status = Column(String(30), nullable=False, default="pending", index=True)
    due_at = Column(DateTime, nullable=True, index=True)
    lease_until = Column(DateTime, nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    last_error = Column(Text, default="")
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)
    __table_args__ = (UniqueConstraint("username", "dedup_key", name="uq_autonomy_job_key"),)


class AutonomyAlert(Base):
    """送达和人工确认分开记录，发送失败不能变成成功。"""
    __tablename__ = "autonomy_alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), nullable=False, index=True)
    region = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(20), default="warning")
    dedup_key = Column(String(128), nullable=False)
    status = Column(String(30), default="pending", index=True)
    attempts = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(DateTime, nullable=True)
    lease_until = Column(DateTime, nullable=True)
    last_error = Column(Text, default="")
    delivered_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)
    __table_args__ = (UniqueConstraint("username", "dedup_key", name="uq_autonomy_alert_key"),)


class SchedulerHeartbeat(Base):
    __tablename__ = "scheduler_heartbeats"
    id = Column(String(100), primary_key=True)
    status = Column(String(30), nullable=False)
    detail = Column(Text, default="")
    process_id = Column(Integer)
    last_seen_at = Column(DateTime, nullable=False)
