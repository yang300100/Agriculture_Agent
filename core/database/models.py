"""SQLAlchemy ORM 模型定义 — 13张表"""
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, ForeignKey
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
    start_date = Column(Date)
    end_date = Column(Date)
    completed_date = Column(Date)
    device_id = Column(String(100))
    device_command = Column(String(100))
    device_params = Column(Text)  # JSON
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
