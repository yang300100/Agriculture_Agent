# 数据库迁移 + DL模型接口 + Bug修复 实施计划

> **对于 agentic workers:** 推荐使用 subagent-driven-development 或 executing-plans 按任务逐步实施。步骤使用 checkbox (`- [ ]`) 语法进行追踪。

**目标:** 将数据存储从JSON文件迁移至SQLite数据库，实现本地深度学习病虫害分类模型接口（替代Vision API），并修复代码审查中发现的bug。

**架构:** 新增 `core/database/` 数据库层（SQLAlchemy ORM + Repository模式），新增 `models/` DL模型接口（抽象基类 + ONNX/Torch双后端 + 注册中心），重写12个数据管理模块适配SQL，删除Vision API配置。

**技术栈:** SQLAlchemy 2.0, Alembic, SQLite, ONNX Runtime, PyTorch, FastAPI, Streamlit

## 全局约束

- SQLite数据库文件: `data/agriculture.db`
- DL模型权重目录: `models/weights/`
- 删除配置项: `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL`, `VISION_TEMPERATURE`, `ENABLE_IMAGE_ANALYSIS`
- 新增配置项: `DL_BACKEND`, `DL_MODELS_DIR`, `DL_DEVICE`, `DL_DEFAULT_MODEL`
- 所有数据写入必须使用原子操作（临时文件 + os.replace）
- 迁移脚本幂等（已存在数据则跳过）

---

### Task 1: 修复 close_registry 连接泄漏

**文件:**
- 修改: `app/scheduler_jobs.py:386,549,727`
- 修改: `app/api_routes.py:683,1055,1101,1116`

- [ ] 修复 `app/scheduler_jobs.py` 中 3 处 `close_registry(loop)` → `close_registry(loop, registry)`

```python
# Line 386 处 (check_device_rules_job 回退分支)
# 旧:
                    close_registry(loop)
# 新:
                    close_registry(loop, registry)

# Line 549 处 (check_camera_capture_job 回退分支)
# 旧:
                close_registry(loop)
# 新:
                close_registry(loop, registry)

# Line 727 处 (check_autonomous_cycle_job 回退分支)
# 旧:
                        close_registry(loop)
# 新:
                        close_registry(loop, registry)
```

- [ ] 修复 `app/api_routes.py` 中 4 处同样问题

```python
# Line 683 (inspection端点)
# 旧:
                    close_registry(loop)
# 新:
                    close_registry(loop, registry)

# Line 1055 (规则测试端点)
# 旧:
                close_registry(loop)
# 新:
                close_registry(loop, registry)

# Line 1101 (confirm_action端点)
# 旧:
                close_registry(loop)
# 新:
                close_registry(loop, registry)

# Line 1116 (reject_action端点)
# 旧:
                close_registry(loop)
# 新:
                close_registry(loop, registry)
```

- [ ] 提交

```bash
git add app/scheduler_jobs.py app/api_routes.py
git commit -m "fix: close_registry传入registry参数，防止驱动连接泄漏"
```

---

### Task 2: 修复 _group_by_region 测试断裂

**文件:**
- 修改: `tests/test_autonomous_farm_manager.py:89`

- [ ] 修复测试中对静态方法的调用

```python
# Line 89 处
# 旧:
        regions = AutonomousFarmManager._group_by_region(devices)
# 新:
        mgr = AutonomousFarmManager()
        regions = mgr._group_by_region(devices)
```

- [ ] 运行测试验证

```bash
python -m pytest tests/test_autonomous_farm_manager.py::test_group_by_region -v
```

- [ ] 提交

```bash
git add tests/test_autonomous_farm_manager.py
git commit -m "fix: _group_by_region测试适配实例方法签名"
```

---

### Task 3: 修复 record_execution 缺少 success 参数

**文件:**
- 修改: `app/agent/agents/device_agent.py:374`

- [ ] 查看上下文确认 result 变量的可用性，然后修改调用

```python
# Line 374 附近，_do_execute 方法中
# 需要查看上下文中 result 变量的位置。假设 result 在执行后可用：
# 旧:
                engine.record_execution(device_id, params)
# 新:
                engine.record_execution(device_id, params, success=result.get("success", False))
```

- [ ] 提交

```bash
git add app/agent/agents/device_agent.py
git commit -m "fix: record_execution传入success参数，失败操作不消耗配额"
```

---

### Task 4: 修复 _safe_parse_capabilities 兜底值

**文件:**
- 修改: `core/device_registry_factory.py:55`

- [ ] 修改兜底值从 `irrigate` 改为空列表 + warning

```python
# Line 55 (_safe_parse_capabilities 函数末尾)
# 旧:
    return caps or [DeviceCapability("irrigate")]
# 新:
    if not caps:
        logger.warning("设备能力解析结果为空，将不赋予任何能力")
    return caps
```

- [ ] 提交

```bash
git add core/device_registry_factory.py
git commit -m "fix: _safe_parse_capabilities空结果不再默认赋予灌溉能力"
```

---

### Task 5: 修复 invalidate_registry_cache event loop 泄漏

**文件:**
- 修改: `core/device_registry_factory.py:330-350`

- [ ] 在 finally 块中确保 tmp_loop 被关闭

```python
# invalidate_registry_cache 函数中的循环体
# 旧:
        try:
            tmp_loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(tmp_loop)
            tmp_loop.run_until_complete(registry.disconnect_all())
            tmp_loop.close()
        except Exception:
            pass
# 新:
        tmp_loop = None
        try:
            tmp_loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(tmp_loop)
            tmp_loop.run_until_complete(registry.disconnect_all())
        except Exception:
            logger.warning("缓存清理时断开驱动失败", exc_info=True)
        finally:
            if tmp_loop is not None:
                try:
                    tmp_loop.close()
                except Exception:
                    pass
            _asyncio.set_event_loop(None)
```

- [ ] 提交

```bash
git add core/device_registry_factory.py
git commit -m "fix: invalidate_registry_cache异常时确保event loop被关闭"
```

---

### Task 6: 修复 InspectionLogger 非原子写入 + _eval_confirm_expr 日志级别

**文件:**
- 修改: `app/scheduler_jobs.py:45-65`
- 修改: `core/device_rule_engine.py:486-488`

- [ ] InspectionLogger.log() 改为原子写入

```python
# InspectionLogger.log 方法
# 在写入部分，将直接 write 改为临时文件 + os.replace
# 旧:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
# 新:
        tmp_path = log_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, log_path)
```

- [ ] _eval_confirm_expr 日志级别提升

```python
# Line 486-488
# 旧:
            except Exception as e:
                logger.debug("确认表达式求值失败: %s, 错误: %s", expr, e)
                pass
# 新:
            except Exception as e:
                logger.warning("确认表达式求值失败: %s, 错误: %s", expr, e)
                pass
```

- [ ] 提交

```bash
git add app/scheduler_jobs.py core/device_rule_engine.py
git commit -m "fix: InspectionLogger原子写入 + _eval_confirm_expr日志级别提升"
```

---

### Task 7: 搭建数据库基础设施

**文件:**
- 创建: `core/database/__init__.py`
- 创建: `core/database/engine.py`
- 创建: `core/database/models.py`
- 创建: `core/database/repository/__init__.py`
- 创建: `core/database/repository/base.py`

- [ ] 创建 `core/database/__init__.py`

```python
"""数据库层 - SQLAlchemy ORM + Repository 模式"""
from core.database.engine import get_session, init_db, Session
from core.database.models import Base
```

- [ ] 创建 `core/database/engine.py`

```python
"""SQLAlchemy 引擎与 Session 工厂"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SASession

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "agriculture.db")
DB_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

_engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=_engine)


def get_session() -> SASession:
    """获取一个新的数据库会话"""
    return Session()


def init_db():
    """初始化数据库，创建所有表"""
    from core.database.models import Base
    Base.metadata.create_all(_engine)
```

- [ ] 创建 `core/database/models.py` — 13个ORM模型

```python
"""SQLAlchemy ORM 模型定义"""
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, ForeignKey, create_engine
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
    device_id = Column(String(100), unique=True, nullable=False)
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
```

- [ ] 创建 `core/database/repository/__init__.py`

```python
"""数据仓库层"""
from core.database.repository.base import BaseRepository
```

- [ ] 创建 `core/database/repository/base.py`

```python
"""通用 Repository 基类"""
from typing import Optional, List, TypeVar, Generic, Type
from sqlalchemy.orm import Session
from core.database.engine import get_session

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """泛型 CRUD 基类"""

    def __init__(self, model: Type[T], session: Optional[Session] = None):
        self.model = model
        self._session = session

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = get_session()
        return self._session

    def get_by_id(self, id: int) -> Optional[T]:
        return self.session.query(self.model).filter(self.model.id == id).first()

    def get_all(self, user_id: Optional[int] = None) -> List[T]:
        q = self.session.query(self.model)
        if user_id is not None and hasattr(self.model, "user_id"):
            q = q.filter(self.model.user_id == user_id)
        return q.all()

    def find_by(self, **filters) -> List[T]:
        return self.session.query(self.model).filter_by(**filters).all()

    def find_one(self, **filters) -> Optional[T]:
        return self.session.query(self.model).filter_by(**filters).first()

    def create(self, **kwargs) -> T:
        obj = self.model(**kwargs)
        self.session.add(obj)
        self.session.commit()
        self.session.refresh(obj)
        return obj

    def update(self, id: int, **kwargs) -> Optional[T]:
        obj = self.get_by_id(id)
        if obj is None:
            return None
        for key, value in kwargs.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        self.session.commit()
        return obj

    def delete(self, id: int) -> bool:
        obj = self.get_by_id(id)
        if obj is None:
            return False
        self.session.delete(obj)
        self.session.commit()
        return True

    def count(self, user_id: Optional[int] = None) -> int:
        q = self.session.query(self.model)
        if user_id is not None and hasattr(self.model, "user_id"):
            q = q.filter(self.model.user_id == user_id)
        return q.count()

    def bulk_create(self, items: List[dict]) -> List[T]:
        objs = [self.model(**item) for item in items]
        self.session.add_all(objs)
        self.session.commit()
        return objs
```

- [ ] 创建各 Repository 子类文件

```python
# core/database/repository/users.py
from core.database.models import User, UserProfile
from core.database.repository.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, session=None):
        super().__init__(User, session)

    def get_by_username(self, username: str):
        return self.find_one(username=username)


class UserProfileRepository(BaseRepository[UserProfile]):
    def __init__(self, session=None):
        super().__init__(UserProfile, session)

    def get_by_user_id(self, user_id: int):
        return self.find_one(user_id=user_id)
```

```python
# core/database/repository/chat.py
from core.database.models import ChatSession, ChatMessage
from core.database.repository.base import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    def __init__(self, session=None):
        super().__init__(ChatSession, session)


class ChatMessageRepository(BaseRepository[ChatMessage]):
    def __init__(self, session=None):
        super().__init__(ChatMessage, session)
```

```python
# core/database/repository/planting.py
from core.database.models import PlantingPlan, PlantingTask
from core.database.repository.base import BaseRepository


class PlantingPlanRepository(BaseRepository[PlantingPlan]):
    def __init__(self, session=None):
        super().__init__(PlantingPlan, session)


class PlantingTaskRepository(BaseRepository[PlantingTask]):
    def __init__(self, session=None):
        super().__init__(PlantingTask, session)

    def get_by_plan(self, plan_id: int):
        return self.find_by(plan_id=plan_id)
```

```python
# core/database/repository/finance.py
from core.database.models import FinanceRecord
from core.database.repository.base import BaseRepository


class FinanceRepository(BaseRepository[FinanceRecord]):
    def __init__(self, session=None):
        super().__init__(FinanceRecord, session)

    def get_by_date_range(self, user_id: int, start_date, end_date):
        return self.session.query(FinanceRecord).filter(
            FinanceRecord.user_id == user_id,
            FinanceRecord.date >= start_date,
            FinanceRecord.date <= end_date,
        ).all()
```

```python
# core/database/repository/fields.py
from core.database.models import Field
from core.database.repository.base import BaseRepository


class FieldRepository(BaseRepository[Field]):
    def __init__(self, session=None):
        super().__init__(Field, session)
```

```python
# core/database/repository/devices.py
from core.database.models import DeviceConfig, DeviceRule, DeviceActionLog
from core.database.repository.base import BaseRepository


class DeviceConfigRepository(BaseRepository[DeviceConfig]):
    def __init__(self, session=None):
        super().__init__(DeviceConfig, session)

    def get_by_device_id(self, device_id: str):
        return self.find_one(device_id=device_id)


class DeviceRuleRepository(BaseRepository[DeviceRule]):
    def __init__(self, session=None):
        super().__init__(DeviceRule, session)


class DeviceLogRepository(BaseRepository[DeviceActionLog]):
    def __init__(self, session=None):
        super().__init__(DeviceActionLog, session)

    def get_recent(self, user_id: int, limit: int = 100):
        return self.session.query(DeviceActionLog).filter(
            DeviceActionLog.user_id == user_id
        ).order_by(DeviceActionLog.created_at.desc()).limit(limit).all()
```

```python
# core/database/repository/reminders.py
from core.database.models import Reminder
from core.database.repository.base import BaseRepository


class ReminderRepository(BaseRepository[Reminder]):
    def __init__(self, session=None):
        super().__init__(Reminder, session)

    def get_active(self, user_id: int):
        return self.find_by(user_id=user_id, status="active")
```

```python
# core/database/repository/disease.py
from core.database.models import DiseaseRisk
from core.database.repository.base import BaseRepository


class DiseaseRiskRepository(BaseRepository[DiseaseRisk]):
    def __init__(self, session=None):
        super().__init__(DiseaseRisk, session)
```

```python
# core/database/repository/inspection.py
from core.database.models import InspectionReport
from core.database.repository.base import BaseRepository


class InspectionRepository(BaseRepository[InspectionReport]):
    def __init__(self, session=None):
        super().__init__(InspectionReport, session)
```

- [ ] 更新 `core/database/__init__.py`

```python
"""数据库层 - SQLAlchemy ORM + Repository 模式"""
from core.database.engine import get_session, init_db, Session
from core.database.models import Base
from core.database.repository.users import UserRepository, UserProfileRepository
from core.database.repository.chat import ChatSessionRepository, ChatMessageRepository
from core.database.repository.planting import PlantingPlanRepository, PlantingTaskRepository
from core.database.repository.finance import FinanceRepository
from core.database.repository.fields import FieldRepository
from core.database.repository.devices import DeviceConfigRepository, DeviceRuleRepository, DeviceLogRepository
from core.database.repository.reminders import ReminderRepository
from core.database.repository.disease import DiseaseRiskRepository
from core.database.repository.inspection import InspectionRepository
```

- [ ] 提交

```bash
git add core/database/
git commit -m "feat: 搭建数据库基础设施(SQLAlchemy ORM + Repository模式)"
```

---

### Task 8: 编写 JSON → SQLite 迁移脚本

**文件:**
- 创建: `scripts/migrate_json_to_sqlite.py`

- [ ] 编写幂等迁移脚本

```python
#!/usr/bin/env python3
"""一次性 JSON → SQLite 数据迁移脚本。幂等：已存在数据则跳过。"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.database.engine import init_db, Session
from core.database.models import (
    User, UserProfile, ChatSession, ChatMessage,
    PlantingPlan, PlantingTask, Reminder, Field,
    FinanceRecord, DeviceConfig, DeviceRule, DeviceActionLog,
    DiseaseRisk, InspectionReport, Base
)
from sqlalchemy import inspect

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def should_skip(table_name):
    """检查表是否已有数据"""
    session = Session()
    try:
        model = Base.metadata.tables.get(table_name)
        if model is None:
            return False
        count = session.query(model).count()
        return count > 0
    finally:
        session.close()


def migrate_users():
    if should_skip("users"):
        print("  users: 已有数据，跳过")
        return 0
    path = os.path.join(DATA_DIR, "users.json")
    if not os.path.exists(path):
        print("  users.json: 文件不存在，跳过")
        return 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    session = Session()
    count = 0
    try:
        for username, info in data.items():
            user = User(username=username, password_hash=info.get("password", ""))
            session.add(user)
            session.flush()
            count += 1
        session.commit()
    finally:
        session.close()
    print(f"  users: {count} 条")
    return count


def migrate_user_profiles():
    if should_skip("user_profiles"):
        print("  user_profiles: 已有数据，跳过")
        return 0
    path = os.path.join(DATA_DIR, "user_profile.json")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    session = Session()
    count = 0
    try:
        for username, profile in data.items():
            user = session.query(User).filter(User.username == username).first()
            if not user:
                continue
            up = UserProfile(
                user_id=user.id,
                region=profile.get("region", ""),
                soil_type=profile.get("soil_type", ""),
                farm_size=profile.get("farm_size"),
                experience_level=profile.get("experience", ""),
                goals=json.dumps(profile.get("goals", []), ensure_ascii=False),
                phone=profile.get("phone", ""),
            )
            session.add(up)
            count += 1
        session.commit()
    finally:
        session.close()
    print(f"  user_profiles: {count} 条")
    return count


def migrate_finance():
    if should_skip("finance_records"):
        print("  finance_records: 已有数据，跳过")
        return 0
    session = Session()
    count = 0
    # 遍历 data/{username}/finance_costs.json 和 finance_income.json
    for username_dir in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, username_dir)
        if not os.path.isdir(user_dir):
            continue
        user = session.query(User).filter(User.username == username_dir).first()
        if not user:
            continue
        for fname, rtype in [("finance_costs.json", "cost"), ("finance_income.json", "income")]:
            path = os.path.join(user_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                fr = FinanceRecord(
                    user_id=user.id,
                    date=datetime.fromisoformat(r.get("date", "2000-01-01")).date() if r.get("date") else None,
                    crop=r.get("crop", ""),
                    plot=r.get("plot", ""),
                    record_type=rtype,
                    category=r.get("cost_type") or r.get("income_type", ""),
                    item_name=r.get("item_name", ""),
                    quantity=r.get("quantity"),
                    unit=r.get("unit", ""),
                    unit_price=r.get("unit_price"),
                    total_amount=r.get("total_amount", 0),
                    buyer=r.get("buyer", ""),
                    notes=r.get("notes", ""),
                )
                session.add(fr)
                count += 1
    session.commit()
    session.close()
    print(f"  finance_records: {count} 条")
    return count


def migrate_device_configs():
    if should_skip("device_configs"):
        print("  device_configs: 已有数据，跳过")
        return 0
    session = Session()
    count = 0
    for username_dir in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, username_dir)
        if not os.path.isdir(user_dir):
            continue
        user = session.query(User).filter(User.username == username_dir).first()
        if not user:
            continue
        path = os.path.join(user_dir, "custom_devices.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            devices = json.load(f)
        for d in devices:
            dc = DeviceConfig(
                user_id=user.id,
                device_id=d.get("device_id", ""),
                name=d.get("name", ""),
                driver=d.get("driver", "simulator"),
                capabilities=json.dumps(d.get("capabilities", []), ensure_ascii=False),
                sensors=json.dumps(d.get("sensors", []), ensure_ascii=False),
                connection=json.dumps(d.get("connection", {}), ensure_ascii=False),
                location=d.get("location", ""),
                plot_id=d.get("plot_id"),
                initial_state=json.dumps(d.get("initial_state", {}), ensure_ascii=False),
            )
            session.add(dc)
            count += 1
    session.commit()
    session.close()
    print(f"  device_configs: {count} 条")
    return count


def migrate_device_rules():
    if should_skip("device_rules"):
        print("  device_rules: 已有数据，跳过")
        return 0
    session = Session()
    count = 0
    for username_dir in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, username_dir)
        if not os.path.isdir(user_dir):
            continue
        user = session.query(User).filter(User.username == username_dir).first()
        if not user:
            continue
        path = os.path.join(user_dir, "device_rules.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rules = data.get("rules", []) if isinstance(data, dict) else data
        for rule in rules:
            dr = DeviceRule(
                user_id=user.id,
                name=rule.get("name", ""),
                enabled=1 if rule.get("enabled", True) else 0,
                conditions=json.dumps(rule.get("trigger", {}).get("conditions", []), ensure_ascii=False),
                actions=json.dumps(rule.get("action", {}), ensure_ascii=False),
                constraints=json.dumps(rule.get("constraints", {}), ensure_ascii=False),
            )
            session.add(dr)
            count += 1
    session.commit()
    session.close()
    print(f"  device_rules: {count} 条")
    return count


def migrate_all():
    print("开始迁移 JSON → SQLite...")
    init_db()
    total = 0
    total += migrate_users()
    total += migrate_user_profiles()
    total += migrate_finance()
    total += migrate_device_configs()
    total += migrate_device_rules()
    print(f"迁移完成！共 {total} 条记录。")


if __name__ == "__main__":
    migrate_all()
```

- [ ] 提交

```bash
git add scripts/migrate_json_to_sqlite.py
git commit -m "feat: JSON→SQLite数据迁移脚本(幂等)"
```

---

### Task 9: 重写 chat_history.py 适配数据库

**文件:**
- 修改: `core/chat_history.py`

- [ ] 将 ChatHistoryStore 改为使用 ChatSessionRepository + ChatMessageRepository

核心改动：`load_sessions()` → `ChatSessionRepository.get_all(user_id)` + 预加载 messages relationship，`save_sessions()` → 逐条 upsert。

```python
# 在 ChatHistoryStore.__init__ 中
from core.database.repository.chat import ChatSessionRepository, ChatMessageRepository
from core.database.engine import Session

# load_sessions 改为从 DB 加载
# save_sessions 改为写入 DB
# 保留 JSON 兼容读取作为迁移 fallback
```

- [ ] 提交

```bash
git add core/chat_history.py
git commit -m "refactor: chat_history适配SQLite数据库"
```

---

### Task 10-16: 重写其余数据模块适配数据库

按相同模式依次重写以下模块（每个模块一个commit）：

**Task 10:** `core/finance_manager.py` → 使用 FinanceRepository
**Task 11:** `core/planting_tracker.py` → 使用 PlantingPlanRepository + PlantingTaskRepository
**Task 12:** `core/reminder_system.py` → 使用 ReminderRepository
**Task 13:** `core/map_manager.py` → 使用 FieldRepository
**Task 14:** `core/device_rule_engine.py` → 使用 DeviceRuleRepository + DeviceLogRepository
**Task 15:** `core/device_executor.py` → 使用 DeviceLogRepository
**Task 16:** `core/device_registry_factory.py` → 使用 DeviceConfigRepository

每个模块的重写模式一致：
1. 移除 `json.load` / `json.dump` + 文件路径拼接
2. 替换为对应 Repository 的 CRUD 方法
3. 保留旧 JSON 读取逻辑作为迁移 fallback（`if not db_rows: load_from_json_fallback()`）

---

### Task 17: 适配 API 层和前端

**文件:**
- 修改: `app/api_routes.py`
- 修改: `app/main.py`
- 修改: `app/views/*.py`

将 API 端点中直接读写 JSON 的代码替换为 Repository 调用：

```python
# 旧: with open("data/user_profile.json") as f: ...
# 新: profile_repo = UserProfileRepository(); profile = profile_repo.get_by_user_id(user_id)
```

- [ ] 提交

---

### Task 18: 更新启动脚本自动迁移

**文件:**
- 修改: `app/start.py`

- [ ] 在启动时添加数据库初始化和迁移检测

```python
# app/start.py 启动逻辑中添加:
def _ensure_database():
    """确保数据库已初始化并迁移"""
    from core.database.engine import init_db
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "agriculture.db")
    if not os.path.exists(db_path):
        print("初始化数据库...")
        init_db()
        print("检测JSON数据...")
        from scripts.migrate_json_to_sqlite import migrate_all
        migrate_all()
```

- [ ] 提交

---

### Task 19: 搭建 DL 模型接口基础设施

**文件:**
- 创建: `models/__init__.py`
- 创建: `models/base.py`
- 创建: `models/registry.py`
- 创建: `models/weights/.gitkeep`

- [ ] 创建 `models/__init__.py`

```python
"""深度学习模型推理接口

支持 ONNX Runtime 和 PyTorch 两种推理后端。
通过 ModelRegistry 注册中心统一管理模型的发现、加载和推理。
"""
from models.base import BaseModelBackend, ModelInfo, ModelInput, ModelOutput, Prediction, ModelCapability
from models.registry import ModelRegistry

# 可选依赖守卫
_ONNX_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import onnxruntime
    _ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torchvision
    _TORCH_AVAILABLE = True
except ImportError:
    pass

def is_onnx_available():
    return _ONNX_AVAILABLE

def is_torch_available():
    return _TORCH_AVAILABLE
```

- [ ] 创建 `models/base.py`

```python
"""DL模型接口的抽象基类与数据结构"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional


class ModelCapability(Enum):
    DISEASE_CLASSIFY = "disease_classify"
    CROP_IDENTIFY = "crop_identify"
    PEST_DETECT = "pest_detect"
    SEVERITY_ASSESS = "severity_assess"


@dataclass
class ModelInfo:
    model_id: str
    model_name: str
    backend_name: str                          # "onnx" | "torch"
    capability: ModelCapability
    model_path: str                            # 权重文件路径
    input_shape: Tuple[int, int, int] = (3, 224, 224)
    classes: List[str] = field(default_factory=list)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInput:
    image_bytes: bytes
    top_k: int = 3
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    class_name: str
    confidence: float
    index: int


@dataclass
class ModelOutput:
    success: bool
    model_id: str
    predictions: List[Prediction]
    inference_time_ms: float
    error_code: str = ""
    raw_output: Any = None

    @classmethod
    def error(cls, model_id: str, error_code: str, message: str = "") -> "ModelOutput":
        return cls(success=False, model_id=model_id, predictions=[], inference_time_ms=0, error_code=error_code)
```

- [ ] 创建 `models/registry.py`

```python
"""模型注册中心 - 管理多个推理后端和模型"""
import asyncio
import logging
from typing import Dict, List, Optional

from models.base import BaseModelBackend, ModelInfo, ModelInput, ModelOutput, ModelCapability

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self._backends: Dict[str, BaseModelBackend] = {}
        self._model_map: Dict[str, str] = {}        # model_id → backend_name
        self._model_info: Dict[str, ModelInfo] = {}  # model_id → ModelInfo

    def register(self, name: str, backend: BaseModelBackend):
        if name in self._backends:
            old = self._backends[name]
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                pass
        self._backends[name] = backend

    async def discover_all(self) -> int:
        new_map = {}
        new_info = {}
        for name, backend in self._backends.items():
            try:
                models = await asyncio.wait_for(backend.discover_models(), timeout=30)
                for model in models:
                    new_map[model.model_id] = name
                    new_info[model.model_id] = model
            except asyncio.TimeoutError:
                logger.warning("后端 %s 模型发现超时", name)
            except Exception as e:
                logger.error("后端 %s 模型发现失败: %s", name, e)
        self._model_map = new_map
        self._model_info = new_info
        return len(self._model_map)

    async def infer(self, model_id: str, input: ModelInput) -> ModelOutput:
        backend_name = self._model_map.get(model_id)
        if backend_name is None:
            return ModelOutput.error(model_id, "MODEL_NOT_FOUND", f"模型 {model_id} 未注册")
        backend = self._backends.get(backend_name)
        if backend is None:
            return ModelOutput.error(model_id, "BACKEND_NOT_FOUND")
        return await backend.infer(model_id, input)

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        return self._model_info.get(model_id)

    def list_models(self) -> List[ModelInfo]:
        return list(self._model_info.values())

    def get_models_by_capability(self, cap: ModelCapability) -> List[ModelInfo]:
        return [m for m in self._model_info.values() if m.capability == cap]

    def unregister(self, name: str):
        if name in self._backends:
            backend = self._backends.pop(name)
            self._model_map = {k: v for k, v in self._model_map.items() if v != name}
            self._model_info = {k: v for k, v in self._model_info.items() if v not in [m for m in self._model_info.values() if m.backend_name == name]}

    @property
    def backend_names(self) -> List[str]:
        return list(self._backends.keys())

    @property
    def model_count(self) -> int:
        return len(self._model_map)
```

- [ ] 提交

---

### Task 20: 实现 ONNX 推理后端

**文件:**
- 创建: `models/onnx_backend.py`
- 创建: `models/presets.py`

- [ ] 创建 `models/onnx_backend.py`

```python
"""ONNX Runtime 推理后端"""
import logging
import time
import os
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List

from models.base import BaseModelBackend, ModelInfo, ModelInput, ModelOutput, Prediction

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXBackend(BaseModelBackend):
    backend_name = "onnx"

    def __init__(self, device: str = "cpu"):
        self._sessions = {}     # model_id → InferenceSession
        self._models = {}       # model_id → ModelInfo
        self._device = device
        if device == "cuda":
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self._providers = ["CPUExecutionProvider"]

    async def load_model(self, model_info: ModelInfo) -> bool:
        if not ONNX_AVAILABLE:
            logger.error("onnxruntime 未安装")
            return False
        try:
            session = ort.InferenceSession(model_info.model_path, providers=self._providers)
            self._sessions[model_info.model_id] = session
            self._models[model_info.model_id] = model_info
            return True
        except Exception as e:
            logger.error("加载ONNX模型失败 %s: %s", model_info.model_id, e)
            return False

    async def unload_model(self, model_id: str) -> None:
        self._sessions.pop(model_id, None)
        self._models.pop(model_id, None)

    async def infer(self, model_id: str, input: ModelInput) -> ModelOutput:
        model_info = self._models.get(model_id)
        session = self._sessions.get(model_id)
        if model_info is None or session is None:
            return ModelOutput.error(model_id, "MODEL_NOT_LOADED")

        try:
            # 预处理
            image = Image.open(BytesIO(input.image_bytes)).convert("RGB")
            preprocess = model_info.preprocessing
            resize = preprocess.get("resize", model_info.input_shape[1:])  # (H, W)
            image = image.resize((resize[1], resize[0]) if isinstance(resize, (list, tuple)) else (resize, resize))

            img_array = np.array(image).astype(np.float32) / 255.0
            mean = np.array(preprocess.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std = np.array(preprocess.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
            img_array = (img_array - mean) / std
            img_array = img_array.transpose(2, 0, 1)  # HWC → CHW
            img_array = np.expand_dims(img_array, axis=0)  # (1, C, H, W)

            # 推理
            input_name = session.get_inputs()[0].name
            start = time.perf_counter()
            outputs = session.run(None, {input_name: img_array})
            elapsed_ms = (time.perf_counter() - start) * 1000

            # 后处理
            logits = outputs[0][0]
            top_k = min(input.top_k, len(model_info.classes))
            top_indices = np.argsort(logits)[::-1][:top_k]

            predictions = [
                Prediction(
                    class_name=model_info.classes[idx] if idx < len(model_info.classes) else f"class_{idx}",
                    confidence=float(logits[idx]),
                    index=int(idx),
                )
                for idx in top_indices
            ]

            return ModelOutput(
                success=True,
                model_id=model_id,
                predictions=predictions,
                inference_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error("ONNX推理失败 %s: %s", model_id, e)
            return ModelOutput.error(model_id, "INFERENCE_ERROR", str(e))

    async def discover_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    async def health_check(self) -> bool:
        return ONNX_AVAILABLE and len(self._sessions) > 0
```

- [ ] 创建 `models/presets.py`

```python
"""内置预训练模型配置预设"""
from models.base import ModelCapability

PRESETS = {
    "plant_village_wheat": {
        "model_name": "PlantVillage 小麦病害分类",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": ["健康", "条锈病", "叶锈病", "秆锈病", "白粉病", "赤霉病"],
        "input_shape": (3, 224, 224),
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [224, 224],
        },
        "preferred_backend": "onnx",
    },
    "plant_village_tomato": {
        "model_name": "PlantVillage 番茄病害分类",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": ["健康", "早疫病", "晚疫病", "叶霉病", "斑枯病", "细菌性斑点病", "黄化曲叶病", "花叶病毒病"],
        "input_shape": (3, 224, 224),
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [224, 224],
        },
        "preferred_backend": "onnx",
    },
    "plant_village_rice": {
        "model_name": "PlantVillage 水稻病害分类",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": ["健康", "稻瘟病", "纹枯病", "白叶枯病", "胡麻斑病", "恶苗病"],
        "input_shape": (3, 224, 224),
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [224, 224],
        },
        "preferred_backend": "onnx",
    },
}
```

- [ ] 提交

---

### Task 21: 实现 Torch 推理后端

**文件:**
- 创建: `models/torch_backend.py`

```python
"""PyTorch 推理后端"""
import logging
import time
from io import BytesIO
from typing import List

from PIL import Image
import numpy as np

from models.base import BaseModelBackend, ModelInfo, ModelInput, ModelOutput, Prediction

logger = logging.getLogger(__name__)

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TorchBackend(BaseModelBackend):
    backend_name = "torch"

    def __init__(self, device: str = "cpu"):
        self._models = {}       # model_id → (model_instance, ModelInfo)
        self._device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

    async def load_model(self, model_info: ModelInfo) -> bool:
        if not TORCH_AVAILABLE:
            logger.error("PyTorch 未安装")
            return False
        try:
            model = torch.load(model_info.model_path, map_location=self._device, weights_only=False)
            model.eval()
            self._models[model_info.model_id] = (model, model_info)
            return True
        except Exception as e:
            logger.error("加载PyTorch模型失败 %s: %s", model_info.model_id, e)
            return False

    async def unload_model(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    async def infer(self, model_id: str, input: ModelInput) -> ModelOutput:
        entry = self._models.get(model_id)
        if entry is None:
            return ModelOutput.error(model_id, "MODEL_NOT_LOADED")
        model, model_info = entry

        try:
            image = Image.open(BytesIO(input.image_bytes)).convert("RGB")
            preprocess = model_info.preprocessing
            resize = preprocess.get("resize", model_info.input_shape[1:])
            if isinstance(resize, (list, tuple)):
                resize = (resize[1], resize[0])
            else:
                resize = (resize, resize)

            mean = preprocess.get("mean", [0.485, 0.456, 0.406])
            std = preprocess.get("std", [0.229, 0.224, 0.225])

            transform = T.Compose([
                T.Resize(resize),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            tensor = transform(image).unsqueeze(0).to(self._device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000

            probs = torch.softmax(outputs[0], dim=0)
            top_k = min(input.top_k, len(model_info.classes))
            top_probs, top_indices = torch.topk(probs, top_k)

            predictions = [
                Prediction(
                    class_name=model_info.classes[idx] if idx < len(model_info.classes) else f"class_{idx}",
                    confidence=float(conf),
                    index=int(idx),
                )
                for conf, idx in zip(top_probs, top_indices)
            ]

            return ModelOutput(
                success=True,
                model_id=model_id,
                predictions=predictions,
                inference_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error("PyTorch推理失败 %s: %s", model_id, e)
            return ModelOutput.error(model_id, "INFERENCE_ERROR", str(e))

    async def discover_models(self) -> List[ModelInfo]:
        return [info for _, info in self._models.values()]

    async def health_check(self) -> bool:
        return TORCH_AVAILABLE and len(self._models) > 0
```

- [ ] 提交

---

### Task 22: 实现模型工厂 + 执行器

**文件:**
- 创建: `core/model_registry_factory.py`
- 创建: `core/model_executor.py`

- [ ] 创建 `core/model_registry_factory.py`

```python
"""模型注册中心工厂 - 初始化 + 缓存 + 自动发现"""
import os
import logging
from typing import Optional

from models.registry import ModelRegistry
from models.base import ModelInfo, ModelCapability
from models.onnx_backend import ONNXBackend, ONNX_AVAILABLE
from models.torch_backend import TorchBackend, TORCH_AVAILABLE
from models.presets import PRESETS

logger = logging.getLogger(__name__)

_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """获取全局模型注册中心（单例）"""
    global _model_registry
    if _model_registry is None:
        _model_registry = setup_model_registry()
    return _model_registry


def setup_model_registry() -> ModelRegistry:
    """初始化模型注册中心：注册后端 → 扫描权重目录 → 自动发现模型"""
    registry = ModelRegistry()
    backend_type = os.getenv("DL_BACKEND", "onnx")
    device = os.getenv("DL_DEVICE", "cpu")
    models_dir = os.getenv("DL_MODELS_DIR", "models/weights")
    default_model = os.getenv("DL_DEFAULT_MODEL", "")

    # 注册后端
    if backend_type in ("onnx",) and ONNX_AVAILABLE:
        registry.register("onnx", ONNXBackend(device=device))
        logger.info("ONNX后端已注册，设备: %s", device)
    if backend_type in ("torch",) and TORCH_AVAILABLE:
        registry.register("torch", TorchBackend(device=device))
        logger.info("Torch后端已注册，设备: %s", device)

    # 扫描预设对应的权重文件
    for preset_id, preset in PRESETS.items():
        backend = preset.get("preferred_backend", "onnx")
        ext = ".onnx" if backend == "onnx" else ".pt"
        weight_path = os.path.join(models_dir, f"{preset_id}{ext}")
        if os.path.exists(weight_path):
            info = ModelInfo(
                model_id=preset_id,
                model_name=preset["model_name"],
                backend_name=backend,
                capability=preset["capability"],
                model_path=os.path.abspath(weight_path),
                input_shape=preset.get("input_shape", (3, 224, 224)),
                classes=preset.get("classes", []),
                preprocessing=preset.get("preprocessing", {}),
            )
            # 异步加载模型
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import threading
                    def _load():
                        new_loop = asyncio.new_event_loop()
                        new_loop.run_until_complete(_load_model(registry, backend, info))
                        new_loop.close()
                    threading.Thread(target=_load, daemon=True).start()
                else:
                    loop.run_until_complete(_load_model(registry, backend, info))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_load_model(registry, backend, info))
                loop.close()

    return registry


async def _load_model(registry: ModelRegistry, backend_name: str, info: ModelInfo):
    backend = registry._backends.get(backend_name)
    if backend:
        success = await backend.load_model(info)
        if success:
            registry._model_map[info.model_id] = backend_name
            registry._model_info[info.model_id] = info
            logger.info("模型已加载: %s (%s)", info.model_id, info.model_name)
```

- [ ] 创建 `core/model_executor.py`

```python
"""模型推理执行器 - 重试 + 超时 + 日志"""
import time
import logging
import asyncio
from typing import Optional

from models.base import ModelInput, ModelOutput
from models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelExecutor:
    def __init__(self, registry: ModelRegistry, max_retries: int = 2, timeout_ms: int = 30000):
        self.registry = registry
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms

    async def infer(self, model_id: str, input: ModelInput) -> ModelOutput:
        last_result = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self.registry.infer(model_id, input),
                    timeout=self.timeout_ms / 1000,
                )
                if result.success:
                    return result
                last_result = result
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning("模型推理失败(attempt %d/%d): %s, %d秒后重试", attempt + 1, self.max_retries + 1, result.error_code, wait)
                    await asyncio.sleep(wait)
            except asyncio.TimeoutError:
                logger.error("模型推理超时(attempt %d/%d): %s", attempt + 1, self.max_retries + 1, model_id)
                last_result = ModelOutput.error(model_id, "TIMEOUT")
        return last_result or ModelOutput.error(model_id, "MAX_RETRIES_EXCEEDED")

    def infer_sync(self, model_id: str, input: ModelInput) -> ModelOutput:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.infer(model_id, input))
            loop.close()
            return result
        return loop.run_until_complete(self.infer(model_id, input))
```

- [ ] 提交

---

### Task 23: 重写 image_analysis.py 使用本地模型

**文件:**
- 修改: `app/agent/nodes/image_analysis.py`

- [ ] 将 `_call_vision_api()` 替换为 `_call_dl_model()` + LLM 增强

```python
# 核心改动：不再调用 Vision API，改为本地模型分类 + LLM 生成回答

def _call_dl_model(state) -> dict:
    """使用本地DL模型进行病虫害分类"""
    from core.model_registry_factory import get_model_registry
    from core.model_executor import ModelExecutor
    from models.base import ModelInput

    registry = get_model_registry()
    executor = ModelExecutor(registry)

    image_bytes = base64.b64decode(state.image_data)
    input = ModelInput(image_bytes=image_bytes, top_k=3)
    result = executor.infer_sync(os.getenv("DL_DEFAULT_MODEL", ""), input)

    if not result.success:
        return {"error": f"模型推理失败: {result.error_code}"}

    return {
        "model_id": result.model_id,
        "predictions": [
            {"class_name": p.class_name, "confidence": round(p.confidence, 4)}
            for p in result.predictions
        ],
        "inference_time_ms": result.inference_time_ms,
    }


def image_analysis_node(state: AgentState) -> AgentState:
    """图片分析节点 — DL模型分类 + LLM增强"""
    model_result = _call_dl_model(state)
    if "error" in model_result:
        state.image_analysis_result = model_result
        state.final_answer = "图片分析未能完成，请稍后重试。"
        return state

    # 将分类结果注入LLM prompt生成丰富回答
    top_prediction = model_result["predictions"][0]
    prompt = f"""根据图像识别结果，作物被诊断为：{top_prediction['class_name']}（置信度 {top_prediction['confidence']}）。

请提供以下信息：
1. 该病害/虫害的详细描述
2. 防治方法和用药建议
3. 预防措施
4. 对当前作物生长阶段的影响"""

    # LLM生成防治建议（使用对话模型，非Vision）
    llm_response = _invoke_llm(prompt, state)
    state.image_analysis_result = {**model_result, "llm_advice": llm_response}
    return state
```

- [ ] 提交

---

### Task 24: 删除 Vision 配置 + 更新 .env 模板

**文件:**
- 修改: `app/agent/config.py`
- 修改: `.env.template`
- 修改: `requirements.txt`

- [ ] 在 `config.py` 中删除以下配置项：

```python
# 删除:
VISION_API_KEY = os.getenv("VISION_API_KEY") or LLM_API_KEY
VISION_BASE_URL = os.getenv("VISION_BASE_URL") or LLM_BASE_URL
VISION_MODEL = os.getenv("VISION_MODEL") or LLM_MODEL
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE") or 0.3)
ENABLE_IMAGE_ANALYSIS = bool(VISION_MODEL)
```

```python
# 新增:
DL_BACKEND = os.getenv("DL_BACKEND", "onnx")
DL_MODELS_DIR = os.getenv("DL_MODELS_DIR", "models/weights")
DL_DEVICE = os.getenv("DL_DEVICE", "cpu")
DL_DEFAULT_MODEL = os.getenv("DL_DEFAULT_MODEL", "")
```

- [ ] 更新 `.env.template`

```env
# ===== LLM 对话模型（必填）=====
LLM_API_KEY=sk-your-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# ===== Embedding 向量模型（可选）=====
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=text-embedding-3-small

# ===== 天气服务（可选）=====
WEATHER_API_PROVIDER=qweather
WEATHER_API_KEY=

# ===== 腾讯云短信（可选）=====
SMS_SECRET_ID=
SMS_SECRET_KEY=
SMS_SDK_APP_ID=
SMS_SIGN_NAME=
SMS_TEMPLATE_ID=
SMS_REGION=ap-guangzhou

# ===== 深度学习模型（本地推理，替代 Vision API）=====
DL_BACKEND=onnx                    # onnx | torch
DL_MODELS_DIR=models/weights       # 模型权重文件目录
DL_DEVICE=cpu                      # cpu | cuda
DL_DEFAULT_MODEL=plant_village_wheat  # 默认病虫害分类模型
```

- [ ] 更新 `requirements.txt`

```diff
+ sqlalchemy>=2.0.0
+ alembic>=1.13.0
+ onnxruntime>=1.17.0
+ torch>=2.0.0
+ torchvision>=0.15.0
+ Pillow>=10.0.0
+ numpy>=1.24.0
```

- [ ] 提交

---

### Task 25: 适配 crop_monitor_agent.py 使用本地模型

**文件:**
- 修改: `app/agent/agents/crop_monitor_agent.py`

将 `_call_vision_api()` 替换为 `_call_dl_model()`（与 Task 23 相同模式）

- [ ] 提交

---

### Task 26: 最终验证与集成测试

- [ ] 运行所有单元测试

```bash
python -m pytest tests/ -v --ignore=tests/__pycache__
```

- [ ] 验证数据库初始化

```bash
python -c "from core.database.engine import init_db; init_db(); print('OK')"
```

- [ ] 验证迁移脚本

```bash
python scripts/migrate_json_to_sqlite.py
```

- [ ] 检查导入

```bash
python -c "from models import ModelRegistry, ModelCapability; print('OK')"
python -c "from core.database import UserRepository; print('OK')"
```

- [ ] 提交最终改动并推送

```bash
git add -A
git commit -m "feat: 完成数据库迁移+DL模型接口+bug修复"
git push origin master
```
