# 数据库迁移 + 深度学习模型接口 设计文档

## 概述

将项目数据存储从 JSON 文件迁移到 SQLite 数据库，并为本地深度学习图像分类模型（病虫害识别）提供标准化的接口抽象，替代当前的多模态 LLM Vision API 方案。

## 一、数据库设计

### 1.1 技术选型

| 项 | 选择 |
|---|---|
| 数据库 | SQLite |
| ORM | SQLAlchemy 2.0 |
| 迁移工具 | Alembic |
| 存储位置 | `data/agriculture.db` |

### 1.2 数据表设计 (13 张表)

```sql
-- 用户账户
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户档案 (替代 user_profile.json)
CREATE TABLE user_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE REFERENCES users(id),
    region VARCHAR(100),
    soil_type VARCHAR(50),
    farm_size REAL,
    experience_level VARCHAR(20),
    goals TEXT,            -- JSON array
    phone VARCHAR(20),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 对话会话 (替代 chat_history.json)
CREATE TABLE chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,    -- user / assistant / system
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 种植计划 (替代 planting_progress.json)
CREATE TABLE planting_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    crop VARCHAR(50) NOT NULL,
    plot_id INTEGER REFERENCES fields(id),
    stage VARCHAR(50),
    stage_number INTEGER,
    total_stages INTEGER,
    start_date DATE,
    expected_end_date DATE,
    actual_end_date DATE,
    progress_percent REAL DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 农事任务 (替代 planting_tasks.json)
CREATE TABLE planting_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    plan_id INTEGER REFERENCES planting_plans(id),
    crop VARCHAR(50),
    task_type VARCHAR(50),
    title VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    priority VARCHAR(10) DEFAULT 'normal',
    start_date DATE,
    end_date DATE,
    completed_date DATE,
    device_id VARCHAR(100),
    device_command VARCHAR(100),
    device_params TEXT,     -- JSON
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 提醒 (替代 reminders.json)
CREATE TABLE reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    crop VARCHAR(50),
    reminder_type VARCHAR(50) NOT NULL,
    task_description TEXT,
    growth_stage VARCHAR(50),
    frequency VARCHAR(20) DEFAULT 'once',
    interval_days INTEGER,
    time_of_day TIME,
    advance_hours INTEGER DEFAULT 0,
    channels TEXT,          -- JSON array
    status VARCHAR(20) DEFAULT 'active',
    last_triggered TIMESTAMP,
    next_trigger TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 地块 (替代 fields.json)
CREATE TABLE fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    coordinates TEXT NOT NULL,  -- JSON: [[lon,lat], ...]
    center_lat REAL,
    center_lon REAL,
    area_mu REAL,
    area_m2 REAL,
    soil_type VARCHAR(50),
    current_crop VARCHAR(50),
    planting_history TEXT,      -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 财务记录 (替代 finance_costs.json + finance_income.json)
CREATE TABLE finance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    date DATE NOT NULL,
    crop VARCHAR(50),
    plot VARCHAR(100),
    record_type VARCHAR(10) NOT NULL CHECK (record_type IN ('income', 'cost')),
    category VARCHAR(50),
    item_name VARCHAR(200) NOT NULL,
    quantity REAL,
    unit VARCHAR(20),
    unit_price REAL,
    total_amount REAL NOT NULL,
    buyer VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 设备配置 (替代 custom_devices.json)
CREATE TABLE device_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    device_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200),
    driver VARCHAR(50) NOT NULL,
    capabilities TEXT,      -- JSON array
    sensors TEXT,           -- JSON array
    connection TEXT,        -- JSON: {host, port, ...}
    location VARCHAR(200),
    plot_id INTEGER,
    initial_state TEXT,     -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 设备规则 (替代 device_rules.json)
CREATE TABLE device_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(200) NOT NULL,
    enabled INTEGER DEFAULT 1,
    conditions TEXT NOT NULL,    -- JSON
    actions TEXT NOT NULL,       -- JSON
    constraints TEXT,            -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 设备操作日志 (替代 device_log.json + device_pending.json)
CREATE TABLE device_action_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    device_id VARCHAR(100) NOT NULL,
    command VARCHAR(100) NOT NULL,
    params TEXT,                -- JSON
    trigger VARCHAR(50),        -- manual / rule / autonomous / sensor
    rule_id INTEGER REFERENCES device_rules(id),
    decision VARCHAR(20) DEFAULT 'auto',  -- auto / confirmed / rejected
    status VARCHAR(20) DEFAULT 'pending', -- pending / executing / success / failed
    success INTEGER DEFAULT 1,
    attempts INTEGER DEFAULT 1,
    message TEXT,
    error_code VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 病虫害风险 (替代 disease_risks.json)
CREATE TABLE disease_risks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop VARCHAR(50) NOT NULL,
    disease VARCHAR(100) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,  -- high / medium / low
    score REAL,
    matched_conditions TEXT,          -- JSON
    advice TEXT,
    assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 巡检报告 (替代 autonomous_reports/ + photos/)
CREATE TABLE inspection_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    device_id VARCHAR(100),
    cycle_id VARCHAR(100),
    photo_path VARCHAR(500),
    analysis_result TEXT,       -- JSON: AI 分析结果
    crop_type VARCHAR(50),
    health_status VARCHAR(50),
    issues_found TEXT,          -- JSON
    actions_taken TEXT,         -- JSON
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 1.3 数据访问层架构

```
core/database/
├── __init__.py              # 导出 public API
├── engine.py                # create_engine + sessionmaker + get_session
├── models.py                # 全部 ORM Model 定义
├── repository/
│   ├── __init__.py
│   ├── base.py              # BaseRepository[T] 泛型基类
│   ├── users.py             # UserRepository
│   ├── planting.py          # PlantingPlanRepository, PlantingTaskRepository
│   ├── finance.py           # FinanceRepository
│   ├── fields.py            # FieldRepository
│   ├── devices.py           # DeviceConfigRepository, DeviceRuleRepository, DeviceLogRepository
│   ├── chat.py              # ChatRepository
│   ├── reminders.py         # ReminderRepository
│   ├── disease.py           # DiseaseRiskRepository
│   └── inspection.py        # InspectionRepository
└── migrations/              # Alembic 迁移目录
```

BaseRepository 提供通用方法：
- `get_by_id(id) -> Optional[T]`
- `get_all(user_id) -> List[T]`
- `create(**kwargs) -> T`
- `update(id, **kwargs) -> T`
- `delete(id) -> bool`
- `find_by(**filters) -> List[T]`

### 1.4 迁移策略

1. 新 `scripts/migrate_json_to_sqlite.py`：读取 `data/` 下所有 JSON → 逐表写入 SQLite
2. `app/start.py` 启动时：
   - `data/agriculture.db` 不存在 → `Base.metadata.create_all()` + 尝试 JSON 迁移
   - 已存在 → 跳过
3. 旧 JSON 文件保留不删（备份），后续读写全部走 SQLite
4. 环境变量 `DATA_STORAGE_DIR` 改为数据库路径（不再用于 JSON 目录），默认 `"data/agriculture.db"`

---

## 二、深度学习模型接口

### 2.1 整体架构

复用 `devices/` 目录的成熟抽象模式：

```
models/
├── __init__.py              # 导出 + try/except 可选依赖守卫
├── base.py                  # BaseModelBackend, ModelInfo, ModelInput, ModelOutput, Prediction
├── registry.py              # ModelRegistry 注册中心
├── onnx_backend.py          # ONNX Runtime 推理后端
├── torch_backend.py         # PyTorch 推理后端
└── presets.py               # 内置预训练模型配置

core/
├── model_registry_factory.py  # 工厂（初始化 → 注册 → 缓存），与 device_registry_factory 对称
└── model_executor.py          # 执行器（重试/超时/审计日志），与 device_executor 对称
```

权重文件目录：`models/weights/`（`.onnx` / `.pt` 文件，由 `.gitignore` 忽略）

### 2.2 核心数据结构

```python
class ModelCapability(Enum):
    DISEASE_CLASSIFY = "disease_classify"
    CROP_IDENTIFY = "crop_identify"
    PEST_DETECT = "pest_detect"
    SEVERITY_ASSESS = "severity_assess"

@dataclass
class ModelInfo:
    model_id: str                     # 如 "resnet50_wheat_rust"
    model_name: str                   # "ResNet50 小麦条锈病分类"
    backend_name: str                 # "onnx" | "torch"
    capability: ModelCapability
    model_path: str                   # 权重文件路径
    input_shape: Tuple[int,int,int]   # (C, H, W)
    classes: List[str]                # 分类标签
    preprocessing: Dict[str, Any]     # mean, std, resize 等
    metadata: Dict[str, Any]          # 框架/量化/训练集 等扩展信息

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
```

### 2.3 抽象基类

```python
class BaseModelBackend(ABC):
    backend_name: str = "base"

    @abstractmethod
    async def load_model(self, model_info: ModelInfo) -> bool: ...
    @abstractmethod
    async def unload_model(self, model_id: str) -> None: ...
    @abstractmethod
    async def infer(self, model_id, input: ModelInput) -> ModelOutput: ...
    @abstractmethod
    async def discover_models(self) -> List[ModelInfo]: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
```

### 2.4 两个后端

| | ONNXBackend | TorchBackend |
|---|---|---|
| 依赖 | `onnxruntime` / `onnxruntime-gpu` | `torch` + `torchvision` |
| 加载 | `ort.InferenceSession(path)` | `torch.load()` + `model.eval()` |
| 推理 | `session.run([output_name], input)` | `model(tensor)` |
| 预处理 | numpy (PIL → resize → normalize → CHW) | torchvision.transforms |
| 设备检测 | 自动 CUDA/CPU | `torch.cuda.is_available()` |
| 模型文件 | `*.onnx` | `*.pt` / `*.pth` |
| GPU推理 | `onnxruntime-gpu` 自动分配 | `.to("cuda")` |

### 2.5 注册中心

```python
class ModelRegistry:
    _backends: Dict[str, BaseModelBackend]
    _model_map: Dict[str, str]              # model_id → backend_name

    async def register(self, name, backend)
    async def unregister(self, name)
    async def discover_all(self) -> int     # 返回发现模型数
    async def infer(self, model_id, input) -> ModelOutput
    def get_model_info(self, model_id) -> Optional[ModelInfo]
    def list_models(self) -> List[ModelInfo]
    def get_models_by_capability(self, cap) -> List[ModelInfo]
```

### 2.6 内置模型预设

`presets.py` 提供可直接使用的模型配置：

```python
PRESETS = {
    "plant_village_wheat": {
        "model_name": "PlantVillage 小麦病害分类",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": ["健康", "条锈病", "叶锈病", "秆锈病", "白粉病", "赤霉病"],
        "input_shape": (3, 224, 224),
        "preprocessing": {"mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225]},
        "preferred_backend": "onnx",
    },
    # 更多预设...
}
```

用户将对应权重文件放入 `models/weights/` 即可使用，也可自行添加自定义预设配置。

### 2.7 与现有病虫害流程的集成

```
原来:
  用户上传图片 → Vision API (多模态LLM) → JSON结果 → LLM润色 → 回答

改为:
  用户上传图片 → ModelRegistry.infer() → 本地模型分类
       ↓
  Prediction: {class_name:"条锈病", confidence:0.93}
       ↓
  注入 LLM prompt (非Vision): "模型识别结果为小麦条锈病(置信度93%)，
  请根据以下知识库内容提供防治建议..."
       ↓
  LLM + RAG → 生成防治建议、施药方案、农事提醒
```

改动涉及的文件：
- `app/agent/nodes/image_analysis.py` — 核心重写：`_call_vision_api()` → `_call_dl_model()` + LLM 增强
- `app/agent/agents/crop_monitor_agent.py` — 同上
- `app/agent/agents/disease_agent.py` — 分析流程串联适配
- `core/autonomous_farm_manager.py` — 自主巡检适配

### 2.8 应用启动流程

`app/start.py` 启动时：

1. 检测 `data/agriculture.db` → 不存在则建表 + JSON 迁移
2. 初始化 `ModelRegistry` → 注册 ONNX/Torch 后端
3. 扫描 `models/weights/` + 预设配置 → 自动发现可用模型
4. 按需 `load_model()` → 首次推理时自动加载

---

## 三、配置变更

### 3.1 新的 `.env` 配置

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

### 3.2 删除的配置项

```
VISION_API_KEY          — 不再需要多模态 Vision API
VISION_BASE_URL         — 同上
VISION_MODEL            — 同上
VISION_TEMPERATURE      — 同上
ENABLE_IMAGE_ANALYSIS   — 改为检查 DL 模型是否可用
```

### 3.3 代码清理

| 文件 | 改动 |
|------|------|
| `app/agent/config.py` | 删 `VISION_*` + `ENABLE_IMAGE_ANALYSIS`，加 `DL_*` 配置项 |
| `app/agent/nodes/image_analysis.py` | `_call_vision_api()` → `_call_dl_model()` |
| `app/agent/agents/crop_monitor_agent.py` | Vision API 调用 → `ModelRegistry` |
| `app/api_routes.py` | `/api/diagnose/vision` → `/api/diagnose/dl-model` |
| `.env.template` | 更新配置模板 |

---

## 四、改动范围总览

### 新增模块 (7 个)

| 模块 | 路径 |
|------|------|
| 数据库层 | `core/database/` (engine, models, repository/*, migrations) |
| DL 模型接口 | `models/` (base, registry, onnx_backend, torch_backend, presets) |
| 模型工厂 | `core/model_registry_factory.py` |
| 模型执行器 | `core/model_executor.py` |
| 迁移脚本 | `scripts/migrate_json_to_sqlite.py` |
| 权重目录 | `models/weights/.gitkeep` |

### 重写模块 (12 个)

| 模块 | 说明 |
|------|------|
| `core/chat_history.py` | JSON → SQLAlchemy Repository |
| `core/finance_manager.py` | JSON → SQLAlchemy Repository |
| `core/planting_tracker.py` | JSON → SQLAlchemy Repository |
| `core/planting_planner.py` | JSON → SQLAlchemy Repository |
| `core/reminder_system.py` | JSON → SQLAlchemy Repository |
| `core/map_manager.py` | JSON → SQLAlchemy Repository |
| `core/device_rule_engine.py` | JSON → SQLAlchemy Repository |
| `core/device_executor.py` | JSON → SQLAlchemy Repository |
| `core/device_registry_factory.py` | JSON → SQLAlchemy Repository |
| `core/plot_manager.py` | JSON → SQLAlchemy Repository |
| `core/autonomous_farm_manager.py` | JSON → SQL + DL 模型适配 |
| `app/agent/nodes/image_analysis.py` | Vision API → 本地模型 + LLM 增强 |

### 修改模块 (10 个)

| 模块 | 说明 |
|------|------|
| `app/agent/config.py` | 配置项替换 |
| `app/agent/agents/disease_agent.py` | 模型→LLM 串联 |
| `app/agent/agents/crop_monitor_agent.py` | Vision→本地模型 |
| `app/api_routes.py` | 适配数据库 + 诊断端点 |
| `app/scheduler_jobs.py` | 适配数据库 |
| `app/main.py` | 适配数据库 |
| `app/views/*.py` | 适配新 API 格式 |
| `app/start.py` | 启动时自动迁移 + 模型初始化 |
| `.env.template` | 更新配置 |
| `requirements.txt` | 加依赖 (sqlalchemy, alembic, onnxruntime) |

### 不变模块 (8 个)

| 模块 | 说明 |
|------|------|
| `devices/` | 设备驱动完全不动 |
| `knowledge/` | 知识库完全不动 |
| `core/lunar_calendar.py` | 农历计算不动 |
| `core/weather_service.py` | 天气服务不动 |
| `core/disease_risk.py` | 病害风险评估不动 |
| `core/spray_advisor.py` | 施药建议不动 |
| `core/crop_comparison.py` | 作物对比不动 |
| `core/tts_components.py` | 语音组件不动 |

---

## 五、依赖变更

```diff
# requirements.txt 新增
+ sqlalchemy>=2.0.0
+ alembic>=1.13.0
+ onnxruntime>=1.17.0         # ONNX 推理后端
+ torch>=2.0.0                # PyTorch 推理后端（可选）
+ torchvision>=0.15.0         # PyTorch 图像预处理（可选）

# requirements.txt 不变 (Vision 相关的不删，用户可能还有其他用途)
  openai
  langchain
  langgraph
  fastapi
  streamlit
  ...
```
