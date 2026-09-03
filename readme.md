# 智能种植规划助手

基于 LangChain + LangGraph + FastAPI + Streamlit 的多 Agent 智能农业助手。7 个专业智能体协同作战，通过调度中心统一管理，支持复合意图并行处理、Agent 间自动联动、IoT 设备自主控制与摄像头定时巡检。

## 功能特性

### 对话交互
- **20 种意图识别**：作物选择、种植时间、种植方法、提醒设置、进度跟踪、病虫害防治、收获规划、图片分析、天气查询、财务查询、政策补贴、地块管理、设备控制、作物监测、问候/感谢/告别/身份/功能/意图不明
- **LLM 智能分类**：三级降级（关键词快速路径 → LLM 推理 → 关键词兜底）
- **FAISS 向量检索**：作物知识 + 政策文档双索引，语义搜索 + 关键词匹配双通道，自动 fallback
- **跨会话记忆**：用户档案 + 对话历史自动持久化，重开浏览器自动恢复
- **长记忆持久化**：每 3 轮自动总结注入上下文

### 种植规划
- 根据地区/土壤/面积/目标生成个性化种植计划（时间表、任务、资源、风险、产量预估）
- **15 种作物**结构化知识库：小麦、玉米、水稻、大豆、棉花、土豆、花生、高粱、谷子、油菜、甘薯、甘蔗、烟草、茶叶、番茄
- 多方案对比评分（价格、产量、风险、适应性）
- **轮作建议**：连作风险检测 + 下季推荐 + 多年轮作规划
- ** 种植向导**：三步一键生成计划 + 进度 + 任务 + 提醒
- **生长阶段自动推进**：根据种植日期自动计算当前阶段（只前进不后退）

### 农事提醒
- 7 种类型：浇水、施肥、除草、病虫害防治、修剪、收获、其他
- 多频率设置：单次、每天、每周、每两周、每月、自定义
- **后台调度器**：FastAPI 常驻进程定时检查，到期自动触发 SMS 推送
- 侧边栏进度卡片、任务列表、收获倒计时

### 病虫害诊断
- 图片上传 → Vision 多模态模型自动识别作物类型、生长阶段、病害
- 文字描述症状智能提取（20+ 症状关键词）→ 精准匹配防治方案
- **Agent 间自动联动**：检测到喷药需求时自动调用气象 Agent 判断施药窗口
- **量化病害风险**：8 种作物 20+ 病害内置温湿度/降雨/阶段阈值，综合评分
- **病虫害气象预警**：FastAPI 定时评估温湿度 + 降雨 + 作物阶段 → 风险推送

### 天气服务
- **双 API 熔断**：和风天气格点 → OpenWeatherMap 自动降级
- 实时天气、5 天预报、灾害预警、农事建议
- **施药气象分析**：风力/降雨/温度/湿度 5 项评估 + 最佳窗口推荐
- 地块天气叠加：地图上每个地块显示实时温度徽标

### 财务管理
- 成本/收入记账（种子、肥料、农药、人工、农机等）
- Plotly 月度收支趋势图 + 成本构成饼图
- 年度报表、按作物汇总、亩均利润
- CSV 导入/导出

### 地块管理
- Folium 交互地图：绘制多边形边界、GPS 定位、卫星图图层
- 自动面积计算（Haversine 公式）、各地块实时天气汇总

### 政策补贴
- 政策文档 FAISS 向量索引
- 8 种常见补贴类型速查表
- 内置作物补贴知识库
- 独立浏览页面，支持搜索

### 农历节气
- **24 节气天文公式计算**（不限年份，永不过期）
- 每个节气配备农事活动、农谚、适宜作物
- 侧边栏农历日期 + 当前节气 + 农事提示
- 农事日历甘特图叠加节气标记线

### 语音交互
- 语音输入：浏览器 Web Speech API（中文普通话）
- 语音播报：每条回复支持 TTS 朗读
- **语音指令解析**：支持「记账：小麦 收入 5000」「提醒：明天 8点 小麦 浇水」「查天气」等快捷语音指令

### 农资计算器
- 播种量计算（千粒重 × 发芽率 × 亩株数）
- 施肥量计算（N-P-K 折算具体化肥品种亩用量）
- 农药稀释计算（按倍数或亩用量）

### IoT 设备控制 NEW
- **7 个专业 Agent 中的设备控制 + 作物监测**：对话即可控制灌溉/施肥/通风/补光/加热
- **设备仪表盘**：实时监控设备状态、传感器读数，一键快捷操作
- **规则引擎**：用户设定安全边界 → Agent 在边界内自主决策执行
- **自动规则**：传感器触发 + 定时轮询，无需人工干预
- **4 种驱动协议**：虚拟模拟器 / MQTT / HTTP REST / Modbus RTU/TCP
- **Agent 间联动**：病虫害检测 → 自动问天气 → 自动执行喷药

### 作物监测 NEW
- 摄像头定时巡检：每 N 分钟自动拍照，Vision AI 分析作物健康状况
- 多维健康评估：养分状态 / 水分状态 / 病虫害检测 / 生长阶段识别
- 自动联动执行：高危问题（干旱/缺肥/虫害）→ 自动匹配设备 → 自主灌溉/施肥/告警
- 分析记录持久化：照片 + AI 分析结果 JSON 存储，支持历史回溯
- 故障摄像头自动跳过，日志记录

### 作物百科
- 15 种作物完整知识浏览（生长阶段 / 施肥灌溉 / 病虫害 / 产量市场 / 种植季节）
- 双作物对比表
- 搜索 + 选择快速定位

### 自适应布局
- 自动检测屏幕宽度，手机/桌面双布局
- 手机端：下拉导航、精简侧边栏、地图适配、按钮全宽

### 用户系统
- 用户注册/登录，数据按用户隔离存储
- **三级自主权**：Low(全部确认) / Medium(规则边界内自动) / High(完全自主)
- 用户档案（地区/土壤/面积/经验/目标）持久化，重开自动恢复

---

## 架构

### 多 Agent 调度

```
用户输入 → parse_input → classify_intent
 │
 ┌─────────┴──────────┐
 │ AgentOrchestrator │ ← 调度中心（复合意图并行 + 回答合并）
 └─────────┬──────────┘
 │
 ┌──────────┬──────────┬──┴─────┬──────────┬──────────┬──────────┐
 │ │ │ │ │ │ │
┌────┴────┐ ┌───┴───┐ ┌───┴───┐ ┌──┴──┐ ┌────┴────┐ ┌───┴────┐ ┌───┴──────┐
│ 种植 │ │ 病虫害│ │ 气象│ │ 财务│ │ 农事 │ │ 设备 │ │ 作物监测│
│ 4 种意图 │ │2 种意图│ │1 种意图│ │2种意图│ │ 3 种意图 │ │1 种意图 │ │1 种意图 │
└────┬────┘ └───┬───┘ └───┬───┘ └──┬──┘ └────┬────┘ └───┬────┘ └───┬──────┘
 │ │ │ │ │ │ │
 └──────────┼─────────┼────────┼──────────┼──────────┼──────────┘
 │ │ │ │ │
 Agent 间自动联动 ────────┴──────────┴──────────┘
 病虫害 → 问天气 → 自动喷药
 霜冻预警 → 自动开加热器
 传感器触发 → 规则引擎 → 自动灌溉
 摄像头巡检 → Vision 分析 → 自主执行
```

能力：
• 单意图 → 直接分派到对应 Agent
• 复合意图 → 并行执行 + 回答合并（如「小麦价格和补贴」→ 财务 + 政策双 Agent）
• Agent 间互调 → 病虫害自动问天气判断施药时机
• **设备控制** → 对话驱动 + 规则自动触发双重通路
• **规则引擎** → 用户定边界，AI 在边界内自主优化
• **作物监测** → 摄像头定时拍照 → Vision AI 分析 → 高危问题自主执行

### 服务架构

```
┌──────────────────────────────────────────────┐
│ FastAPI (:18001) 常驻后端 │
│ │
│ APScheduler 定时任务 │
│ ├─ 每 3min: 任务自动执行检查 │
│ ├─ 每 5min: 提醒检查 + SMS 推送 │
│ ├─ 每 5min: 设备规则轮询 + 传感器触发 │
│ ├─ 每 30min: 天气预警缓存 + 持续异常检测 │
│ ├─ 每 30min: 摄像头定时巡检 + Vision 分析 │
│ └─ 每 6h: 病虫害风险评估 │
│ │
│ REST API (~35 端点) │
│ ├─ POST /api/chat Agent 对话 │
│ ├─ GET /api/dashboard 仪表盘聚合 │
│ ├─ GET/POST /api/progress 种植进度 CRUD │
│ ├─ GET/POST /api/tasks 农事任务 CRUD │
│ ├─ GET/POST/DELETE /api/fields 地块管理 CRUD │
│ ├─ GET/POST /api/finance/* 财务 CRUD+报表 │
│ ├─ GET/POST /api/profile 用户档案 │
│ ├─ GET /api/weather/* 天气 + 预警 │
│ ├─ GET /api/solar-terms 农历节气 │
│ ├─ POST /api/reminders 提醒管理 │
│ ├─ GET /api/encyclopedia/* 作物百科 │
│ ├─ GET /api/policy/search 政策搜索 │
│ ├─ POST /api/plan 种植方案向导 │
│ ├─ GET/POST/DELETE /api/devices 设备管理 CRUD │
│ ├─ POST /api/devices/{id}/command 设备指令 │
│ ├─ GET /api/devices/{id}/state 设备状态 │
│ ├─ GET /api/devices/{id}/snapshot 摄像头拍照 │
│ ├─ GET /api/camera/analysis/{id} AI分析记录 │
│ ├─ GET/POST/PUT/DELETE /api/rules 规则 CRUD │
│ ├─ GET/POST /api/actions/* 操作确认+日志 │
│ └─ GET /api/health 健康检查 │
│ │
│ 直接调用 core/ agent/ knowledge/ 模块 │
└──────────────────────────────────────────────┘
 ↑ HTTP
┌──────────────────────────────────────────────┐
│ Streamlit (:8501) 纯展示前端 │
│ │
│ 所有业务逻辑通过 API 调用 │
│ UI 渲染代码完全不变 │
└──────────────────────────────────────────────┘
```

---

## 硬件接入

Agent 支持通过 4 种协议接入真实 IoT 设备，实现自主控制。通过设备仪表盘即可注册设备，无需编写任何代码。

### 支持的驱动类型

| 驱动 | 协议 | 适用硬件 | 需要额外安装 |
|------|------|---------|------------|
| **Simulator** | 本地内存 | 开发测试（7 个内置虚拟设备） | 无 |
| **MQTT** | MQTT 3.1.1 | ESP32/ESP8266、树莓派、任意 MQTT 设备 | `pip install paho-mqtt` |
| **HTTP REST** | HTTP | 智能插座(Tasmota/ESPHome)、树莓派 GPIO 控制器 | 无（使用 `requests`） |
| **Modbus** | RTU/TCP | PLC、变频器、工业传感器 | `pip install pymodbus` |
| **CoAP** | CoAP/CoAPS | 低功耗传感器、受限网络节点、边缘设备 | `pip install aiocoap` |
| **OPC UA** | OPC UA TCP | PLC、SCADA、工业网关 | `pip install asyncua` |

### 硬件模拟器（开发测试用）

无需真实硬件即可测试全部 IoT 功能。模拟器启动**完整三协议栈**（HTTP/MQTT/Modbus），终端/前端/API 均通过真实协议通道控制设备，所有操作实时反馈到终端。

```bash
# 一键启动全部协议模拟器
python hardware_examples/all_hardware_simulator.py
```

**架构**：
```
UnifiedDeviceManager（共享状态）
 ├── 🌐 HTTP Server (Flask :5000)  ← 灌溉泵/补光灯/施肥机
 ├── 📡 MQTT Broker (内嵌 :1883) + 设备处理器  ← 通风扇/加热器
 ├── 🔧 Modbus TCP Server (内嵌 :5020)  ← 温湿度传感器/摄像头
 └── 🖥️ 终端 CLI（协议客户端）
      ├── HTTP 设备 → requests.post(:5000)
      ├── MQTT 设备 → MQTT publish(:1883)
      └── Modbus 设备 → TCP write(:5020)
```

**设备协议分布**：
| 设备 | 协议 | 地址 |
|------|------|------|
| 灌溉泵 / 补光灯 / 施肥一体机 | HTTP | :5000 |
| 通风扇 / 加热器 | MQTT | :1883 |
| 温湿度传感器 / 摄像头 | Modbus TCP | :5020 |

**终端命令**：
```bash
▸ list                    # 查看所有设备状态（含协议标签）
▸ boot pump               # 灌溉泵通电 → 通过 HTTP POST 发送
▸ start pump dur=30       # 灌溉泵工作30分钟 → HTTP POST
▸ start fan speed=60      # 通风扇60%转速 → MQTT publish
▸ stop fan                # 停止通风扇 → MQTT publish
▸ shutdown pump           # 关机断电
```

所有设备**默认通电待机**，启动即可操作。前端操控设备时终端实时显示协议层反馈。

### 接入步骤

**1. 在设备端实现指令接收 + 状态上报**（以 ESP32 + MQTT 为例）：

```cpp
// ESP32 订阅 control_topic，收到指令后控制继电器
// 完整代码见 docs/devices/设备连接指南.md

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
 // 解析 {"command":"start","params":{"duration":30}}
 if (command == "start") {
 digitalWrite(RELAY_PIN, HIGH); // 开水泵
 publish_state(); // 上报状态
 }
}
```

**2. 在设备仪表盘注册设备**：

> 设备仪表盘 → 添加设备 → 填写表单 → 注册

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 设备ID | `greenhouse_pump_01` | 全局唯一标识 |
| 设备名称 | 大棚水泵#1 | 人类可读名称 |
| 驱动类型 | MQTT | 选择通信协议 |
| Broker 地址 | `192.168.1.100` | MQTT Broker IP |
| 控制主题 | `greenhouse/pump/control` | 设备订阅的主题 |
| 状态主题 | `greenhouse/pump/state` | 设备上报状态的主题 |

**3. 对话控制或设定自动规则**：

```
用户: "帮小麦浇30分钟水"
Agent: 指令已执行！设备：greenhouse_pump_01 → start 参数：duration=30

# 或者设定自动规则：
规则: 当 soil_moisture < 30% 且 未来24h无雨 → 自动浇水30分钟
```

### 设备端要求

无论使用哪种协议，设备端只需实现两个接口：

**接收指令**（Agent → 设备）：
```json
{"command": "start|stop|set_param", "params": {"duration": 30}, "timestamp": "..."}
```

**上报状态**（设备 → Agent）：
```json
{"power": true, "status": "running", "temperature": 22.5, "timestamp": "..."}
```

### 设备端最小实现（Python 示例）

```python
# 最简单的 HTTP 设备端，30 行代码即可接入
from flask import Flask, request, jsonify
app = Flask(__name__)
state = {"power": False, "status": "idle"}

@app.route("/command", methods=["POST"])
def command():
 data = request.get_json()
 if data["command"] == "start":
 state["power"] = True; state["status"] = "running"
 # 这里操作你的 GPIO / 继电器
 return jsonify({"success": True})

@app.route("/state", methods=["GET"])
def get_state():
 return jsonify(state)

app.run(host="0.0.0.0", port=8080)
```

> **完整设备端代码**（ESP32 C++、树莓派 Python、Tasmota 适配器、Modbus 从站模拟器、接线图）见 **[docs/devices/设备连接指南.md](docs/devices/设备连接指南.md)**

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```env
# LLM 对话模型（必填）
LLM_API_KEY=sk-your-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# Vision 视觉模型（可选）
VISION_API_KEY=
VISION_BASE_URL=
VISION_MODEL=

# Embedding 向量模型（可选，用于 FAISS）
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=text-embedding-3-small

# 天气服务（可选）
WEATHER_API_PROVIDER=qweather
WEATHER_API_KEY=

# 腾讯云短信（可选）
SMS_SECRET_ID=
SMS_SECRET_KEY=
SMS_SDK_APP_ID=
SMS_SIGN_NAME=
SMS_TEMPLATE_ID=
SMS_REGION=ap-guangzhou
```

### 3. 构建知识库（可选，需要 Embedding API）

```bash
python knowledge/build_agriculture_rag.py
python knowledge/build_faiss_rag.py
```

### 4. 启动

```bash
# 一键启动后端 + 前端
python app/start.py all

# 或分别启动
python app/start.py backend # 终端 1: FastAPI :18001
python app/start.py web # 终端 2: Streamlit :8501
```

---

## 项目结构

```
Agriculture_Agent/
├── app/
│ ├── test1.py # Streamlit 入口 + 页面路由 + 设备检测
│ ├── start.py # CLI 启动脚本（backend/web/all 命令）
│ ├── api_server.py # FastAPI 主进程 + APScheduler
│ ├── api_routes.py # ~20 个 REST API 端点
│ ├── scheduler_jobs.py # 提醒/天气/病害定时任务
│ ├── scheduler_runner.py # 独立单实例调度进程入口
│ ├── agent/
│ │ ├── config.py # 环境变量 + 关键词常量
│ │ ├── state.py # AgentState（18 种意图）
│ │ ├── graph.py # LangGraph 工作流 + 多 Agent 调度
│ │ ├── agents/ # 7 个专业 Agent + 调度中心
│ │ │ ├── base.py # Agent 基类（互调支持）
│ │ │ ├── orchestrator.py # 调度中心（路由+并行+互调+合并）
│ │ │ ├── planting_agent.py # 种植规划 Agent（4种意图）
│ │ │ ├── disease_agent.py # 病虫害诊断 Agent（联动气象）
│ │ │ ├── weather_agent.py # 气象服务 Agent
│ │ │ ├── finance_agent.py # 财务与政策 Agent（2种意图）
│ │ │ ├── farming_agent.py # 农事管理 Agent（3种意图）
│ │ │ ├── device_agent.py # 设备控制 Agent（LLM解析+规则引擎+自主权）
│ │ │ └── crop_monitor_agent.py # 作物监测 Agent（Vision分析+自主决策）
│ │ └── nodes/ # 工作流节点（16个）
│ │ ├── parse_input.py # 输入解析 + 前季作物提取 + 语音指令
│ │ ├── classify_intent.py # 意图分类（LLM 推理 + 关键词降级）
│ │ ├── rag_retrieval.py # FAISS + 关键词双通道检索
│ │ ├── llm_response.py # LLM 通用回答 + 追问引导
│ │ ├── planting_plan.py # 种植规划 + 轮作建议
│ │ ├── reminder.py # 提醒管理
│ │ ├── image_analysis.py # Vision 图片分析 + 防治方案
│ │ ├── weather.py # 天气查询 + 施药气象分析
│ │ ├── finance.py # 财务查询
│ │ ├── field.py # 地块管理 + 轮作
│ │ ├── policy.py # 政策补贴查询
│ │ ├── progress.py # 进度跟踪
│ │ ├── extract_tasks.py # 自动提取任务
│ │ └── update_memory.py # 长记忆更新（每3轮自动总结）
│ ├── ui/
│ │ ├── theme.py # 设计系统 CSS + 导航栏 + 响应式
│ │ └── sidebar.py # 侧边栏（进度/任务/天气/农历）
│ └── views/
│ ├── dashboard.py # 概览仪表盘（默认首页）
│ ├── chat.py # 对话页面（语音 + TTS）
│ ├── profile.py # 基本信息
│ ├── fields.py # 地块管理（地图 + 天气叠加）
│ ├── finance.py # 财务管理（记账 + 图表）
│ ├── calendar.py # 农事日历（甘特图）
│ ├── policy.py # 政策补贴查询
│ ├── encyclopedia.py # 作物百科
│ ├── calculator.py # 农资计算器
│ ├── wizard.py # 种植方案向导
│ ├── devices.py # 设备仪表盘
│ └── rules.py # 规则编辑器
│
├── core/
│ ├── planting_planner.py # 种植规划引擎
│ ├── planting_tracker.py # 进度跟踪 + 任务卡片 + 自动推进
│ ├── crop_comparison.py # 多作物对比评分
│ ├── crop_rotation.py # 轮作建议
│ ├── weather_service.py # 双 API 天气 + 熔断
│ ├── weather_alerts.py # 灾害预警 + 收获倒计时
│ ├── weather_history.py # 天气持续异常检测
│ ├── spray_advisor.py # 施药气象评估
│ ├── reminder_system.py # 提醒管理
│ ├── reminder_scheduler.py # 后台调度 + SMS 推送
│ ├── finance_manager.py # 财务管理 + 报表
│ ├── market_service.py # 市场价格查询
│ ├── map_manager.py # Folium 交互地图管理
│ ├── chat_history.py # 对话持久化
│ ├── sms_service.py # 腾讯云短信
│ ├── wechat_notify.py # 微信通知
│ ├── voice_components.py # 语音输入 (STT)
│ ├── tts_components.py # 语音播报 (TTS)
│ ├── disease_risk.py # 病虫害量化风险评估
│ ├── device_rule_engine.py # 设备规则引擎（条件匹配+动作评估+自主权）
│ ├── device_executor.py # 设备指令执行器（重试+日志+待确认队列）
│ ├── device_registry_factory.py # 多驱动设备注册工厂
│ └── lunar_calendar.py # 农历24节气天文计算
│
├── knowledge/
│ ├── simple_agriculture_rag.py # 关键词检索
│ ├── faiss_agriculture_rag.py # FAISS 向量检索
│ ├── build_agriculture_rag.py # 作物知识索引构建
│ └── build_faiss_rag.py # 政策文档索引构建
│
├── devices/ # IoT 设备驱动模块
│ ├── base.py # 驱动抽象基类 + DeviceCommand/DeviceInfo
│ ├── registry.py # 驱动注册中心 + 设备发现
│ ├── simulator_driver.py # 虚拟设备模拟器（6 个内置设备）
│ ├── mqtt_driver.py # MQTT 3.1.1 协议驱动（ESP32/树莓派）
│ ├── http_driver.py # HTTP REST 驱动（Tasmota/ESPHome/Flask）
│ ├── camera_driver.py # 摄像头驱动（拍照 + Vision 分析联动）
│ └── modbus_driver.py # Modbus RTU/TCP 驱动（PLC/工业传感器）
├── agriculture_knowledge/crops/ # 15 种作物结构化知识 JSON
├── data/ # 运行时 JSON 存储（按用户分目录）
├── tests/ # 单元测试（9 个测试文件）
├── docs/ # 设备连接指南等文档
└── .env # 环境变量配置
```

---

## 对话示例

| 意图 | 示例问法 |
|------|---------|
| 作物推荐 | "华北地区壤土 50亩适合种什么？" |
| 种植时间 | "小麦什么时候播种？" |
| 种植方法 | "玉米怎么施肥？" |
| 提醒设置 | "给小麦设置每周浇水提醒" |
| 进度查询 | "我的小麦现在该做什么？" |
| 病虫害 | "番茄叶子发黄长斑怎么办？" |
| 图片诊断 | 上传病虫害图片 + "请分析" |
| 天气查询 | "明天适合喷药吗？" |
| 施药判断 | "今天风大适合打除草剂吗？" |
| 财务查询 | "今年大豆赚了多少？" |
| 地块管理 | "我有几个地块？" |
| 轮作建议 | "去年种了花生，今年种什么好？" |
| 政策补贴 | "种小麦有什么补贴？" |
| 市场价格 | "土豆现在多少钱一斤？" |
| 设备控制 | "帮小麦浇30分钟水" |
| 作物监测 | "看看大棚里的番茄长势怎么样" |

---

## 数据存储

所有数据以 JSON 格式本地存储于 `data/` 目录，按用户分目录管理，FastAPI 和 Streamlit 共享读写：

| 文件/目录 | 内容 |
|-----------|------|
| `{username}/planting_tasks.json` | 农事任务卡片（含设备控制字段） |
| `{username}/planting_progress.json` | 种植进度记录 |
| `{username}/reminders.json` | 提醒数据 |
| `{username}/finance_costs.json` | 成本支出记录 |
| `{username}/finance_income.json` | 销售收入记录 |
| `{username}/fields.json` | 地块边界、面积、当前作物 |
| `{username}/device_rules.json` | 用户自定义设备规则 |
| `{username}/custom_devices.json` | 用户注册的自定义设备 |
| `{username}/device_action_log.json` | 设备操作执行日志 |
| `{username}/photos/{device_id}/` | 摄像头巡检照片 + AI 分析结果 |
| `user_profile.json` | 用户档案（区域/土壤/面积等） |
| `users.json` | 用户账户（用户名+密码哈希） |
| `chat_history.json` | 跨会话对话历史 |
| `weather_alerts_cache.json` | 天气预警缓存 |
| `weather_persistence.json` | 天气持续异常检测结果 |
| `disease_risks.json` | 病虫害风险评估结果 |

---

## 开发

```bash
# 核心模块可独立运行调试
python core/planting_planner.py
python core/weather_service.py
python core/reminder_system.py
python core/finance_manager.py

# 搜索知识库
python knowledge/build_agriculture_rag.py search "小麦什么时候播种"

# 运行测试
python -m pytest tests/ -v

# 调试模式
DEBUG_MODE=true streamlit run app/main.py
```
