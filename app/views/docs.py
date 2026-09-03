"""📚 文档中心 — 使用手册 / API 接口 / 技术手册 / 硬件示例"""

import streamlit as st


def render_docs_page():
    st.markdown("# 📚 文档中心")
    st.caption("使用手册 · API 接口 · 技术手册 · 硬件示例代码")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 使用手册", "🔌 API 接口", "⚙️ 技术手册", "🔧 硬件示例"
    ])

    # ═══════════════════════════════════════════════
    # Tab 1: 使用手册
    # ═══════════════════════════════════════════════
    with tab1:
        st.markdown("""
## 📖 使用手册

### 🚀 快速开始

1. **启动服务**：双击 `app/start.py` 或在终端运行 `python app/start.py`
2. **打开浏览器**：访问 `http://localhost:8501`
3. **注册/登录**：首次使用输入用户名和密码即可注册
4. **填写基本信息**：按照引导填写地区、土壤类型、种植面积等
5. **开始对话**：在聊天框输入问题，如"河北适合种什么？"

### 💬 对话功能

| 功能 | 说明 | 示例 |
|------|------|------|
| 作物推荐 | 根据地区/土壤推荐合适作物 | "河北壤土适合种什么" |
| 种植规划 | 生成完整种植方案 | "帮我规划小麦种植" |
| 病虫害防治 | 诊断病害并推荐用药 | "小麦叶子发黄了怎么办" |
| 天气查询 | 查询天气和施药建议 | "今天适合喷药吗" |
| 财务管理 | 记录成本收入，生成报表 | "记账：小麦种子 50元" |
| 政策查询 | 查询农业补贴政策 | "小麦有什么补贴" |
| 设备控制 | 语音或文字控制灌溉/通风等 | "帮小麦浇30分钟水" |
| 地块管理 | 在地图上标注和管理地块 | 进入「地块管理」页面操作 |

### 🎤 语音指令

在聊天框右侧点击 🎤 按钮说出指令，支持以下格式：

| 指令格式 | 说明 | 示例 |
|----------|------|------|
| `记账：小麦 收入 500元` | 快速记录财务 | "记账：玉米 支出 肥料 200元" |
| `提醒：明天 8点 小麦 浇水` | 设置农事提醒 | "提醒：后天 小麦 施肥" |
| `添加任务：小麦 浇水` | 创建待办任务 | "添加任务：玉米 除草" |
| `查天气` | 查询当前天气 | "查天气 北京" |
| `记录进度：小麦 播种完成` | 更新种植进度 | — |

### 📊 主要页面

| 页面 | 功能 |
|------|------|
| 🏠 首页 | 仪表盘：概览、进度、财务摘要 |
| 💬 对话 | AI 对话助手（默认页面） |
| 🗺️ 地块管理 | 在地图上创建和管理农田地块 |
| 💰 财务管理 | 记录成本/收入，查看报表 |
| 📅 种植日历 | 甘特图查看种植进度 |
| 📋 政策查询 | 查询国家和地方农业政策 |
| 📚 百科 | 查看作物详细信息 |
| 🧮 计算器 | 种子/肥料/农药用量计算 |
| 🪄 种植向导 | 三步生成完整种植计划 |
| 🔌 设备仪表盘 | 管理IoT设备，执行灌溉/通风等 |
| 📏 规则引擎 | 配置设备自动化规则 |
| 👤 基本信息 | 修改地区、土壤、目标等 |
""")

    # ═══════════════════════════════════════════════
    # Tab 2: API 接口
    # ═══════════════════════════════════════════════
    with tab2:
        st.markdown("""
## 🔌 API 接口文档

后端运行在 `http://localhost:18001`，所有接口返回 JSON。

### 对话

**`POST /api/chat`**

发送对话消息，获取AI回复。

```json
// 请求
{
  "user_question": "河北适合种什么？",
  "username": "default",
  "user_profile": {
    "region": "河北",
    "soil_type": "壤土",
    "farm_size": 10.0,
    "experience": "初学者",
    "goals": ["高产"]
  },
  "image_data": "base64...（可选，图片分析时使用）",
  "image_mime_type": "image/jpeg（可选）"
}

// 响应
{
  "final_answer": "根据您的条件，推荐小麦...",
  "short_term_facts": {"crop": "小麦", "region": "河北"}
}
```

### 仪表盘

**`GET /api/dashboard?username=xxx`**

获取仪表盘概览数据（进度、财务摘要、天气预警）。

**`GET /api/progress?username=xxx`**

获取种植进度列表。

**`GET /api/tasks?username=xxx`**

获取待办任务列表。

### 天气

**`GET /api/weather/{location}?username=xxx`**

查询指定地区的当前天气和预报。

```json
// 响应
{
  "current": {"temperature": 25.0, "humidity": 60, "weather_desc": "晴"},
  "forecast": [...],
  "spray_assessment": {"suitable": true, "score": 85, ...}
}
```

**`GET /api/weather/alerts/{region}?username=xxx`**

获取天气预警信息。

### 设备控制

**`GET /api/devices?username=xxx`** — 获取所有设备列表

**`POST /api/devices/command`** — 向设备发送指令

```json
// 请求
{
  "device_id": "my_irrigation_pump",
  "command": "start",
  "params": {"duration": 30},
  "username": "default"
}
```

**`GET /api/devices/logs?username=xxx&limit=50`** — 获取设备操作日志

**`GET /api/devices/pending?username=xxx`** — 获取待确认操作列表

### 规则引擎

**`GET /api/rules?username=xxx`** — 获取所有设备规则

**`POST /api/rules`** — 创建新规则

```json
{
  "name": "自动灌溉",
  "trigger": {
    "conditions": [{"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30}],
    "logic": "AND"
  },
  "action": {"device_id": "my_irrigation_pump", "command": "start", "params": {"duration": 30}},
  "constraints": {"max_duration_per_use": 60, "forbidden_hours": [22,23,0,1,2,3,4,5]},
  "username": "default"
}
```

### 财务管理

**`GET /api/finance/summary?username=xxx&crop=xxx`** — 获取财务汇总

**`POST /api/finance/cost`** — 添加成本记录
```json
{"crop": "小麦", "cost_type": "种子", "quantity": 20, "unit_price": 3.5, "username": "default"}
```

**`POST /api/finance/income`** — 添加收入记录

**`GET /api/finance/export?username=xxx`** — 导出CSV

### 地块管理

**`GET /api/fields?username=xxx`** — 获取地块列表

**`POST /api/fields`** — 创建地块

### 百科

**`GET /api/encyclopedia`** — 获取所有作物列表

**`GET /api/encyclopedia/{crop_name}`** — 获取指定作物详情

### 种植规划

**`POST /api/plan?username=xxx`** — 生成种植计划
```json
{"crop": "小麦", "region": "河北", "soil_type": "壤土", "farm_size": 10.0}
```
""")

    # ═══════════════════════════════════════════════
    # Tab 3: 技术手册
    # ═══════════════════════════════════════════════
    with tab3:
        st.markdown("""
## ⚙️ 技术手册

### 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit 前端                      │
│  chat.py  dashboard.py  devices.py  finance.py ...   │
│         ↕ HTTP (api_client.py)                       │
├─────────────────────────────────────────────────────┤
│                FastAPI 后端 (api_server.py)           │
│         ↕ LangGraph Agent 管道                       │
├─────────────────────────────────────────────────────┤
│  parse_input → classify_intent → agent_dispatch      │
│     → rag_retrieval → llm_response → extract_tasks   │
│     → update_memory                                  │
├─────────────────────────────────────────────────────┤
│  设备驱动层 (devices/)  │  知识库 (knowledge/)        │
│  MQTT/Modbus/HTTP/      │  FAISS + 关键词双通道       │
│  Camera/Simulator        │  检索                       │
└─────────────────────────────────────────────────────┘
```

### 环境配置 (.env)

```bash
# LLM 配置
LLM_MODEL=kimi-k2.6              # 对话模型
LLM_API_KEY=sk-xxxx               # API Key
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_TEMPERATURE=1.0

# 嵌入模型（知识库检索）
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_BASE_URL=https://api.openai.com/v1

# 视觉模型（作物图片分析）
VISION_MODEL=gpt-4o
VISION_API_KEY=sk-xxxx

# 天气服务
WEATHER_API_PROVIDER=openweathermap   # 或 qweather
WEATHER_API_KEY=xxxx

# 数据存储
DATA_STORAGE_DIR=data

# 自主决策
AUTONOMY_LEVEL=medium            # low/medium/high
AUTO_DECISION_TIMEOUT=120        # 决策超时(秒)
```

### LangGraph Agent 管道

| 节点 | 功能 | LLM调用 |
|------|------|---------|
| `parse_input` | 解析用户输入，提取事实 | 无 |
| `classify_intent` | 意图分类（LLM/关键词） | 0-1次 |
| `agent_dispatch` | 调度到专业Agent | 0-2次 |
| `rag_retrieval` | 双通道知识检索 | 嵌入API 1-2次 |
| `llm_response` | 生成最终回答 | 1次 |
| `extract_tasks` | 提取待办任务 | 1次 |
| `update_memory` | 更新长期记忆 | 0-1次（每3轮） |

### 设备驱动架构

所有设备驱动继承 `BaseDeviceDriver` (devices/base.py)：

```python
class BaseDeviceDriver(ABC):
    driver_name: str = "base"

    @abstractmethod
    async def connect(self) -> bool: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def execute(self, device_id, command) -> DeviceResult: ...
    @abstractmethod
    async def read_state(self, device_id) -> Dict: ...
    @abstractmethod
    async def discover(self) -> List[DeviceInfo]: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
```

已实现的驱动：
| 驱动 | 文件 | 协议 |
|------|------|------|
| SimulatorDriver | simulator_driver.py | 虚拟设备（开发测试） |
| MQTTDriver | mqtt_driver.py | MQTT 物联网协议 |
| ModbusDriver | modbus_driver.py | Modbus RTU/TCP |
| HTTPDriver | http_driver.py | REST API 设备 |
| CameraDriver | camera_driver.py | USB/IP/RTSP 摄像头 |

### 设备注册中心

`DeviceDriverRegistry` (devices/registry.py) 管理所有驱动和设备的注册、发现、执行。

`setup_registry()` (core/device_registry_factory.py) 是统一工厂函数，自动加载内置虚拟设备 + 用户自定义设备。

### 规则引擎

`RuleEngine` (core/device_rule_engine.py) 实现：
- 条件匹配（传感器/天气/时间）
- 硬限制（代码级不可突破）
- 软约束（用户可配置）
- AI 微调（LLM 调整参数）
- 自主权级别（low/medium/high）
""")

    # ═══════════════════════════════════════════════
    # Tab 4: 硬件示例代码
    # ═══════════════════════════════════════════════
    with tab4:
        st.markdown("""
## 🔧 硬件示例代码

### 1. ESP32 + 土壤传感器 (MQTT)

```cpp
// ESP32 MQTT 土壤监测节点
#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "YourWiFi";
const char* password = "YourPassword";
const char* mqtt_server = "192.168.1.100";
const char* device_id = "soil_sensor_01";

WiFiClient espClient;
PubSubClient client(espClient);

// 土壤湿度传感器 (A0)
#define SOIL_PIN 34

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);

    client.setServer(mqtt_server, 1883);
}

void reconnect() {
    while (!client.connected()) {
        if (client.connect(device_id)) {
            // 订阅控制指令
            String ctrl_topic = "devices/" + String(device_id) + "/control";
            client.subscribe(ctrl_topic.c_str());
        }
        delay(2000);
    }
}

void loop() {
    if (!client.connected()) reconnect();
    client.loop();

    // 读取传感器
    int raw = analogRead(SOIL_PIN);
    float moisture = map(raw, 0, 4095, 0, 100);

    // 发布状态
    String state_topic = "devices/" + String(device_id) + "/state";
    String payload = "{\\"soil_moisture\\":" + String(moisture)
                   + ",\\"temperature\\":25.0,\\"humidity\\":60}";
    client.publish(state_topic.c_str(), payload.c_str());

    delay(10000);  // 10秒上报一次
}
```

### 2. 继电器控制灌溉阀门 (Modbus RTU)

```python
# Python Modbus 灌溉阀门控制
import minimalmodbus
import time

class IrrigationValve:
    def __init__(self, port="/dev/ttyUSB0", slave_id=1):
        self.device = minimalmodbus.Instrument(port, slave_id)
        self.device.serial.baudrate = 9600
        self.device.serial.timeout = 1.0

    def start(self, duration_minutes=30):
        \"\"\"打开阀门\"\"\"
        self.device.write_register(0, 1)  # HR[0]=1 启动
        print(f"灌溉阀门已开启，时长 {duration_minutes} 分钟")

    def stop(self):
        \"\"\"关闭阀门\"\"\"
        self.device.write_register(0, 0)  # HR[0]=0 停止
        print("灌溉阀门已关闭")

    def read_state(self):
        \"\"\"读取状态\"\"\"
        regs = self.device.read_registers(0, 4)
        return {
            "power": regs[0] == 1,
            "status": {0: "idle", 1: "running"}.get(regs[1], "unknown"),
            "flow_rate": regs[2] * 0.1,  # 0.1 L/min 精度
            "total_liters": regs[3],
        }

# 使用示例
valve = IrrigationValve(port="COM3", slave_id=1)
valve.start(duration_minutes=30)
time.sleep(10)
state = valve.read_state()
valve.stop()
```

### 3. 摄像头接入 (RTSP/HTTP)

```python
# 摄像头图片采集与分析
import cv2
import base64
import requests

def capture_and_analyze(camera_url, api_base="http://localhost:18001"):
    \"\"\"拍照并通过 API 分析\"\"\"
    cap = cv2.VideoCapture(camera_url)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "拍照失败"}

    # 编码为 JPEG
    _, jpeg = cv2.imencode(".jpg", frame)
    image_b64 = base64.b64encode(jpeg.tobytes()).decode()

    # 发送到后端分析
    resp = requests.post(f"{api_base}/api/chat", json={
        "user_question": "请分析这张农作物图片",
        "username": "default",
        "image_data": image_b64,
        "image_mime_type": "image/jpeg",
        "user_profile": {"region": "河北", "soil_type": "壤土"},
    })
    return resp.json()

# USB 摄像头
result = capture_and_analyze(0)

# IP 摄像头 (RTSP)
result = capture_and_analyze("rtsp://admin:password@192.168.1.10/stream")

# ESP32-CAM
result = capture_and_analyze("http://192.168.1.11/capture")
```

### 4. 自定义HTTP设备

```python
# 简单的 HTTP 智能插座控制器
from flask import Flask, request, jsonify

app = Flask(__name__)
device_state = {"power": False, "status": "idle"}

@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(device_state)

@app.route("/api/command", methods=["POST"])
def execute():
    cmd = request.json
    if cmd.get("command") == "start":
        device_state["power"] = True
        device_state["status"] = "running"
        # 这里控制实际的 GPIO/继电器
        return jsonify({"success": True, "message": "已启动"})
    elif cmd.get("command") == "stop":
        device_state["power"] = False
        device_state["status"] = "idle"
        return jsonify({"success": True, "message": "已停止"})
    return jsonify({"success": False, "message": "未知命令"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

然后在设备管理页面添加此设备：
- 驱动类型：`HTTP`
- Base URL：`http://192.168.1.20:5000`

### 5. 自定义设备配置文件

在 `data/{username}/custom_devices.json` 中配置：

```json
[
  {
    "device_id": "greenhouse_temp_01",
    "name": "温室温度传感器",
    "driver": "mqtt",
    "capabilities": ["read_sensor"],
    "sensors": ["temperature", "humidity"],
    "location": "大棚A区",
    "connection": {
      "host": "192.168.1.100",
      "port": 1883,
      "control_topic": "devices/greenhouse_temp_01/control",
      "state_topic": "devices/greenhouse_temp_01/state"
    }
  },
  {
    "device_id": "irrigation_pump_01",
    "name": "灌溉水泵",
    "driver": "modbus",
    "capabilities": ["irrigate"],
    "location": "大棚A区",
    "connection": {
      "port": "/dev/ttyUSB0",
      "slave_id": 1,
      "mode": "rtu"
    }
  }
]
```
""")

    # ── 底部 ──
    st.divider()
    st.caption("💡 更多问题请在对话页面咨询 AI 助手")
