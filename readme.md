# 智能种植规划与事件提醒 Agent

基于 LangChain + LangGraph + Streamlit 构建的智能农业助手，为农户和农业从业者提供全周期种植管理服务。

## 功能特性

- **智能作物推荐** - 根据地区气候、土壤条件推荐适宜作物
- **种植规划** - 生成完整的种植时间表、资源需求估算和风险评估
- **农事提醒** - 支持浇水、施肥、除草等多种提醒类型，可自定义频率
- **进度跟踪** - 自动计算种植进度，管理各阶段任务
- **病虫害诊断** - 支持图片上传，多模态AI智能识别病虫害
- **政策知识库** - 集成农业政策文档，解答补贴、法规相关问题

## 技术架构

```
用户交互层 (Streamlit Web界面)
    ↓
业务逻辑层 (LangGraph Agent)
    ↓
数据存储层 (JSON文件)
```

### 核心依赖

- **LangChain + LangGraph** - Agent工作流和LLM调用
- **Streamlit** - Web界面
- **OpenAI API** - 大语言模型支持（支持自定义API端点）
- **FAISS** - 向量检索（政策文档知识库）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env` 文件并填写你的API密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# LLM对话模型配置
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 可选：天气API配置
WEATHER_API_PROVIDER=qweather
WEATHER_API_KEY=your-weather-api-key
```

### 3. 启动应用

```bash
streamlit run app/test1.py
```

或直接运行：

```bash
python app/start.py
```

## 项目结构

```
.
├── app/                      # 主应用程序
│   ├── test1.py             # 核心Agent逻辑和Web界面
│   └── start.py             # 启动入口
├── core/                     # 业务模块
│   ├── planting_planner.py  # 种植规划
│   ├── reminder_system.py   # 提醒管理
│   ├── planting_tracker.py  # 进度跟踪
│   └── voice_components.py  # 语音输入
├── agriculture_knowledge/   # 农业知识库
│   └── crops/               # 作物数据(JSON)
├── faiss_index/             # 政策文档向量库
├── policy_docs/             # 政策文档
├── data/                    # 运行时数据存储
├── .env                     # 环境变量配置
└── requirements.txt         # 依赖列表
```

## 使用指南

### 首次使用

1. 在侧边栏填写你的基础信息（地区、土壤类型、种植面积等）
2. 开始对话，询问种植相关问题
3. 系统会自动根据上下文提供个性化建议

### 支持的对话类型

| 类型 | 示例 |
|------|------|
| 作物选择 | "华北地区适合种什么？" |
| 种植时间 | "小麦什么时候播种？" |
| 种植方法 | "怎么种玉米？" |
| 提醒设置 | "设置浇水提醒" |
| 进度查询 | "查看我的种植进度" |
| 病虫害 | "番茄叶子发黄怎么办？" |
| 图片诊断 | 上传病虫害图片 |

## 扩展开发

### 添加新作物

1. 在 `agriculture_knowledge/crops/` 创建JSON文件
2. 参考现有格式定义作物信息（生长阶段、施肥指南、病虫害等）

### 自定义API模型

支持为不同功能配置不同的模型：
- `LLM_*` - 对话模型（文本生成）
- `VISION_*` - 视觉模型（图片分析）
- `EMBEDDING_*` - 向量模型（知识库检索）

## 数据存储

所有数据以JSON格式本地存储：
- `data/reminders.json` - 提醒数据
- `data/planting_tasks.json` - 种植任务
- `data/planting_progress.json` - 种植进度

无需数据库，部署简单。

## License

MIT License

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Streamlit](https://github.com/streamlit/streamlit)
