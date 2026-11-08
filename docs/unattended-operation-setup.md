# 无人值守部署与验收指南

本指南介绍当前代码支持的配置和检查方法。上线前请先用隔离环境、模拟设备和测试通知地址验证，再由现场负责人按实际设备、作物及当地条件确定参数。软件测试和模拟演示不等于 PLC、固件、传感器或现场失联安全认证；这些仍须在真实设备和现场环境中单独验收。

## 1. 模型与摄像头

HMPD-Net 已接入 PyTorch 推理路径。本次提供的检查点使用 ConvNeXt V2 Base，输入 256×256，包含 47 个联合类别及 16 类作物。将可信的 `hmpd_net.pth` 放在项目根目录下 `models/weights/hmpd_net.pth`；`.pth` 被 `.gitignore` 忽略，不应提交到 Git。保留原始权重和部署备份，并在目标环境核对来源和文件摘要。需要 PyTorch、torchvision、timm 等模型运行依赖，摄像头驱动还需要 OpenCV。

默认 `DL_MODELS_DIR=models/weights`、`DL_DEFAULT_MODEL=hmpd_net`、`DL_DEVICE=cpu`。模型注册会按权重格式加载 Torch 后端；可按部署环境设置 `DL_DEVICE=cuda`。`ENABLE_IMAGE_ANALYSIS` 未显式设置时会检查默认权重文件是否存在，显式设置时 `true` 才启用。权重存在不代表模型已成功加载；应检查后端启动日志及一次隔离图片推理结果。模型不可用时系统可能回退到其他已加载模型，应核对实际使用的模型和类别，而不能仅凭配置名认定 HMPD 正在工作。

摄像设备 `driver` 为 `camera`，能力包含 `capture`，`connection.camera_type` 可为 `usb`、`ip` 或 `esp32cam`，`connection.source` 分别填写 USB 索引或设备路径、网络流地址、HTTP MJPEG 地址。按需填写 `connection.username` 和 `connection.password`，不要把凭据写进共享文档或提交。摄像头应明确绑定地块；巡检使用该地块当前作物检查图片识别结果。缺少有效图像时间、图像过期、作物不一致或无法确认一致时需要人工复核。

`CROP_MONITOR_MIN_CONFIDENCE` 默认 `0.7`，低于阈值的结果标为未知，不给出自动操作建议；无有效预测也按未知处理。这个阈值**尚未校准**，不能把高于阈值当作确诊，也不能替代病害现场诊断。根据目标作物、摄像条件和独立标注样本评估阈值。

## 2. 设备安全契约

在设备配置的 `connection` 中填写契约。下例为结构示意，只有设备固件和协议真实实现并通过现场验证后才能声明；`timeout_seconds` 必须按现场需求确定：

```json
{
  "autonomy": {
    "local_timed_stop": {"duration_param": "duration", "unit": "minutes"},
    "link_loss_stop": {"mode": "firmware_watchdog", "timeout_seconds": 30},
    "execution_id": {"field": "execution_id", "deduplicates": true},
    "ack": {"execution_id_field": "execution_id", "success_field": "success"}
  }
}
```

端侧必须真正按分钟执行定时停止、通信中断时由固件看门狗停止，并对同一 `execution_id` 去重。回执须返回相同 `execution_id` 及 `success`；MQTT 还须在 `ack` 中填写具体 `topic`，不能使用 `+` 或 `#` 通配符。发布成功本身不是执行确认。模拟器内建测试契约；当前真实设备无人值守启动门禁只接线 HTTP、CoAP、MQTT，且要求上述契约。Modbus、OPC UA 等驱动即使能连接，也不能因此视为已支持无人值守启动。停止入口不受启动契约限制，但仍须核对实际停机回执与现场状态。

动作参数经过绝对边界及适用范围的策略检查，自动操作涉及高风险参数时还需明确配置农艺范围。以下是某设备或地块策略的 `limits` 片段，仅展示格式，不是推荐值：

```json
{
  "parameter_ranges": {
    "target_temp": [18, 28],
    "concentration_percent": [0, 2]
  },
  "max_duration_per_use_minutes": 20,
  "max_duration_per_day_minutes": 60
}
```

范围填写 `[下限, 上限]`，不得突破代码的绝对边界；可按设备和农艺条件收紧。支持的参数包括 `target_temp`、`concentration_percent`、`brightness_percent`、`speed_percent`、`power_percent`、`position_percent`、`target_soil_moisture`、`target_lux` 和 `amount_kg`。连续启动须提供正的有限时长；安全策略、设备能力、时段和区域互斥等门禁仍会生效。

## 3. 效果复查与地块数据

为作业设备配置 `connection.feedback`，指定同一地块的传感器及判据。例如：

```json
{
  "feedback": {
    "sensor_id": "soil_sensor_01",
    "field": "soil_moisture",
    "min_delta": 5,
    "direction": "increase",
    "target_min": 30,
    "max_age_seconds": 300,
    "settle_seconds": 30
  }
}
```

这是格式示意，数值须由现场人员确定。`sensor_id` 和作业设备必须明确绑定本次巡检的同一 `plot_id`。`min_delta` 是正数，方向为 `increase` 或 `decrease`；也可使用 `target_min` 和／或 `target_max`，多个配置的判据需要全部满足。传感器读数必须质量合格、带实际观测时间，且严格晚于作业基线和设备完成时间 `ended_at`；设备状态必须证明相同运行编号已完成并提供有效完成时间。缺少判据、状态或有效观测时标为 `unverified` 并告警，不能显示维护成功。效果不达标时，系统仅在设备仍运行且运行编号匹配时尝试停止，需人工检查停止是否确认。

绑定计划后才按地块分析生长阶段；未绑定的进度仍是个人计划，不会猜测它属于哪块田。计划中的非立即操作没有明确 `scheduled_at` 时进入待安排，不会立即下发。依赖作业的后续动作必须等待前项的实际完成证据；启动 ACK 不代表作业完成，时间不明或证据不足时应检查待安排状态并人工处理。

自主巡检只接受当前账号所属地块坐标对应的一小时内天气观测，使用服务商实际观测时间；缺少地块坐标或有效观测时显示未评估。当前预报接口没有可信生成时间，共享持续天气历史也没有地块归属，因此这两类数据暂不进入自动控制决策。规则排期到期时会重新匹配当前条件，观测不可用或条件不再满足时不启动设备。

## 4. 调度与通知

后端和调度器是独立进程。后端可按项目现有方式从项目根目录启动：

```powershell
python app/start.py backend
```

在目标部署环境确认设备、策略、通知渠道和自动化范围后，在**另一个终端**从项目根目录启动唯一调度器进程：

```powershell
$env:ENABLE_SCHEDULER = "true"
python -m app.scheduler_runner
```

该命令会启用定时任务并可能下发设备动作和发送通知，不要在仅需查看配置时执行。`ENABLE_SCHEDULER` 默认关闭；服务进程与调度进程要使用同一数据库及一致的相关环境配置。单机实例锁默认在 `data/.scheduler.lock`，可用 `SCHEDULER_LOCK_FILE` 指定；该锁不提供跨主机单实例保证，跨主机部署须另行保证只有一个调度进程。正常退出可使用 Ctrl+C。

调度器每 20 秒写心跳、每 5 秒处理维护任务和告警；`GET /api/autonomous/status` 区分配置状态 `configured`、运行状态 `running` 与心跳状态。心跳超过 60 秒时不应视为运行中；后端进程的配置也须正确，才能准确解释 `configured`。可通过已登录应用或相应用户上下文查看：

- `GET /api/autonomous/jobs`：排期、待安排、复查及其错误状态。
- `POST /api/autonomous/jobs/{job_id}/schedule`：提交 `{"due_at":"未来的 ISO 8601 时间"}`，仅可安排本用户状态为 `awaiting_schedule` 的动作；`review` 复查建议不能直接安排，需先制定明确操作与判据。
- `GET /api/autonomous/alerts`：告警及送达、处理状态。
- `GET /api/autonomous/reports/{cycle_id}`：巡检报告，包括读数、质量和风险评估质量。
- `GET /api/actions/log`、`GET /api/actions/pending`：设备操作日志和待确认操作。

设置 `AUTONOMY_ALERT_WEBHOOK` 后，维护告警通过 HTTP POST 发送；生产地址须使用 HTTPS，本机测试允许 `http://127.0.0.1:...` 或 `http://localhost:...`。无渠道时告警保留待处理，不会标为已送达。通知服务返回 2xx 且回执没有明确拒绝时才视为投递成功；若 JSON 回执含 `success: false` 或 `ok: false`，即使 HTTP 2xx 也视为拒绝。投递成功不代表人员已经阅读；人工处理后可通过 `POST /api/autonomous/alerts/{alert_id}/ack` 记录确认。请在隔离地址验证幂等键、失败重试与接收方处理，避免对真实人员发送测试通知。

## 5. 验收与异常处理

1. 检查权重、依赖、实际加载模型、作物类别、阈值和摄像头时间；用已知图片验证不一致、低置信度、过期图片不会变成确诊。
2. 对每块地核对作物、传感器、设备和计划绑定；验证读数时间、单位、范围、质量及病害风险是否已评估。天气不足时风险为未评估，不是无风险。
3. 在隔离模拟环境验证策略范围、自动启动拒绝、时间和依赖门禁、排期恢复、效果未核验和告警待送达。检查心跳、任务和报告；未配置通知渠道时不得显示已送达。
4. 在受控现场，由具备资质的人员独立验收本地定时停止、网络丢失保护、端侧执行编号去重、回执匹配、断电重启、手动和紧急停止、区域互斥以及传感器失效。记录证据和恢复流程后再考虑启用真实自动化。
5. 对 `uncertain` 或超时结果，先到设备中心使用“只读核对”，或调用 `POST /api/devices/{device_id}/reconcile`；该入口读取关联运行状态，不重发原指令。证据不足时继续人工核查，不要反复确认或重新启动。必要时使用现场安全停止措施，再核对设备实际停机和记录。
