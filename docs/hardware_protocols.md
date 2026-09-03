# 软硬件协议接入说明

系统通过 `BaseDeviceDriver` 统一协议行为，当前支持 MQTT、HTTP REST、Modbus RTU/TCP、CoAP、OPC UA 和摄像头。设备配置保存在数据库的 `connection` JSON 字段中。

## 选型建议

| 场景 | 推荐协议 | 说明 |
|---|---|---|
| ESP32、树莓派、无线节点 | MQTT | 支持异步状态上报；生产环境建议 TLS + 独立账号 |
| 已提供 Web API 的设备 | HTTP REST | 接入简单，适合局域网网关和智能插座 |
| PLC、变频器、RS-485 传感器 | Modbus RTU/TCP | 需确认寄存器表、字节序和缩放系数 |
| 低功耗、受限网络传感器 | CoAP/CoAPS | 报文开销小；公网使用 CoAPS 或安全网关 |
| PLC、SCADA、工业数据模型 | OPC UA | 通过白名单节点映射读写，避免开放任意节点 |

Zigbee、LoRaWAN 等链路通常先接入网关，再由网关通过 MQTT、HTTP 或 OPC UA 对接本系统，这样更便于鉴权、审计和协议升级。

## CoAP 配置

```json
{
  "driver": "coap",
  "connection": {
    "base_uri": "coap://192.168.1.50:5683",
    "command_path": "/command",
    "state_path": "/state",
    "auth_token": null
  }
}
```

- `GET /state` 应返回 JSON 对象，例如 `{"temperature": 24.2, "humidity": 68}`。
- `POST /command` 接收 `device_id`、`command`、`params` 和 `timestamp`，返回 `{"success": true, "message": "ok"}`。
- `auth_token` 是兼容简单设备的应用层字段，不替代 CoAPS/DTLS。

## OPC UA 配置

```json
{
  "driver": "opcua",
  "connection": {
    "endpoint": "opc.tcp://192.168.1.60:4840",
    "username": "operator",
    "password": "由部署环境安全保存",
    "security_string": "Basic256Sha256,SignAndEncrypt,client-cert.pem,client-key.pem",
    "command_nodes": {
      "start": {"node_id": "ns=2;s=Pump.Start", "value": true},
      "stop": {"node_id": "ns=2;s=Pump.Start", "value": false},
      "set_speed": {"node_id": "ns=2;s=Pump.Speed", "variant_type": "Int16"}
    },
    "state_nodes": {
      "status": "ns=2;s=Pump.Status",
      "temperature": "ns=2;s=Sensor.Temperature"
    }
  }
}
```

调用 `set_speed` 时可在 `params.value` 中传入本次写入值。未列入 `command_nodes` 的指令会被拒绝。

## Modbus 寄存器约定

当前通用驱动与项目模拟器采用以下布局：

- HR[0]：设备状态，0 关机、1 待机、2 工作中、3 故障；
- HR[1]：电源，0 关闭、1 开启；
- HR[2]：命令码；
- HR[3]：持续时间（秒）；
- HR[10:15]：温度、湿度、土壤湿度、pH、光照。

真实设备寄存器表不一致时，应新增设备专用映射，不能直接假设本约定适用于所有 PLC。

## 安全边界

- 不要把 MQTT、Modbus、CoAP、OPC UA 或摄像头端口直接暴露到公网。
- MQTT 优先启用 TLS，按设备分配最小权限主题；OPC UA 使用独立只读/控制账号。
- 密码和证书路径只保存在部署环境或受控数据库中，不写入 Git。
- 控制节点和命令使用白名单；高风险动作继续经过规则引擎和确认队列。
