"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  Activity,
  Bot,
  Camera,
  Check,
  ChevronRight,
  CircleOff,
  Code2,
  Cpu,
  Droplets,
  Edit3,
  FileJson,
  Gauge,
  Play,
  Plus,
  Power,
  RefreshCw,
  Save,
  Server,
  Settings2,
  ShieldCheck,
  Square,
  Timer,
  TestTube2,
  Trash2,
  Wifi,
  X,
} from "lucide-react";
import { ApiError, get, post, remove } from "../api";
import type { Device, Field, JsonMap, Rule } from "../types";
import {
  Card,
  Empty,
  ErrorState,
  Loading,
  Notice,
  PageHeader,
  StatusPill,
} from "./Common";
import { PendingActionEditor } from "./PendingActionEditor";

type DeviceSummaryKind = "all" | "online" | "pending" | "today";

function useEscapeClose(onClose: () => void) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);
}

export function DevicesPage() {
  const [devices, setDevices] = useState<Device[]>([]);
  const [fields, setFields] = useState<Field[]>([]);
  const [pending, setPending] = useState<JsonMap[]>([]);
  const [logs, setLogs] = useState<JsonMap[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [snapshot, setSnapshot] = useState("");
  const [summary, setSummary] = useState<DeviceSummaryKind | null>(null);
  const [editingDevice, setEditingDevice] = useState<Device | null>(null);
  const [editingAction, setEditingAction] = useState<JsonMap | null>(null);
  const [busy, setBusy] = useState("");
  const load = useCallback(async () => {
    setError("");
    try {
      const [deviceRows, fieldRows, pendingRows, logRows] = await Promise.all([
        get<Device[]>("/api/devices"),
        get<Field[]>("/api/fields"),
        get<JsonMap[]>("/api/actions/pending"),
        get<JsonMap[]>("/api/actions/log?limit=30"),
      ]);
      setDevices(deviceRows);
      setFields(fieldRows);
      setPending(
        pendingRows.filter((item) =>
          ["pending", "failed"].includes(String(item.status || "pending")),
        ),
      );
      setLogs(logRows);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "设备读取失败");
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);
  async function command(device: Device, name: string, params: JsonMap = {}) {
    setBusy(device.device_id);
    try {
      await post(`/api/devices/${device.device_id}/command`, {
        command: name,
        params: JSON.stringify(params),
      });
      await load();
    } finally {
      setBusy("");
    }
  }
  async function takeSnapshot(device: Device) {
    setBusy(device.device_id);
    try {
      const result = await get<JsonMap>(
        `/api/devices/${device.device_id}/snapshot`,
      );
      if (result.success && result.image_base64)
        setSnapshot(
          `data:${result.mime_type || "image/jpeg"};base64,${result.image_base64}`,
        );
      else setError(result.error || "拍照失败");
    } finally {
      setBusy("");
    }
  }
  async function decide(id: string, accept: boolean) {
    setBusy(`action:${id}`);
    setError("");
    try {
      const result = await post<JsonMap>(
        `/api/actions/${id}/${accept ? "confirm" : "reject"}`,
      );
      if (!result.success) {
        throw new Error(result.message || result.error || "设备操作处理失败");
      }
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "设备操作处理失败");
    } finally {
      setBusy("");
    }
  }
  if (loading) return <Loading label="正在连接设备中心" />;
  const onlineDevices = devices.filter(
    (device) => !["offline", "error", "disconnected"].includes(device.status),
  );
  const todayLogs = logs.filter((log) =>
    String(log.timestamp || "").startsWith(
      new Date().toISOString().slice(0, 10),
    ),
  );
  return (
    <>
      <PageHeader
        eyebrow="FARM AUTOMATION"
        title="设备中心"
        description="统一接入传感器、灌溉、施肥、通风、补光与摄像设备。"
        actions={
          <>
            <button
              className="secondary-button"
              onClick={async () => {
                await post("/api/devices/refresh");
                await load();
              }}
            >
              <RefreshCw />
              重新连接
            </button>
            <button
              className="primary-button"
              onClick={() => setShowCreate(true)}
            >
              <Plus />
              注册设备
            </button>
          </>
        }
      />
      {error && <ErrorState message={error} retry={load} />}
      <div className="metric-grid four">
        <button
          type="button"
          className="card device-metric"
          onClick={() => setSummary("all")}
          aria-label="查看全部设备详情"
        >
          <span className="device-metric-icon">
            <Cpu />
          </span>
          <span className="device-metric-copy">
            <small>设备总数</small>
            <strong>{devices.length}</strong>
          </span>
          <ChevronRight className="device-metric-arrow" />
        </button>
        <button
          type="button"
          className="card device-metric online"
          onClick={() => setSummary("online")}
          aria-label="查看在线设备详情"
        >
          <span className="device-metric-icon">
            <Wifi />
          </span>
          <span className="device-metric-copy">
            <small>在线设备</small>
            <strong>{onlineDevices.length}</strong>
          </span>
          <ChevronRight className="device-metric-arrow" />
        </button>
        <button
          type="button"
          className="card device-metric warning"
          onClick={() => setSummary("pending")}
          aria-label="查看待确认动作详情"
        >
          <span className="device-metric-icon">
            <ShieldCheck />
          </span>
          <span className="device-metric-copy">
            <small>待确认动作</small>
            <strong>{pending.length}</strong>
          </span>
          <ChevronRight className="device-metric-arrow" />
        </button>
        <button
          type="button"
          className="card device-metric"
          onClick={() => setSummary("today")}
          aria-label="查看今日动作详情"
        >
          <span className="device-metric-icon">
            <Activity />
          </span>
          <span className="device-metric-copy">
            <small>今日动作</small>
            <strong>{todayLogs.length}</strong>
          </span>
          <ChevronRight className="device-metric-arrow" />
        </button>
      </div>
      {pending.length > 0 && (
        <Card className="pending-card" title="需要你确认">
          <div className="pending-list">
            {pending.map((item) => (
              <div
                key={item.id}
                className={item.status === "failed" ? "failed" : ""}
              >
                <span>
                  <ShieldCheck />
                </span>
                <div>
                  <b>{item.description || item.command || "设备操作"}</b>
                  <small>
                    {item.device_id} ·{" "}
                    {item.created_at || item.timestamp || "刚刚"}
                  </small>
                  {item.status === "failed" && (
                    <small className="pending-error">
                      {item.last_error || "上次执行失败，请确认设备状态后重试"}
                    </small>
                  )}
                </div>
                <button
                  className="secondary-button"
                  onClick={() => setEditingAction(item)}
                  disabled={busy === `action:${item.id}`}
                >
                  <Edit3 />
                  修改参数
                </button>
                <button
                  className="secondary-button"
                  onClick={() => decide(item.id, false)}
                  disabled={busy === `action:${item.id}`}
                >
                  <X />
                  拒绝
                </button>
                <button
                  className="primary-button"
                  onClick={() => decide(item.id, true)}
                  disabled={busy === `action:${item.id}`}
                >
                  <Check />
                  {item.status === "failed" ? "重试执行" : "确认执行"}
                </button>
              </div>
            ))}
          </div>
        </Card>
      )}
      <div className="device-grid">
        {devices.length ? (
          devices.map((device) => (
            <DeviceCard
              key={device.device_id}
              device={device}
              busy={busy === device.device_id}
              onSnapshot={() => takeSnapshot(device)}
              onCommand={(name, params) => command(device, name, params)}
              onEdit={() => setEditingDevice(device)}
              onDelete={async () => {
                if (confirm("删除这个自定义设备？")) {
                  await remove(`/api/devices/${device.device_id}`);
                  await load();
                }
              }}
            />
          ))
        ) : (
          <Card className="full-span">
            <Empty
              title="还没有注册设备"
              body="可以添加模拟器、MQTT、HTTP、Modbus、CoAP 或 OPC UA 设备。"
            />
          </Card>
        )}
      </div>
      <Card title="最近执行日志">
        <div className="log-table">
          <div className="table-head">
            <span>时间</span>
            <span>设备</span>
            <span>动作</span>
            <span>结果</span>
          </div>
          {logs.length ? (
            logs.slice(0, 12).map((log, index) => (
              <div className="table-row" key={index}>
                <span>{String(log.timestamp || "").slice(0, 19) || "—"}</span>
                <b>{log.device_id || "系统"}</b>
                <span>
                  {log.command || log.action || log.description || "—"}
                </span>
                <StatusPill
                  tone={
                    log.success === false || log.status === "failed"
                      ? "danger"
                      : "success"
                  }
                >
                  {log.status || (log.success === false ? "失败" : "完成")}
                </StatusPill>
              </div>
            ))
          ) : (
            <Empty title="暂无设备日志" body="设备执行动作后会记录在这里。" />
          )}
        </div>
      </Card>
      {showCreate && (
        <DeviceCreator
          fields={fields}
          onClose={() => setShowCreate(false)}
          onSaved={async () => {
            setShowCreate(false);
            await load();
          }}
        />
      )}{" "}
      {editingDevice && (
        <DeviceEditor
          device={editingDevice}
          fields={fields}
          onClose={() => setEditingDevice(null)}
          onSaved={async () => {
            setEditingDevice(null);
            await load();
          }}
        />
      )}
      {editingAction && (
        <PendingActionEditor
          action={editingAction}
          onClose={() => setEditingAction(null)}
          onSaved={async () => {
            setEditingAction(null);
            await load();
          }}
        />
      )}
      {snapshot && (
        <div className="modal-backdrop" onClick={() => setSnapshot("")}>
          <div className="modal snapshot-modal">
            <div className="modal-head">
              <h2>摄像头实时快照</h2>
              <button className="icon-button" onClick={() => setSnapshot("")}>
                ×
              </button>
            </div>
            {/* 摄像头快照是动态 Base64 数据，不经过远程图片优化代理。 */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={snapshot} alt="设备快照" />
          </div>
        </div>
      )}
      {summary && (
        <DeviceSummaryModal
          kind={summary}
          devices={devices}
          onlineDevices={onlineDevices}
          pending={pending}
          todayLogs={todayLogs}
          busy={busy}
          onDecide={decide}
          onEdit={(action) => {
            setSummary(null);
            setEditingAction(action);
          }}
          onClose={() => setSummary(null)}
        />
      )}
    </>
  );
}

function DeviceSummaryModal({
  kind,
  devices,
  onlineDevices,
  pending,
  todayLogs,
  busy,
  onDecide,
  onEdit,
  onClose,
}: {
  kind: DeviceSummaryKind;
  devices: Device[];
  onlineDevices: Device[];
  pending: JsonMap[];
  todayLogs: JsonMap[];
  busy: string;
  onDecide: (id: string, accept: boolean) => Promise<void>;
  onEdit: (action: JsonMap) => void;
  onClose: () => void;
}) {
  const config = {
    all: {
      title: "全部设备详情",
      description: `当前共接入 ${devices.length} 台设备`,
    },
    online: {
      title: "在线设备详情",
      description: `当前有 ${onlineDevices.length} 台设备在线`,
    },
    pending: {
      title: "待确认动作",
      description: `共有 ${pending.length} 项动作等待处理`,
    },
    today: {
      title: "今日动作详情",
      description: `今天共记录 ${todayLogs.length} 项设备动作`,
    },
  }[kind];
  const visibleDevices = kind === "online" ? onlineDevices : devices;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal wide-modal device-summary-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="device-summary-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-head device-summary-head">
          <div>
            <h2 id="device-summary-title">{config.title}</h2>
            <p>{config.description}</p>
          </div>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="关闭详情"
          >
            ×
          </button>
        </div>

        {(kind === "all" || kind === "online") &&
          (visibleDevices.length ? (
            <div className="summary-device-list">
              {visibleDevices.map((device) => {
                const offline = ["offline", "error", "disconnected"].includes(
                  device.status,
                );
                const stateRows = deviceParameterEntries(device);
                return (
                  <article
                    className="summary-device-item"
                    key={device.device_id}
                  >
                    <div className="summary-device-head">
                      <span className={`device-icon ${device.status}`}>
                        <DeviceIcon capabilities={device.capabilities} />
                      </span>
                      <div>
                        <b>{device.name}</b>
                        <small>{device.device_id}</small>
                      </div>
                      <StatusPill tone={offline ? "danger" : "success"}>
                        {deviceStatusLabel(device.status)}
                      </StatusPill>
                    </div>
                    <div className="summary-device-meta">
                      <span>
                        <small>所属地块</small>
                        <b>{device.plot_name || device.location || "未绑定"}</b>
                      </span>
                      <span>
                        <small>接入方式</small>
                        <b>{device.driver || "未知"}</b>
                      </span>
                      <span>
                        <small>设备能力</small>
                        <b>
                          {device.capabilities
                            ?.map(capabilityLabel)
                            .join(" · ") || "基础控制"}
                        </b>
                      </span>
                    </div>
                    <div className="summary-sensor-row">
                      {stateRows.length ? (
                        stateRows.map(([key, value]) => (
                          <span key={key}>
                            <small>{sensorLabel(key)}</small>
                            <b>{formatDeviceValue(key, value)}</b>
                          </span>
                        ))
                      ) : (
                        <small>暂无实时数据</small>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          ) : (
            <Empty
              title={kind === "online" ? "暂无在线设备" : "暂无设备"}
              body="设备状态更新后会显示在这里。"
            />
          ))}

        {kind === "pending" &&
          (pending.length ? (
            <div className="summary-action-list">
              {pending.map((item) => {
                const actionId = String(item.id || "");
                const actionBusy = busy === `action:${actionId}`;
                return (
                  <article
                    key={actionId}
                    className={item.status === "failed" ? "failed" : ""}
                  >
                    <span className="summary-action-icon">
                      <ShieldCheck />
                    </span>
                    <div>
                      <b>{item.description || item.command || "设备操作"}</b>
                      <small>
                        {item.device_id || "未知设备"} ·{" "}
                        {item.created_at || item.timestamp || "刚刚"}
                      </small>
                      {item.status === "failed" && (
                        <small className="pending-error">
                          {item.last_error || "上次执行失败，可检查设备后重试"}
                        </small>
                      )}
                    </div>
                    <div className="summary-action-buttons">
                      <button
                        className="secondary-button"
                        onClick={() => onEdit(item)}
                        disabled={actionBusy}
                      >
                        <Edit3 />
                        修改参数
                      </button>
                      <button
                        className="secondary-button"
                        onClick={() => onDecide(actionId, false)}
                        disabled={actionBusy}
                      >
                        <X />
                        拒绝
                      </button>
                      <button
                        className="primary-button"
                        onClick={() => onDecide(actionId, true)}
                        disabled={actionBusy}
                      >
                        <Check />
                        {actionBusy
                          ? "处理中"
                          : item.status === "failed"
                            ? "重试执行"
                            : "确认执行"}
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          ) : (
            <Empty
              title="没有待确认动作"
              body="需要人工确认的设备操作会显示在这里。"
            />
          ))}

        {kind === "today" &&
          (todayLogs.length ? (
            <div className="summary-log-list">
              {todayLogs.map((log, index) => (
                <article key={`${log.id || log.timestamp || "log"}-${index}`}>
                  <span className="summary-log-time">
                    {String(log.timestamp || "").slice(11, 19) || "—"}
                  </span>
                  <div>
                    <b>{log.device_name || log.device_id || "系统"}</b>
                    <small>
                      {log.command ||
                        log.action ||
                        log.description ||
                        "设备动作"}
                    </small>
                  </div>
                  <StatusPill
                    tone={
                      log.success === false || log.status === "failed"
                        ? "danger"
                        : "success"
                    }
                  >
                    {log.status || (log.success === false ? "失败" : "完成")}
                  </StatusPill>
                </article>
              ))}
            </div>
          ) : (
            <Empty
              title="今天还没有设备动作"
              body="设备执行操作后会记录在这里。"
            />
          ))}
      </div>
    </div>
  );
}

function DeviceCard({
  device,
  busy,
  onSnapshot,
  onCommand,
  onEdit,
  onDelete,
}: {
  device: Device;
  busy: boolean;
  onSnapshot: () => void;
  onCommand: (name: string, params?: JsonMap) => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const sensorEntries = deviceParameterEntries(device);
  const offline = ["offline", "error", "disconnected"].includes(device.status);
  const isCamera = device.capabilities?.includes("capture");
  const capabilities =
    device.capabilities?.slice(0, 3).map(capabilityLabel).join(" · ") ||
    "基础控制";

  return (
    <Card className="device-card">
      <div className="device-head">
        <div className="device-identity">
          <span className={`device-icon ${device.status}`}>
            <DeviceIcon capabilities={device.capabilities} />
          </span>
          <div className="device-title">
            <b title={device.name}>{device.name}</b>
            <small title={device.plot_name || device.location || "未绑定地块"}>
              {device.plot_name || device.location || "未绑定地块"}
            </small>
          </div>
        </div>
        <div className="device-status">
          <StatusPill tone={offline ? "danger" : "success"}>
            {deviceStatusLabel(device.status)}
          </StatusPill>
        </div>
      </div>

      {isCamera && <CameraLiveView device={device} offline={offline} />}

      <div className="device-section-heading">
        <span>实时数据</span>
        <small>{sensorEntries.length} 项</small>
      </div>
      <div
        className={`sensor-grid ${sensorEntries.length ? "" : "empty"} ${sensorEntries.length <= 2 ? "compact" : ""}`}
      >
        {sensorEntries.length ? (
          sensorEntries.map(([key, value]) => (
            <div key={key}>
              <span>{sensorLabel(key)}</span>
              <b title={formatDeviceValue(key, value)}>
                {formatDeviceValue(key, value)}
              </b>
            </div>
          ))
        ) : (
          <div className="sensor-empty">
            <Gauge />
            <span>暂无实时数据</span>
          </div>
        )}
      </div>

      <div className="device-details">
        <div>
          <small>接入方式</small>
          <span title={device.driver || "未知"}>
            <Server />
            {device.driver || "未知"}
          </span>
        </div>
        <div>
          <small>设备能力</small>
          <span title={capabilities}>{capabilities}</span>
        </div>
      </div>

      <DeviceRunControls
        key={`${device.device_id}:${device.capabilities?.join("|")}`}
        device={device}
        busy={busy}
        onStart={(params) => onCommand("start", params)}
      />

      <div className="device-actions">
        {isCamera && (
          <button onClick={onSnapshot} disabled={busy}>
            <Camera />
            <span>拍照</span>
          </button>
        )}
        {isControllableDevice(device) && (
          <>
            <button onClick={() => onCommand("power_on")} disabled={busy}>
              <Power />
              <span>通电</span>
            </button>
            <button onClick={() => onCommand("stop")} disabled={busy}>
              <Square />
              <span>停止</span>
            </button>
          </>
        )}
        <button onClick={onEdit} disabled={busy || device.editable === false}>
          <Edit3 />
          <span>配置</span>
        </button>
        <button className="delete" onClick={onDelete} disabled={busy}>
          <Trash2 />
          <span>删除</span>
        </button>
      </div>
    </Card>
  );
}

function CameraLiveView({
  device,
  offline,
}: {
  device: Device;
  offline: boolean;
}) {
  const [frame, setFrame] = useState("");
  const [frameTime, setFrameTime] = useState("");
  const [frameError, setFrameError] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const refreshFrame = useCallback(async () => {
    if (offline) return;
    setRefreshing(true);
    try {
      const result = await get<JsonMap>(
        `/api/devices/${device.device_id}/snapshot`,
      );
      if (!result.success || !result.image_base64) {
        throw new Error(result.error || "暂时无法获取摄像头画面");
      }
      setFrame(
        `data:${result.mime_type || "image/jpeg"};base64,${result.image_base64}`,
      );
      setFrameTime(result.timestamp || new Date().toISOString());
      setFrameError("");
    } catch (reason) {
      setFrameError(
        reason instanceof Error ? reason.message : "暂时无法获取摄像头画面",
      );
    } finally {
      setRefreshing(false);
    }
  }, [device.device_id, offline]);

  useEffect(() => {
    if (offline) {
      setFrame("");
      setFrameError("摄像头当前离线");
      return;
    }
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function poll() {
      await refreshFrame();
      if (!stopped) timer = setTimeout(poll, 5000);
    }
    poll();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [offline, refreshFrame]);

  return (
    <div className="camera-live-panel">
      <div className="camera-live-head">
        <span>
          <i className={!offline && frame ? "live" : ""} />
          实时画面
        </span>
        <button
          type="button"
          onClick={refreshFrame}
          disabled={offline || refreshing}
          aria-label="刷新摄像头画面"
        >
          <RefreshCw className={refreshing ? "spinning" : ""} />
          {frameTime ? `${String(frameTime).slice(11, 19)} 更新` : "每 5 秒刷新"}
        </button>
      </div>
      <div className="camera-live-frame">
        {frame ? (
          <>
            {/* 摄像头画面来自已配置设备的动态 Base64 数据。 */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={frame} alt={`${device.name} 当前画面`} />
            <span className="camera-live-badge">LIVE</span>
          </>
        ) : (
          <div className="camera-live-empty">
            <Camera />
            <b>{refreshing ? "正在连接摄像头" : "暂无实时画面"}</b>
            <small>{frameError || "正在等待摄像头返回第一帧"}</small>
          </div>
        )}
      </div>
    </div>
  );
}

type DeviceParameterDefinition = {
  key: string;
  label: string;
  unit: string;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
};

const actuatorCapabilities = [
  "irrigate",
  "fertigate",
  "ventilate",
  "heat",
  "cool",
  "shade",
  "light",
];

function isControllableDevice(device: Device) {
  return actuatorCapabilities.some((capability) =>
    device.capabilities?.includes(capability),
  );
}

function deviceParameterEntries(device: Device): [string, unknown][] {
  const entries = Object.entries(device.state || {}).filter(([, value]) =>
    ["string", "number", "boolean"].includes(typeof value),
  );
  const known = new Set(entries.map(([key]) => key));
  for (const sensor of device.sensors || []) {
    if (!known.has(sensor)) entries.push([sensor, null]);
  }
  return entries;
}

function formatDeviceValue(key: string, value: unknown) {
  if (value === null || value === undefined || value === "") return "暂无数据";
  if (key === "status") return deviceStatusLabel(String(value));
  if (key === "power" && typeof value === "boolean")
    return value ? "开启" : "关闭";
  const units: Record<string, string> = {
    soil_moisture: "%",
    humidity: "%",
    temperature: "℃",
    target_temp: "℃",
    flow_rate: " L/min",
    light_lux: " lx",
    co2_ppm: " ppm",
    pressure: " kPa",
    wind_speed: " m/s",
    brightness_percent: "%",
    last_duration: " 分钟",
    last_amount_kg: " kg",
    ph: "",
  };
  return `${String(value)}${units[key] || ""}`;
}

function commandParameterDefinitions(
  device: Device,
): DeviceParameterDefinition[] {
  if (!isControllableDevice(device)) return [];
  const definitions: DeviceParameterDefinition[] = [
    {
      key: "duration",
      label: "运行时长",
      unit: "分钟",
      defaultValue: 10,
      min: 1,
      max: 120,
      step: 1,
    },
  ];
  if (device.capabilities?.includes("irrigate")) {
    definitions.push({
      key: "flow_rate",
      label: "目标流量",
      unit: "L/min",
      defaultValue: 12,
      min: 0.1,
      max: 500,
      step: 0.1,
    });
  }
  if (device.capabilities?.includes("fertigate")) {
    definitions.push({
      key: "amount_kg",
      label: "施肥量",
      unit: "kg",
      defaultValue: 1,
      min: 0.1,
      max: 50,
      step: 0.1,
    });
  }
  if (
    device.capabilities?.includes("heat") ||
    device.capabilities?.includes("cool")
  ) {
    definitions.push({
      key: "target_temp",
      label: "目标温度",
      unit: "℃",
      defaultValue: 24,
      min: 0,
      max: 50,
      step: 0.5,
    });
  }
  if (
    device.capabilities?.includes("light") ||
    device.capabilities?.includes("shade")
  ) {
    definitions.push({
      key: "brightness_percent",
      label: "目标亮度",
      unit: "%",
      defaultValue: 70,
      min: 0,
      max: 100,
      step: 1,
    });
  }
  return definitions;
}

function DeviceRunControls({
  device,
  busy,
  onStart,
}: {
  device: Device;
  busy: boolean;
  onStart: (params: JsonMap) => void;
}) {
  const definitions = commandParameterDefinitions(device);
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      definitions.map((definition) => [
        definition.key,
        String(definition.defaultValue),
      ]),
    ),
  );
  if (!definitions.length) return null;

  function startWithParameters() {
    const params = Object.fromEntries(
      definitions.map((definition) => {
        const parsed = Number(values[definition.key]);
        const safeValue = Number.isFinite(parsed)
          ? Math.min(definition.max, Math.max(definition.min, parsed))
          : definition.defaultValue;
        return [definition.key, safeValue];
      }),
    );
    onStart(params);
  }

  return (
    <div className="device-run-panel">
      <div className="device-section-heading">
        <span>运行参数</span>
        <small>启动时应用</small>
      </div>
      <div className="device-run-fields">
        {definitions.map((definition) => (
          <label key={definition.key}>
            <span>{definition.label}</span>
            <div>
              <input
                type="number"
                step="any"
                value={values[definition.key] || ""}
                onChange={(event) =>
                  setValues((current) => ({
                    ...current,
                    [definition.key]: event.target.value,
                  }))
                }
                aria-label={`${device.name}${definition.label}`}
              />
              <small>{definition.unit}</small>
            </div>
          </label>
        ))}
      </div>
      <button
        type="button"
        className="device-start-button"
        onClick={startWithParameters}
        disabled={busy}
      >
        {device.capabilities?.includes("irrigate") ? <Droplets /> : <Timer />}
        {busy ? "执行中" : "按参数启动"}
      </button>
    </div>
  );
}

function deviceStatusLabel(status: string) {
  const labels: Record<string, string> = {
    online: "在线",
    connected: "在线",
    ready: "在线",
    running: "运行中",
    active: "运行中",
    offline: "离线",
    disconnected: "离线",
    error: "异常",
    unknown: "未知",
  };
  return labels[status] || status || "未知";
}

function capabilityLabel(value: string) {
  const labels: Record<string, string> = {
    irrigate: "灌溉",
    fertigate: "施肥",
    ventilate: "通风",
    heat: "加热",
    cool: "降温",
    shade: "遮阳",
    light: "补光",
    capture: "拍照",
    read_sensor: "传感监测",
    power: "电源控制",
    start: "启动",
    stop: "停止",
    control: "设备控制",
  };
  return labels[value] || value;
}

function deviceConnectionPlaceholder(driver: string) {
  const examples: Record<string, string> = {
    simulator: "{}",
    http: '{"base_url":"http://192.168.1.10"}',
    mqtt: '{"host":"localhost","port":1883}',
    modbus:
      '{"mode":"tcp","host":"192.168.1.20","port":502,"slave_id":1}',
    coap: '{"base_uri":"coap://192.168.1.30"}',
    opcua:
      '{"endpoint":"opc.tcp://192.168.1.40:4840","command_nodes":{},"state_nodes":{}}',
    camera: '{"source":"rtsp://192.168.1.50/stream"}',
  };
  return examples[driver] || "{}";
}

function parseRuleParameterValue(value: string): unknown {
  const text = value.trim();
  if (!text) return "";
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function DeviceIcon({ capabilities }: { capabilities: string[] }) {
  if (capabilities?.includes("capture")) return <Camera />;
  if (capabilities?.includes("read_sensor")) return <Gauge />;
  return <Bot />;
}
function sensorLabel(key: string) {
  return (
    (
      {
        power: "电源",
        status: "状态",
        soil_moisture: "土壤湿度",
        temperature: "温度",
        humidity: "空气湿度",
        flow_rate: "流量",
        target_temp: "目标温度",
        light_lux: "光照强度",
        co2_ppm: "二氧化碳",
        pressure: "压力",
        wind_speed: "风速",
        brightness_percent: "亮度",
        last_duration: "上次时长",
        last_amount_kg: "上次施肥量",
        ph: "土壤 pH",
      } as Record<string, string>
    )[key] || key
  );
}

function DeviceCreator({
  fields,
  onClose,
  onSaved,
}: {
  fields: Field[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<JsonMap>({
    device_id: "",
    name: "",
    driver: "simulator",
    plot_id: "",
    zone_id: "",
    location: "",
    capabilities: ["irrigate"],
    connection: "{}",
  });
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  useEscapeClose(onClose);
  const caps = [
    "irrigate",
    "fertigate",
    "ventilate",
    "heat",
    "cool",
    "shade",
    "light",
    "read_sensor",
    "capture",
  ];
  async function save(event: FormEvent) {
    event.preventDefault();
    if (saving) return;
    setError("");
    setSaving(true);
    try {
      const result = await post<JsonMap>("/api/devices", {
        device_id: form.device_id,
        name: form.name,
        driver: form.driver,
        plot_id: form.plot_id,
        zone_id: form.zone_id,
        location:
          form.location ||
          fields.find((field) => field.id === form.plot_id)?.name ||
          "",
        capabilities: form.capabilities,
        sensors: form.capabilities.includes("read_sensor")
          ? ["soil_moisture", "temperature"]
          : [],
        connection: JSON.parse(form.connection || "{}"),
        initial_state: { power: false, status: "powered_off" },
      });
      if (!result.success) throw new Error(result.error || "注册失败");
      onSaved();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "连接参数格式错误");
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <form
        className="modal device-create-modal"
        onSubmit={save}
        role="dialog"
        aria-modal="true"
        aria-labelledby="device-create-title"
      >
        <div className="modal-head">
          <h2 id="device-create-title">注册新设备</h2>
          <button type="button" className="icon-button" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="form-grid two">
          <label>
            设备 ID
            <input
              value={form.device_id}
              onChange={(e) => setForm({ ...form, device_id: e.target.value })}
              placeholder="my_pump_01"
              required
            />
          </label>
          <label>
            设备名称
            <input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="东地块水泵"
              required
            />
          </label>
          <label>
            驱动协议
            <select
              value={form.driver}
              onChange={(e) => setForm({ ...form, driver: e.target.value })}
            >
              {[
                "simulator",
                "mqtt",
                "http",
                "modbus",
                "coap",
                "opcua",
                "camera",
              ].map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            安装位置
            <input
              value={form.location}
              onChange={(e) => setForm({ ...form, location: e.target.value })}
              placeholder="如：东侧温室北区"
            />
          </label>
          <label>
            所属地块
            <select
              value={form.plot_id}
              onChange={(e) => setForm({ ...form, plot_id: e.target.value })}
            >
              <option value="">不绑定</option>
              {fields.map((field) => (
                <option value={field.id} key={field.id}>
                  {field.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            作业分区 ID
            <input
              value={form.zone_id}
              onChange={(event) =>
                setForm({ ...form, zone_id: event.target.value })
              }
              placeholder="如：north_irrigation"
            />
          </label>
        </div>
        <div className="choice-group">
          <span>设备能力</span>
          <div>
            {caps.map((cap) => (
              <button
                type="button"
                className={form.capabilities.includes(cap) ? "selected" : ""}
                key={cap}
                onClick={() =>
                  setForm({
                    ...form,
                    capabilities: form.capabilities.includes(cap)
                      ? form.capabilities.filter((item: string) => item !== cap)
                      : [...form.capabilities, cap],
                  })
                }
              >
                {capabilityLabel(cap)}
              </button>
            ))}
          </div>
        </div>
        <label>
          连接参数（JSON）
          <textarea
            rows={5}
            value={form.connection}
            onChange={(e) => setForm({ ...form, connection: e.target.value })}
            placeholder={deviceConnectionPlaceholder(String(form.driver))}
            spellCheck={false}
          />
          <small>
            可在注册时直接填写；模拟设备没有额外参数时保留空对象即可。
          </small>
        </label>
        {error && <ErrorState message={error} />}
        <button className="primary-button" disabled={saving}>
          <Save />
          {saving ? "正在注册" : "注册设备"}
        </button>
      </form>
    </div>
  );
}

function DeviceEditor({
  device,
  fields,
  onClose,
  onSaved,
}: {
  device: Device;
  fields: Field[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<JsonMap>({
    name: device.name,
    driver: device.driver || "simulator",
    plot_id: device.plot_id || "",
    zone_id: device.zone_id || "",
    location: device.location || "",
    capabilities: device.capabilities || [],
    sensors: device.sensors || [],
    connection: JSON.stringify(device.connection || {}, null, 2),
  });
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  useEscapeClose(onClose);
  const caps = [
    "irrigate",
    "fertigate",
    "ventilate",
    "heat",
    "cool",
    "shade",
    "light",
    "read_sensor",
    "capture",
  ];

  async function save(event: FormEvent) {
    event.preventDefault();
    setError("");
    setSaving(true);
    try {
      const connection = JSON.parse(form.connection || "{}");
      const sensors = form.capabilities.includes("read_sensor")
        ? form.sensors?.length
          ? form.sensors
          : ["soil_moisture", "temperature"]
        : [];
      const result = await post<JsonMap>(
        `/api/devices/${device.device_id}/config`,
        {
          name: form.name,
          driver: form.driver,
          plot_id: form.plot_id,
          zone_id: form.zone_id,
          location:
            form.location ||
            fields.find((field) => field.id === form.plot_id)?.name ||
            "",
          capabilities: form.capabilities,
          sensors,
          connection,
          initial_state: device.initial_state || {
            power: false,
            status: "powered_off",
          },
        },
      );
      if (!result.success) throw new Error(result.error || "保存失败");
      onSaved();
    } catch (reason) {
      if (reason instanceof ApiError && [404, 405].includes(reason.status)) {
        setError("后端尚未加载设备配置接口，请重启 FastAPI 后再保存。");
      } else {
        setError(reason instanceof Error ? reason.message : "连接参数格式错误");
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-backdrop">
      <form
        className="modal device-edit-modal"
        onSubmit={save}
        role="dialog"
        aria-modal="true"
        aria-labelledby="device-edit-title"
      >
        <div className="modal-head">
          <div>
            <h2 id="device-edit-title">修改设备配置</h2>
            <p>保存后设备会使用新配置重新连接。</p>
          </div>
          <button type="button" className="icon-button" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="form-grid two">
          <label>
            设备 ID
            <input value={device.device_id} disabled />
          </label>
          <label>
            设备名称
            <input
              value={form.name}
              onChange={(event) =>
                setForm({ ...form, name: event.target.value })
              }
              required
            />
          </label>
          <label>
            驱动协议
            <select
              value={form.driver}
              onChange={(event) =>
                setForm({ ...form, driver: event.target.value })
              }
            >
              {[
                "simulator",
                "mqtt",
                "http",
                "modbus",
                "coap",
                "opcua",
                "camera",
              ].map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            所属地块
            <select
              value={form.plot_id}
              onChange={(event) =>
                setForm({ ...form, plot_id: event.target.value })
              }
            >
              <option value="">不绑定</option>
              {fields.map((field) => (
                <option value={field.id} key={field.id}>
                  {field.name}
                </option>
              ))}
            </select>
          </label>
          <label className="full-span">
            安装位置
            <input
              value={form.location}
              onChange={(event) =>
                setForm({ ...form, location: event.target.value })
              }
              placeholder="如：东侧温室北区"
            />
          </label>
          <label>
            作业分区 ID
            <input
              value={form.zone_id}
              onChange={(event) =>
                setForm({ ...form, zone_id: event.target.value })
              }
              placeholder="如：north_irrigation"
            />
          </label>
        </div>
        <div className="choice-group">
          <span>设备能力</span>
          <div>
            {caps.map((cap) => (
              <button
                type="button"
                className={form.capabilities.includes(cap) ? "selected" : ""}
                key={cap}
                onClick={() =>
                  setForm({
                    ...form,
                    capabilities: form.capabilities.includes(cap)
                      ? form.capabilities.filter((item: string) => item !== cap)
                      : [...form.capabilities, cap],
                  })
                }
              >
                {capabilityLabel(cap)}
              </button>
            ))}
          </div>
        </div>
        {form.driver !== "simulator" && (
          <label>
            连接参数（JSON）
            <textarea
              rows={7}
              value={form.connection}
              onChange={(event) =>
                setForm({ ...form, connection: event.target.value })
              }
              spellCheck={false}
            />
          </label>
        )}
        {error && <ErrorState message={error} />}
        <div className="editor-actions">
          <button type="button" className="secondary-button" onClick={onClose}>
            取消
          </button>
          <button className="primary-button" disabled={saving}>
            <Save />
            {saving ? "正在保存" : "保存并重新连接"}
          </button>
        </div>
      </form>
    </div>
  );
}

export function RulesPage() {
  const [rules, setRules] = useState<Rule[]>([]);
  const [devices, setDevices] = useState<Device[]>([]);
  const [devicesLoading, setDevicesLoading] = useState(true);
  const [selected, setSelected] = useState<Rule | null>(null);
  const [form, setForm] = useState<JsonMap>({
    name: "",
    enabled: true,
    logic: "AND",
    conditions:
      '[{"type":"sensor","field":"soil_moisture","op":"<","value":30}]',
    device_id: "",
    command: "start",
    parameterName: "duration",
    parameterValue: "30",
    maxUse: 60,
    maxDay: 180,
    ai: false,
  });
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    setDevicesLoading(true);
    try {
      const [ruleRows, deviceRows] = await Promise.all([
        get<Rule[]>("/api/rules"),
        get<Device[]>("/api/devices"),
      ]);
      setRules(ruleRows);
      setDevices(deviceRows);
      setForm((current) =>
        !current.device_id && deviceRows[0]
          ? { ...current, device_id: deviceRows[0].device_id }
          : current,
      );
    } finally {
      setDevicesLoading(false);
    }
  }, []);
  useEffect(() => {
    load().catch((reason) => setError(reason.message));
  }, [load]);
  function edit(rule: Rule) {
    setSelected(rule);
    setForm({
      name: rule.name,
      enabled: rule.enabled,
      logic: rule.trigger?.logic || "AND",
      conditions: JSON.stringify(rule.trigger?.conditions || [], null, 2),
      device_id: rule.action?.device_id || devices[0]?.device_id || "",
      command: rule.action?.command || "start",
      parameterName: Object.keys(rule.action?.params || {})[0] || "duration",
      parameterValue: String(
        Object.values(rule.action?.params || {})[0] ?? 30,
      ),
      maxUse: rule.constraints?.max_duration_per_use || 60,
      maxDay: rule.constraints?.max_duration_per_day || 180,
      ai: rule.ai_enhance?.enabled || false,
    });
  }
  function reset() {
    setSelected(null);
    setForm({
      name: "",
      enabled: true,
      logic: "AND",
      conditions:
        '[{"type":"sensor","field":"soil_moisture","op":"<","value":30}]',
      device_id: devices[0]?.device_id || "",
      command: "start",
      parameterName: "duration",
      parameterValue: "30",
      maxUse: 60,
      maxDay: 180,
      ai: false,
    });
  }
  async function save(event: FormEvent) {
    event.preventDefault();
    setError("");
    if (!String(form.name || "").trim()) {
      setError("请输入规则名称");
      return;
    }
    if (!devices.length || !form.device_id) {
      setError("请先在设备中心注册一个设备");
      return;
    }
    if (form.command === "set_param" && !String(form.parameterName).trim()) {
      setError("请输入要设置的参数名");
      return;
    }
    try {
      const actionParams =
        form.command === "set_param"
          ? {
              [String(form.parameterName).trim()]: parseRuleParameterValue(
                String(form.parameterValue),
              ),
            }
          : form.command === "start"
            ? { duration: 30 }
            : {};
      const payload = {
        name: form.name || "未命名规则",
        enabled: form.enabled,
        trigger: { logic: form.logic, conditions: JSON.parse(form.conditions) },
        action: {
          device_id: form.device_id,
          command: form.command,
          params: actionParams,
        },
        constraints: {
          max_duration_per_use: Number(form.maxUse),
          max_duration_per_day: Number(form.maxDay),
          min_interval_minutes: 120,
          forbidden_hours: [22, 23, 0, 1, 2, 3, 4, 5],
        },
        ai_enhance: {
          enabled: form.ai,
          can_adjust: ["duration"],
          adjust_range: { duration: [-10, 10] },
        },
      };
      const result = selected
        ? await put<JsonMap>(`/api/rules/${selected.id}`, payload)
        : await post<JsonMap>("/api/rules", payload);
      if (!result.success) throw new Error(result.error || "保存失败");
      setMessage("自动规则已保存");
      reset();
      await load();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "触发条件 JSON 格式错误",
      );
    }
  }
  async function testRule(rule: Rule) {
    const result = await post<JsonMap>(`/api/rules/${rule.id}/test`);
    setMessage(
      result.rule_matched
        ? `规则“${rule.name}”当前条件匹配`
        : `规则“${rule.name}”当前条件不匹配`,
    );
  }
  async function deleteRule(rule: Rule) {
    setError("");
    const result = await remove<JsonMap>(`/api/rules/${rule.id}`);
    if (!result.success) {
      throw new Error(result.error || "规则删除失败");
    }
    setRules((current) =>
      current.filter((item) => String(item.id) !== String(rule.id)),
    );
    reset();
    setMessage("规则已彻底删除");
    await load();
  }
  return (
    <>
      <PageHeader
        eyebrow="AUTOMATION RULES"
        title="规则管理"
        description="把传感器条件、目标设备和安全边界组合成可靠的自动化。"
        actions={
          <button className="primary-button" onClick={reset}>
            <Plus />
            新建规则
          </button>
        }
      />
      {message && <Notice>{message}</Notice>}
      {error && <ErrorState message={error} />}
      <div className="rules-layout">
        <Card className="rule-list-card" title={`我的规则 · ${rules.length}`}>
          {rules.length ? (
            <div className="rule-list">
              {rules.map((rule) => (
                <button
                  className={selected?.id === rule.id ? "selected" : ""}
                  onClick={() => edit(rule)}
                  key={rule.id}
                >
                  <span className={rule.enabled ? "enabled" : "disabled"}>
                    {rule.enabled ? <Play /> : <CircleOff />}
                  </span>
                  <div>
                    <b>{rule.name}</b>
                    <small>
                      {rule.action?.device_id || "未绑定设备"} ·{" "}
                      {rule.action?.command || "—"}
                    </small>
                  </div>
                  <ChevronRight />
                </button>
              ))}
            </div>
          ) : (
            <Empty
              title="暂无自动规则"
              body="新建一条规则，让农场开始自动响应。"
            />
          )}
        </Card>
        <Card
          className="rule-editor"
          title={selected ? `编辑 · ${selected.name}` : "新建自动规则"}
        >
          <form onSubmit={save} noValidate>
            <div className="form-grid two">
              <label>
                规则名称
                <input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="小麦自动灌溉"
                  required
                />
              </label>
              <label className="switch-label">
                启用规则
                <button
                  type="button"
                  className={`switch ${form.enabled ? "on" : ""}`}
                  onClick={() => setForm({ ...form, enabled: !form.enabled })}
                >
                  <i />
                </button>
              </label>
            </div>
            <div className="rule-section">
              <span>
                <Activity />
                触发条件
              </span>
              <div className="segmented logic-tabs">
                <button
                  type="button"
                  className={form.logic === "AND" ? "active" : ""}
                  onClick={() => setForm({ ...form, logic: "AND" })}
                >
                  全部满足 AND
                </button>
                <button
                  type="button"
                  className={form.logic === "OR" ? "active" : ""}
                  onClick={() => setForm({ ...form, logic: "OR" })}
                >
                  任一满足 OR
                </button>
              </div>
              <label>
                条件 JSON
                <textarea
                  rows={7}
                  value={form.conditions}
                  onChange={(e) =>
                    setForm({ ...form, conditions: e.target.value })
                  }
                />
              </label>
            </div>
            <div className="rule-section">
              <span>
                <Cpu />
                执行动作
              </span>
              {devicesLoading ? (
                <Loading label="正在读取已注册设备" />
              ) : devices.length ? (
                <div className="form-grid two">
                  <label>
                    目标设备
                    <select
                      value={form.device_id}
                      onChange={(e) =>
                        setForm({ ...form, device_id: e.target.value })
                      }
                    >
                      {devices.map((device) => (
                        <option value={device.device_id} key={device.device_id}>
                          {device.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    指令
                    <select
                      value={form.command}
                      onChange={(e) =>
                        setForm({ ...form, command: e.target.value })
                      }
                    >
                      <option value="start">启动</option>
                      <option value="stop">停止</option>
                      <option value="set_param">设置参数</option>
                    </select>
                  </label>
                </div>
              ) : (
                <Notice tone="danger">请先在设备中心注册一个设备。</Notice>
              )}
              {devices.length > 0 && form.command === "set_param" && (
                <div className="form-grid two rule-parameter-fields">
                  <label>
                    参数名
                    <input
                      value={form.parameterName}
                      onChange={(event) =>
                        setForm({ ...form, parameterName: event.target.value })
                      }
                      placeholder="如：target_temp"
                      required
                    />
                  </label>
                  <label>
                    参数值
                    <input
                      value={form.parameterValue}
                      onChange={(event) =>
                        setForm({ ...form, parameterValue: event.target.value })
                      }
                      placeholder="如：24 或 true"
                      required
                    />
                  </label>
                  <small>
                    数字、true、false 和 JSON 会自动转换，其余内容按文本保存。
                  </small>
                </div>
              )}
            </div>
            <div className="rule-section">
              <span>
                <ShieldCheck />
                安全边界
              </span>
              <div className="form-grid two">
                <label>
                  单次最长（分钟）
                  <input
                    type="number"
                    min="1"
                    max="120"
                    value={form.maxUse}
                    onChange={(e) =>
                      setForm({ ...form, maxUse: e.target.value })
                    }
                  />
                </label>
                <label>
                  每日上限（分钟）
                  <input
                    type="number"
                    min="1"
                    max="600"
                    value={form.maxDay}
                    onChange={(e) =>
                      setForm({ ...form, maxDay: e.target.value })
                    }
                  />
                </label>
              </div>
              <label className="check-line">
                <input
                  type="checkbox"
                  checked={form.ai}
                  onChange={(e) => setForm({ ...form, ai: e.target.checked })}
                />
                <span>
                  <b>启用 AI 微调</b>
                  <small>允许智能体在安全范围内微调执行时长</small>
                </span>
              </label>
            </div>
            <div className="editor-actions">
              {selected && (
                <>
                  <button
                    type="button"
                    className="danger-button"
                    onClick={() =>
                      deleteRule(selected).catch((reason) =>
                        setError(
                          reason instanceof Error
                            ? reason.message
                            : "规则删除失败",
                        ),
                      )
                    }
                  >
                    <Trash2 />
                    删除
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => testRule(selected)}
                  >
                    <TestTube2 />
                    测试规则
                  </button>
                </>
              )}
              <button className="primary-button">
                <Save />
                保存规则
              </button>
            </div>
          </form>
        </Card>
      </div>
    </>
  );
}

export function DocsPage() {
  const [tab, setTab] = useState("guide");
  const endpoints = [
    ["POST", "/api/auth/login", "用户登录并获取签名令牌"],
    ["POST", "/api/chat", "农业智能体对话"],
    ["GET", "/api/dashboard", "农场概览聚合数据"],
    ["GET / POST", "/api/fields", "地块查询与创建"],
    ["GET / POST", "/api/tasks", "农事任务管理"],
    ["GET / POST", "/api/progress", "种植进度管理"],
    ["GET / POST", "/api/finance/*", "财务记录与报表"],
    ["GET / POST", "/api/devices", "多协议设备管理"],
    ["GET / POST", "/api/rules", "自动化规则管理"],
  ];
  const connectionExamples = [
    {
      name: "MQTT",
      note: "适用于 ESP32、树莓派和消息型传感器；主题中的设备 ID 应与卡片一致。",
      config: {
        host: "192.168.1.10",
        port: 1883,
        control_topic: "devices/device_01/control",
        state_topic: "devices/device_01/state",
        qos: 0,
        use_tls: false,
      },
    },
    {
      name: "HTTP REST",
      note: "base_url 必须以 http:// 或 https:// 开头。",
      config: {
        base_url: "http://192.168.1.10:5000",
        api_key: "",
      },
    },
    {
      name: "Modbus TCP",
      note: "适用于局域网 PLC；slave_id 是设备站号。",
      config: {
        mode: "tcp",
        host: "192.168.1.20",
        port: 502,
        slave_id: 1,
        timeout: 2,
      },
    },
    {
      name: "Modbus RTU",
      note: "Windows 串口通常填写 COM3、COM4 等实际端口。",
      config: {
        mode: "rtu",
        port: "COM3",
        baudrate: 9600,
        slave_id: 1,
        timeout: 2,
      },
    },
    {
      name: "CoAP",
      note: "base_uri 必须以 coap:// 或 coaps:// 开头。",
      config: {
        base_uri: "coap://192.168.1.50:5683",
        command_path: "/command",
        state_path: "/state",
        auth_token: null,
      },
    },
    {
      name: "OPC UA",
      note: "仅映射允许读写的节点；未列入 command_nodes 的命令会被拒绝。",
      config: {
        endpoint: "opc.tcp://192.168.1.60:4840",
        username: "operator",
        password: "",
        command_nodes: {
          start: { node_id: "ns=2;s=Pump.Start", value: true },
          stop: { node_id: "ns=2;s=Pump.Start", value: false },
        },
        state_nodes: {
          status: "ns=2;s=Pump.Status",
          temperature: "ns=2;s=Sensor.Temperature",
        },
      },
    },
    {
      name: "USB 摄像头",
      note: "source 是本机摄像头编号，通常从 0 开始。",
      config: { camera_type: "usb", source: "0" },
    },
    {
      name: "IP / ESP32 摄像头",
      note: "IP 摄像头使用 RTSP/HTTP 流；ESP32-CAM 将 camera_type 改为 esp32cam。",
      config: {
        camera_type: "ip",
        source: "rtsp://192.168.1.30/stream",
        username: "admin",
        password: "",
      },
    },
  ];
  return (
    <>
      <PageHeader
        eyebrow="DOCUMENTATION"
        title="文档中心"
        description="了解产品工作方式、API 契约和硬件接入流程。"
      />
      <div className="docs-layout">
        <Card className="docs-nav">
          <button
            className={tab === "guide" ? "active" : ""}
            onClick={() => setTab("guide")}
          >
            <FileJson />
            使用指南
          </button>
          <button
            className={tab === "api" ? "active" : ""}
            onClick={() => setTab("api")}
          >
            <Code2 />
            API 接口
          </button>
          <button
            className={tab === "hardware" ? "active" : ""}
            onClick={() => setTab("hardware")}
          >
            <Cpu />
            硬件接入
          </button>
          <button
            className={tab === "architecture" ? "active" : ""}
            onClick={() => setTab("architecture")}
          >
            <Server />
            系统架构
          </button>
        </Card>
        <Card className="docs-content">
          {tab === "guide" && (
            <article>
              <span className="doc-kicker">GETTING STARTED</span>
              <h2>从登录到自动化的完整流程</h2>
              <p>
                新前端是独立的 React 应用，所有业务数据通过 FastAPI
                与后端交换。现有 Streamlit 前端不会受到影响。
              </p>
              <ol>
                <li>
                  <b>完善种植档案</b>
                  <span>
                    设置地区、土壤、面积和种植目标，让智能建议更准确。
                  </span>
                </li>
                <li>
                  <b>建立地块与计划</b>
                  <span>绘制地块边界，使用种植向导生成全周期进度和任务。</span>
                </li>
                <li>
                  <b>接入农场设备</b>
                  <span>注册模拟器或真实协议设备，并绑定所属地块。</span>
                </li>
                <li>
                  <b>设置自动规则</b>
                  <span>定义传感器条件、执行动作和每日安全边界。</span>
                </li>
              </ol>
              <div className="doc-callout">
                <ShieldCheck />
                <div>
                  <b>安全原则</b>
                  <p>所有硬件动作仍受自主权级别、规则边界和待确认队列约束。</p>
                </div>
              </div>
            </article>
          )}
          {tab === "api" && (
            <article>
              <span className="doc-kicker">HTTP API</span>
              <h2>前后端接口概览</h2>
              <p>
                业务请求默认携带 <code>username</code>{" "}
                查询参数；开启生产鉴权后还需发送 Bearer Token。
              </p>
              <div className="endpoint-list">
                {endpoints.map((endpoint, index) => (
                  <div key={index}>
                    <code>{endpoint[0]}</code>
                    <b>{endpoint[1]}</b>
                    <span>{endpoint[2]}</span>
                  </div>
                ))}
              </div>
            </article>
          )}
          {tab === "hardware" && (
            <article>
              <span className="doc-kicker">HARDWARE</span>
              <h2>支持的设备协议</h2>
              <div className="protocol-grid">
                {[
                  ["Simulator", "终端模拟器，用于无硬件开发与验证"],
                  ["MQTT", "传感器、继电器与边缘节点消息通信"],
                  ["HTTP REST", "通用网络设备状态和命令接口"],
                  ["Modbus", "RTU / TCP 工业设备"],
                  ["CoAP", "低功耗物联网设备"],
                  ["OPC UA", "工业节点读写与状态订阅"],
                  ["Camera", "USB、IP 与 ESP32-CAM 当前画面"],
                ].map((item) => (
                  <div key={item[0]}>
                    <span>
                      <Cpu />
                    </span>
                    <b>{item[0]}</b>
                    <p>{item[1]}</p>
                  </div>
                ))}
              </div>
              <section className="protocol-config-guide">
                <div className="protocol-config-head">
                  <span className="doc-kicker">CONNECTION CONFIG</span>
                  <h3>更换驱动协议与连接参数</h3>
                  <p>
                    在设备卡片中点击“配置”，选择新的驱动协议，然后将旧的连接参数
                    JSON 完整替换为对应格式。保存后设备会使用新配置重新连接；设备
                    ID 不会改变。
                  </p>
                </div>
                <div className="protocol-config-steps">
                  <span><b>1</b> 选择驱动协议</span>
                  <span><b>2</b> 替换连接参数 JSON</span>
                  <span><b>3</b> 核对设备能力</span>
                  <span><b>4</b> 保存并重新连接</span>
                </div>
                <div className="connection-example-grid">
                  {connectionExamples.map((example) => (
                    <article key={example.name}>
                      <div>
                        <b>{example.name}</b>
                        <p>{example.note}</p>
                      </div>
                      <pre><code>{JSON.stringify(example.config, null, 2)}</code></pre>
                    </article>
                  ))}
                </div>
                <div className="doc-callout warning">
                  <ShieldCheck />
                  <div>
                    <b>连接安全</b>
                    <p>
                      不要将 MQTT、Modbus、CoAP、OPC UA 或摄像头端口直接暴露到公网；
                      密码和证书仅填写在受控后端中，不要提交到 Git。
                    </p>
                  </div>
                </div>
              </section>
              <div className="doc-callout">
                <Settings2 />
                <div>
                  <b>模拟器保持终端模式</b>
                  <p>
                    硬件模拟器没有独立网页，设备中心通过真实协议通道读取同一份状态。
                  </p>
                </div>
              </div>
            </article>
          )}
          {tab === "architecture" && (
            <article>
              <span className="doc-kicker">ARCHITECTURE</span>
              <h2>清晰的部署边界</h2>
              <div className="architecture-flow">
                <div>
                  <b>React 前端</b>
                  <span>界面 · 交互 · 响应式布局</span>
                </div>
                <ChevronRight />
                <div>
                  <b>FastAPI</b>
                  <span>鉴权 · API · 调度</span>
                </div>
                <ChevronRight />
                <div>
                  <b>业务与数据</b>
                  <span>Agent · PostgreSQL · 设备协议</span>
                </div>
              </div>
              <p>
                部署时前端只需要配置公开的 HTTPS API 地址；数据库密钥、LLM
                密钥和设备连接参数始终留在后端。
              </p>
            </article>
          )}
        </Card>
      </div>
    </>
  );
}
