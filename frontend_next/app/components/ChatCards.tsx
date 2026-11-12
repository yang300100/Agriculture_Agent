"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CalendarDays,
  ChevronDown,
  Cpu,
  Gauge,
  Play,
  RefreshCw,
  ShieldCheck,
  Square,
} from "lucide-react";
import { get, post, put } from "../api";
import {
  canPrepareDevice,
  chatCardKey,
  controllableCapabilities,
  deviceObservation,
  runningTimeLabel,
  normalizeChatCards,
  pendingNumericDrafts,
  mergePendingParameters,
  prepareParams,
  resolvesUnknownExecution,
  taskStatusLabel,
  taskReminderDetails,
} from "../chat-cards";
import type { ChatCardReference, JsonMap } from "../types";
import { useVisiblePolling } from "../use-visible-polling";
import { StatusPill } from "./Common";

const capabilityNames: Record<string, string> = {
  irrigate: "灌溉",
  fertigate: "施肥",
  ventilate: "通风",
  heat: "加热",
  cool: "降温",
  shade: "遮阳",
  light: "补光",
  read_sensor: "传感器读数",
  capture: "摄像采集",
};
const valueNames: Record<string, string> = {
  soil_moisture: "土壤湿度",
  humidity: "空气湿度",
  temperature: "温度",
  soil_temperature: "土壤温度",
  ph: "酸碱度",
  flow_rate: "流量",
  pressure: "压力",
  light_lux: "光照",
  co2_ppm: "二氧化碳",
  duration: "运行时长（分钟）",
  duration_minutes: "运行时长（分钟）",
  amount_kg: "施肥量（kg）",
  target_temp: "目标温度",
  brightness_percent: "亮度",
  power: "电源",
  volume_liters: "目标水量（升）",
  target_soil_moisture: "目标土壤湿度（%）",
  concentration_percent: "肥液浓度（%）",
  speed_percent: "风机速度（%）",
  target_lux: "目标光照（lux）",
  power_percent: "功率（%）",
  position_percent: "遮阳位置（%）",
};
const actionNames: Record<string, string> = {
  start: "启动",
  stop: "停止",
  power_on: "开机",
  power_off: "关机",
};
const pendingNames: Record<string, string> = {
  pending: "待确认",
  failed: "执行失败",
  uncertain: "结果不确定",
  executing: "执行中",
  executed: "已执行",
  reconciled: "已核对",
  rejected: "已拒绝",
  cancelled: "已取消",
  expired: "已过期",
};

function timestamp(value: unknown) {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value)))
    return "时间未知";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function shownValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "未知";
  if (typeof value === "boolean") return value ? "开启" : "关闭";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function ParameterList({ values }: { values: JsonMap }) {
  const entries = Object.entries(values);
  return entries.length ? (
    <dl className="chat-card-parameters">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{valueNames[key] || key}</dt>
          <dd>{shownValue(value)}</dd>
        </div>
      ))}
    </dl>
  ) : (
    <p className="chat-card-muted">无附加参数</p>
  );
}

function useCardVisibility() {
  const element = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (!element.current) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(([entry]) =>
      setVisible(entry.isIntersecting),
    );
    observer.observe(element.current);
    return () => observer.disconnect();
  }, []);
  return { element, visible };
}

function useCardResource(path: string, visible: boolean) {
  const [data, setData] = useState<JsonMap | null>(null);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [updatedAt, setUpdatedAt] = useState("");
  const mounted = useRef(false);
  const currentPath = useRef(path);
  currentPath.current = path;
  const read = useRef<Promise<void> | null>(null);
  const version = useRef(0);
  const refresh = useCallback(
    async (force = false) => {
      if (currentPath.current !== path) return;
      if (read.current) {
        await read.current;
        if (!force) return;
      }
      if (!mounted.current || currentPath.current !== path) return;
      const requestVersion = ++version.current;
      setRefreshing(true);
      const request = (async () => {
        try {
          const result = await get<JsonMap>(path);
          if (
            mounted.current &&
            currentPath.current === path &&
            requestVersion === version.current
          ) {
            setData(result);
            setError("");
            setUpdatedAt(new Date().toISOString());
          }
        } catch (reason) {
          if (
            mounted.current &&
            currentPath.current === path &&
            requestVersion === version.current
          )
            setError(reason instanceof Error ? reason.message : "卡片读取失败");
        } finally {
          if (
            mounted.current &&
            currentPath.current === path &&
            requestVersion === version.current
          )
            setRefreshing(false);
        }
      })();
      read.current = request;
      await request;
      if (read.current === request) read.current = null;
    },
    [path],
  );
  useEffect(() => {
    mounted.current = true;
    setData(null);
    setError("");
    setUpdatedAt("");
    return () => {
      mounted.current = false;
      version.current += 1;
    };
  }, [path]);
  useEffect(() => {
    if (visible) void refresh(true);
  }, [visible, refresh]);
  return { data, error, refreshing, refresh, mounted, updatedAt };
}

type DeviceCardReference = Extract<ChatCardReference, { type: "device" }>;
type CardCallbacks = {
  onReferenceChange?: (reference: ChatCardReference) => void;
  onPendingChange?: () => void;
  disabled?: boolean;
};

function DeviceChatCard({
  reference,
  onReferenceChange,
  onPendingChange,
  disabled = false,
}: CardCallbacks & { reference: DeviceCardReference }) {
  const { element, visible } = useCardVisibility();
  const query = reference.pending_action_id
    ? `?pending_action_id=${encodeURIComponent(reference.pending_action_id)}`
    : "";
  const {
    data,
    error: readError,
    refreshing,
    refresh,
    mounted,
  } = useCardResource(
    `/api/chat/cards/device/${encodeURIComponent(reference.device_id)}${query}`,
    visible,
  );
  const [duration, setDuration] = useState("10");
  const [selectedCapability, setSelectedCapability] = useState("");
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(false);
  const [rawParams, setRawParams] = useState("");
  const [numericDrafts, setNumericDrafts] = useState<Record<string, string>>(
    {},
  );
  const [unknownExecution, setUnknownExecution] = useState(false);
  const unknownActionId = useRef("");
  const blockedActionIds = useRef(new Set<string>());
  const submitting = useRef(false);
  const prepareRequest = useRef<{ signature: string; id: string } | null>(null);
  const pending = data?.pending_action as JsonMap | undefined;
  const uncertain = unknownExecution || pending?.status === "uncertain";
  const actionable =
    !!pending &&
    ["pending", "failed"].includes(String(pending.status)) &&
    !uncertain &&
    !blockedActionIds.current.has(String(pending.id));
  const unresolved =
    !!pending &&
    ["pending", "failed", "uncertain", "executing"].includes(
      String(pending.status),
    );
  const observation = deviceObservation(data?.state || {});
  const runningTime = readError ? null : runningTimeLabel(data?.state || {});
  const controllable = canPrepareDevice(data?.capabilities || []);
  const capabilityOptions = controllableCapabilities(data?.capabilities || []);
  useEffect(() => {
    if (
      ["executed", "reconciled", "rejected", "cancelled", "expired"].includes(
        String(pending?.status),
      )
    )
      prepareRequest.current = null;
    if (
      unknownExecution &&
      resolvesUnknownExecution(
        unknownActionId.current,
        pending,
        observation.fresh && !readError,
      )
    ) {
      setUnknownExecution(false);
      prepareRequest.current = null;
    }
  }, [unknownExecution, pending, observation.fresh, readError]);
  useVisiblePolling(() => refresh(), 5000, visible && !busy && !editing);

  async function run(kind: string, operation: () => Promise<void>) {
    if (submitting.current || disabled) return;
    submitting.current = true;
    setBusy(kind);
    setError("");
    setMessage("");
    try {
      await operation();
    } catch (reason) {
      if (mounted.current)
        setError(reason instanceof Error ? reason.message : "设备操作处理失败");
    } finally {
      if (mounted.current) {
        await refresh(true);
        if (mounted.current) {
          setBusy("");
          onPendingChange?.();
        }
      }
      submitting.current = false;
    }
  }

  function prepare(command: "start" | "stop") {
    void run(command, async () => {
      if (!data || !controllable || unresolved || uncertain || readError)
        return;
      const params = prepareParams(
        data.capabilities || [],
        selectedCapability,
        command,
        duration,
      );
      const signature = JSON.stringify({ command, params });
      if (prepareRequest.current?.signature !== signature) {
        prepareRequest.current = {
          signature,
          id:
            globalThis.crypto?.randomUUID?.() ||
            `card-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        };
      }
      const result = await post<JsonMap>(
        `/api/chat/cards/device/${encodeURIComponent(reference.device_id)}/prepare`,
        {
          command,
          params,
          request_id: prepareRequest.current.id,
        },
      );
      if (!mounted.current) return;
      if (!result.success || !result.pending_action?.id)
        throw new Error(result.message || result.error || "未能准备设备操作");
      onReferenceChange?.({
        type: "device",
        device_id: reference.device_id,
        pending_action_id: result.pending_action.id,
      });
      setMessage("操作已准备，请核对参数后确认执行。");
    });
  }

  function decide(accept: boolean) {
    void run(accept ? "confirm" : "reject", async () => {
      if (!pending?.id || !actionable || readError) return;
      let result: JsonMap;
      try {
        result = await post<JsonMap>(
          `/api/actions/${encodeURIComponent(pending.id)}/${accept ? "confirm" : "reject"}`,
        );
      } catch (reason) {
        if (accept && mounted.current) {
          unknownActionId.current = pending.id;
          blockedActionIds.current.add(String(pending.id));
          setUnknownExecution(true);
        }
        throw reason;
      }
      if (!mounted.current) return;
      if (
        result.action_status === "uncertain" ||
        result.result?.status === "uncertain"
      ) {
        unknownActionId.current = pending.id;
        blockedActionIds.current.add(String(pending.id));
        setUnknownExecution(true);
      }
      if (!result.success)
        throw new Error(
          result.message ||
            result.error ||
            result.result?.error ||
            "设备操作未完成",
        );
      prepareRequest.current = null;
      setMessage(
        result.result?.message ||
          result.message ||
          (accept ? "操作已确认，请查看最新设备状态。" : "已拒绝该操作。"),
      );
    });
  }

  function saveParams() {
    void run("edit", async () => {
      if (!pending?.id || !actionable || readError) return;
      const params = mergePendingParameters(
        pending.params || {},
        numericDrafts,
        rawParams,
      );
      const result = await put<JsonMap>(
        `/api/actions/${encodeURIComponent(pending.id)}`,
        { params },
      );
      if (result.success === false)
        throw new Error(result.message || result.error || "参数保存失败");
      if (mounted.current) {
        setEditing(false);
        setMessage("参数已保存，请重新核对后确认。");
      }
    });
  }

  function reconcile() {
    void run("reconcile", async () => {
      const result = await post<JsonMap>(
        `/api/devices/${encodeURIComponent(reference.device_id)}/reconcile`,
      );
      if (mounted.current) {
        if (
          resolvesUnknownExecution(
            unknownActionId.current,
            pending,
            false,
            result,
          )
        ) {
          setUnknownExecution(false);
          prepareRequest.current = null;
        }
        setMessage(
          result.message || "已读取设备当前状态；不确定操作不会再次下发。",
        );
      }
    });
  }

  const locked = disabled || !!busy;
  const stateEntries = Object.fromEntries(
    Object.entries(data?.state || {}).filter(
      ([key, value]) =>
        !key.startsWith("_") &&
        ![
          "status",
          "error",
          "run_history",
          "run_until",
          "run_started_at",
        ].includes(key) &&
        ["string", "number", "boolean"].includes(typeof value),
    ),
  );
  return (
    <article
      ref={element}
      className="chat-business-card chat-device-card"
      aria-label={`设备卡：${data?.name || reference.device_id}`}
    >
      <header>
        <span className="chat-business-icon">
          {controllable ? <Cpu /> : <Gauge />}
        </span>
        <div>
          <h3>{data?.name || reference.device_id}</h3>
          <p>
            {data?.plot_name || "未关联地块"}
            <span className="chat-card-device-id">{reference.device_id}</span>
          </p>
        </div>
        <button
          type="button"
          className="chat-card-refresh"
          onClick={() => void refresh()}
          disabled={refreshing || !!busy}
          aria-label="刷新设备卡"
        >
          <RefreshCw className={refreshing ? "spin" : ""} />
        </button>
      </header>
      {data ? (
        <>
          <div className="chat-card-statuses">
            <StatusPill
              tone={observation.fresh && !readError ? "success" : "warning"}
            >
              {readError ? "在线状态未知" : observation.connection}
            </StatusPill>
            <StatusPill tone="neutral">
              {readError ? "运行状态未知" : observation.operation}
            </StatusPill>
          </div>
          <p className="chat-card-observed">
            观测时间：{timestamp(observation.observedAt)}
          </p>
          {runningTime && <p className="chat-card-observed">{runningTime}</p>}
          {(!observation.fresh || readError) && (
            <p className="chat-card-warning">
              当前状态无法核实，请刷新或只读核对。历史读数仅供参考。
            </p>
          )}
          <details className="chat-card-details">
            <summary>
              设备详情
              <ChevronDown />
            </summary>
            <p>
              能力：
              {(data.capabilities || [])
                .map((item: string) => capabilityNames[item] || item)
                .join("、") || "未知"}
            </p>
            <ParameterList values={stateEntries} />
          </details>
          {pending && (
            <section
              className={`chat-card-pending ${uncertain ? "uncertain" : ""}`}
              aria-label="设备操作确认"
            >
              <div className="chat-card-pending-title">
                <ShieldCheck />
                <b>
                  {uncertain
                    ? "结果不确定"
                    : pendingNames[pending.status] || "操作状态未知"}
                </b>
                <span>{actionNames[pending.command] || pending.command}</span>
              </div>
              <p>
                {pending.description ||
                  pending.reason ||
                  "请核对设备、操作与参数。"}
              </p>
              <ParameterList values={pending.params || {}} />
              {pending.created_at && (
                <p className="chat-card-muted">
                  准备时间：{timestamp(pending.created_at)}
                </p>
              )}
              {pending.last_error && (
                <p className="chat-card-error">{String(pending.last_error)}</p>
              )}
              {actionable && (
                <div className="chat-card-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={locked || !!readError}
                    onClick={() => decide(false)}
                  >
                    拒绝操作
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={locked || !!readError}
                    onClick={() => {
                      const drafts = pendingNumericDrafts(pending.params || {});
                      setNumericDrafts(drafts);
                      setRawParams(
                        JSON.stringify(
                          Object.fromEntries(
                            Object.entries(pending.params || {}).filter(
                              ([key]) =>
                                !Object.hasOwn(drafts, key) &&
                                !["action", "capability"].includes(key),
                            ),
                          ),
                          null,
                          2,
                        ),
                      );
                      setEditing(!editing);
                    }}
                  >
                    修改参数
                  </button>
                  <button
                    type="button"
                    className="primary-button"
                    disabled={locked || editing || !!readError}
                    onClick={() => decide(true)}
                  >
                    {busy === "confirm"
                      ? "正在确认"
                      : pending.status === "failed"
                        ? "重新确认执行"
                        : "确认执行"}
                  </button>
                </div>
              )}
              {editing && actionable && (
                <div className="chat-card-param-editor">
                  {Object.entries(numericDrafts).map(([key, value]) => (
                    <label key={key}>
                      {valueNames[key] || key}
                      <input
                        type="number"
                        step="any"
                        min={
                          ["duration", "duration_minutes"].includes(key)
                            ? 1
                            : undefined
                        }
                        value={value}
                        onChange={(event) =>
                          setNumericDrafts((current) => ({
                            ...current,
                            [key]: event.target.value,
                          }))
                        }
                        disabled={locked}
                      />
                    </label>
                  ))}
                  <details className="chat-card-details">
                    <summary>
                      高级参数
                      <ChevronDown />
                    </summary>
                    <label>
                      其他操作参数（JSON 对象）
                      <textarea
                        value={rawParams}
                        onChange={(event) => setRawParams(event.target.value)}
                        rows={5}
                        disabled={locked}
                      />
                    </label>
                    <p>已有参数与操作能力会保留，常用数值以上方输入为准。</p>
                  </details>
                  <p>保存不会执行；确认时会重新检查设备限制。</p>
                  <div className="chat-card-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => setEditing(false)}
                      disabled={locked}
                    >
                      取消修改
                    </button>
                    <button
                      type="button"
                      className="primary-button"
                      onClick={saveParams}
                      disabled={locked}
                    >
                      保存参数
                    </button>
                  </div>
                </div>
              )}
            </section>
          )}
          {uncertain && (
            <div className="chat-card-uncertain">
              <p>
                执行结果不确定，禁止重复下发。请只读核对设备状态，必要时到现场检查。
              </p>
              <button
                type="button"
                className="secondary-button"
                disabled={locked}
                onClick={reconcile}
              >
                只读核对
              </button>
            </div>
          )}
          {controllable && !unresolved && !uncertain && (
            <div className="chat-card-prepare">
              {capabilityOptions.length > 1 && (
                <label>
                  操作能力
                  <select
                    value={selectedCapability}
                    onChange={(event) =>
                      setSelectedCapability(event.target.value)
                    }
                    disabled={locked}
                  >
                    <option value="">请选择操作能力</option>
                    {capabilityOptions.map((capability) => (
                      <option key={capability} value={capability}>
                        {capabilityNames[capability] || capability}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label>
                启动时长（分钟）
                <input
                  type="number"
                  min="1"
                  step="any"
                  value={duration}
                  onChange={(event) => setDuration(event.target.value)}
                  disabled={locked}
                />
              </label>
              <div className="chat-card-actions">
                <button
                  type="button"
                  className="primary-button"
                  disabled={locked || !!readError}
                  onClick={() => prepare("start")}
                >
                  <Play />
                  {busy === "start" ? "正在准备" : "准备启动"}
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={locked || !!readError}
                  onClick={() => prepare("stop")}
                >
                  <Square />
                  {busy === "stop" ? "正在准备" : "准备停止"}
                </button>
              </div>
              <p>准备后还需确认，才会执行。</p>
            </div>
          )}
          {!controllable && (
            <p className="chat-card-muted">此设备仅支持查看数据。</p>
          )}
        </>
      ) : (
        !readError && <p className="chat-card-muted">正在读取设备最新信息…</p>
      )}
      {readError && (
        <p className="chat-card-error" role="alert">
          设备卡读取失败：{readError}
        </p>
      )}
      {error && (
        <p className="chat-card-error" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="chat-card-feedback" role="status">
          {message}
        </p>
      )}
    </article>
  );
}

function TaskChatCard({
  reference,
}: {
  reference: Extract<ChatCardReference, { type: "task" }>;
}) {
  const { element, visible } = useCardVisibility();
  const { data, error, refreshing, refresh, updatedAt } = useCardResource(
    `/api/chat/cards/task/${encodeURIComponent(reference.task_id)}`,
    visible,
  );
  const reminder = data ? taskReminderDetails(data) : null;
  return (
    <article
      ref={element}
      className="chat-business-card chat-task-card"
      aria-label={`预约任务卡：${data?.title || reference.task_id}`}
    >
      <header>
        <span className="chat-business-icon">
          <CalendarDays />
        </span>
        <div>
          <h3>{data?.title || "预约任务"}</h3>
          <p>{data?.crop || "作物未指定"}</p>
        </div>
        <button
          type="button"
          className="chat-card-refresh"
          onClick={() => void refresh()}
          disabled={refreshing}
          aria-label="刷新预约任务卡"
        >
          <RefreshCw className={refreshing ? "spin" : ""} />
        </button>
      </header>
      {data ? (
        <>
          <StatusPill
            tone={
              !error && ["completed", "已完成"].includes(data.status)
                ? "success"
                : "neutral"
            }
          >
            {error ? "状态未知" : taskStatusLabel(data.status)}
          </StatusPill>
          <dl className="chat-task-times">
            {reminder && (
              <>
                <div>
                  <dt>提醒时间</dt>
                  <dd>
                    {new Date(reminder.time).toLocaleString("zh-CN", {
                      year: "numeric",
                      month: "2-digit",
                      day: "2-digit",
                      hour: "2-digit",
                      minute: "2-digit",
                      hour12: false,
                    })}
                  </dd>
                </div>
                <div>
                  <dt>提醒频率</dt>
                  <dd>{reminder.frequency}</dd>
                </div>
              </>
            )}
            <div>
              <dt>开始</dt>
              <dd>{data.start_date || "未安排"}</dd>
            </div>
            <div>
              <dt>结束</dt>
              <dd>{data.end_date || "未安排"}</dd>
            </div>
          </dl>
          {data.description && <p>{String(data.description)}</p>}
          {data.device_id && (
            <div className="chat-task-device">
              <b>{data.device_name || data.device_id}</b>
              <p>
                预约操作：
                {actionNames[data.device_command] ||
                  data.device_command ||
                  "未指定"}
              </p>
              <ParameterList values={data.device_params || {}} />
            </div>
          )}
          <p className="chat-card-muted">读取时间：{timestamp(updatedAt)}</p>
          <p className="chat-card-muted">预约将按任务安排与安全策略处理。</p>
        </>
      ) : (
        !error && <p className="chat-card-muted">正在读取预约任务…</p>
      )}
      {error && (
        <p className="chat-card-error" role="alert">
          任务卡读取失败：{error}
        </p>
      )}
    </article>
  );
}

export function ChatCards({
  cards,
  onReferenceChange,
  onPendingChange,
  disabled,
}: {
  cards: unknown;
  onReferenceChange: (index: number, reference: ChatCardReference) => void;
  onPendingChange: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="chat-business-cards">
      {normalizeChatCards(cards).map((reference, index) =>
        reference.type === "device" ? (
          <DeviceChatCard
            key={`${index}:device:${reference.device_id}`}
            reference={reference}
            disabled={disabled}
            onReferenceChange={(next) => onReferenceChange(index, next)}
            onPendingChange={onPendingChange}
          />
        ) : (
          <TaskChatCard key={chatCardKey(reference)} reference={reference} />
        ),
      )}
    </div>
  );
}
