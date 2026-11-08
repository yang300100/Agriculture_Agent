"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  CalendarClock,
  Check,
  CircleDollarSign,
  CloudSun,
  Cpu,
  Plus,
  Sprout,
  TriangleAlert,
} from "lucide-react";
import { get, post } from "../api";
import type { Device, JsonMap, Progress, Task } from "../types";
import { summarizeWeather } from "../weather-state";
import { isTaskRunning, overallProgress } from "../task-view";
import { useVisiblePolling } from "../use-visible-polling";
import {
  Card,
  Empty,
  ErrorState,
  Loading,
  MiniBars,
  PageHeader,
  StatusPill,
} from "./Common";

export function DashboardPage({
  onNavigate,
}: {
  onNavigate: (page: string) => void;
}) {
  const [data, setData] = useState<JsonMap | null>(null);
  const [financeReport, setFinanceReport] = useState<JsonMap | null>(null);
  const [progress, setProgress] = useState<Progress[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [devices, setDevices] = useState<Device[]>([]);
  const [deviceLogs, setDeviceLogs] = useState<JsonMap[]>([]);
  const [pendingDeviceActions, setPendingDeviceActions] = useState<JsonMap[]>(
    [],
  );
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");

  const load = useCallback(async () => {
    setError("");
    try {
      const [
        dashboard,
        progressRows,
        taskRows,
        financeSummary,
        deviceRows,
        actionRows,
        pendingRows,
      ] = await Promise.all([
        get("/api/dashboard"),
        get<Progress[]>("/api/progress"),
        get<Task[]>("/api/tasks"),
        get("/api/finance/summary"),
        get<Device[]>("/api/devices").catch(() => []),
        get<JsonMap[]>("/api/actions/log?limit=12").catch(() => []),
        get<JsonMap[]>("/api/actions/pending").catch(() => []),
      ]);
      setData(dashboard);
      setProgress(progressRows);
      setTasks(taskRows);
      setFinanceReport(financeSummary);
      setDevices(deviceRows);
      setDeviceLogs(actionRows);
      setPendingDeviceActions(
        pendingRows.filter((item) => item.status === "pending"),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "农场数据加载失败");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);
  useVisiblePolling(load, 10000, !busy);

  async function completeTask(id: string) {
    if (busy) return;
    const task = tasks.find((item) => item.id === id);
    if (!task || isTaskRunning(task)) return;
    if (
      task.device_id &&
      !window.confirm(
        `确认“${task.title}”已在现场完成吗？此操作仅登记完成，不会启动设备。`,
      )
    )
      return;
    setBusy(id);
    setError("");
    try {
      const result = await post<JsonMap>(`/api/tasks/${id}/complete`);
      if (result.success === false)
        throw new Error(result.error || "任务更新失败");
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务更新失败");
    } finally {
      setBusy("");
    }
  }
  async function advance(id: string) {
    setBusy(id);
    setError("");
    try {
      const result = await post<JsonMap>(`/api/progress/${id}/advance`);
      if (result.success === false) {
        throw new Error(result.message || "种植阶段更新失败");
      }
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "种植阶段更新失败");
    } finally {
      setBusy("");
    }
  }

  if (error)
    return (
      <>
        <PageHeader
          eyebrow="TODAY'S FARM"
          title="农场概览"
          description="今天的关键进度、风险和经营情况。"
        />
        <ErrorState message={error} retry={load} />
      </>
    );
  if (!data)
    return (
      <>
        <PageHeader
          eyebrow="TODAY'S FARM"
          title="农场概览"
          description="今天的关键进度、风险和经营情况。"
        />
        <Loading />
      </>
    );

  const finance = data.finance || {};
  const weatherSummary = summarizeWeather(data.weather_alerts);
  const incomeTrend = Object.entries(financeReport?.monthly_data || {})
    .filter(([, value]) => Number((value as JsonMap).income || 0) > 0)
    .sort(([left], [right]) => left.localeCompare(right))
    .slice(-6);
  const activeTasks = tasks.filter((task) => task.status !== "已完成");
  const alerts = [
    ...(data.weather_persistence?.alerts || []),
    ...(data.disease_risks || []),
  ];
  const onlineDevices = devices.filter(
    (device) => !["offline", "error", "disconnected"].includes(device.status),
  );
  const abnormalDevices = devices.filter((device) => {
    const connectionStatus = String(device.status || "unknown").toLowerCase();
    const runtimeStatus = String(device.state?.status || "").toLowerCase();
    return (
      ["offline", "error", "disconnected"].includes(connectionStatus) ||
      ["error", "fault", "failed"].includes(runtimeStatus)
    );
  });
  const runningDevices = devices.filter((device) => {
    const runtimeStatus = String(
      device.state?.status || device.status || "",
    ).toLowerCase();
    return ["running", "active", "busy", "working"].includes(runtimeStatus);
  });
  const deviceWithCurrentAction = devices.find((device) =>
    Boolean(
      device.state?.current_action ||
      device.state?.action ||
      device.state?.command,
    ),
  );
  const latestDeviceAction = [...deviceLogs].sort((left, right) =>
    String(right.timestamp || "").localeCompare(String(left.timestamp || "")),
  )[0];
  const pendingDeviceAction = pendingDeviceActions[0];
  const actionDevice = deviceWithCurrentAction || runningDevices[0];
  const actionValue = actionDevice
    ? actionDevice.state?.current_action ||
      actionDevice.state?.action ||
      actionDevice.state?.command ||
      "running"
    : latestDeviceAction?.command ||
      latestDeviceAction?.action ||
      pendingDeviceAction?.command ||
      pendingDeviceAction?.action;
  const actionDeviceName =
    actionDevice?.name ||
    devices.find(
      (device) =>
        device.device_id ===
        (latestDeviceAction?.device_id || pendingDeviceAction?.device_id),
    )?.name ||
    latestDeviceAction?.device_id ||
    pendingDeviceAction?.device_id;
  const actionHeading = actionDevice
    ? "当前动作"
    : latestDeviceAction
      ? "最近动作"
      : pendingDeviceAction
        ? "待确认动作"
        : "动作记录";
  const actionSummary = actionValue
    ? `${actionDeviceName || "设备"} · ${deviceActionLabel(String(actionValue))}`
    : "暂无设备动作记录";
  const now = new Date();
  const dateLabel = new Intl.DateTimeFormat("zh-CN", {
    month: "long",
    day: "numeric",
    weekday: "long",
  }).format(now);
  const hour = now.getHours();
  const greeting =
    hour < 6
      ? "夜深了"
      : hour < 12
        ? "早上好"
        : hour < 14
          ? "中午好"
          : hour < 18
            ? "下午好"
            : "晚上好";

  return (
    <>
      <PageHeader
        eyebrow={dateLabel}
        title={`${greeting}，看看今天的农场`}
        description="这是今天最值得关注的农场动态。"
        actions={
          <>
            <button
              className="secondary-button"
              onClick={() => onNavigate("calendar")}
            >
              <CalendarClock />
              查看日历
            </button>
            <button
              className="primary-button"
              onClick={() => onNavigate("wizard")}
            >
              <Plus />
              新建计划
            </button>
          </>
        }
      />
      <div className="hero-grid">
        <Card className="farm-hero">
          <div className="hero-finance-head">
            <span>本季经营概况</span>
            <small>FINANCE OVERVIEW</small>
          </div>
          <div className="hero-finance-body">
            <div className="hero-profit">
              <span>本月净利润</span>
              <strong>
                <small>¥</small>
                {Number(finance.profit || 0).toLocaleString()}
              </strong>
            </div>
            <div className="hero-breakdown">
              <div>
                <small>本月收入</small>
                <b>¥{Number(finance.month_income || 0).toLocaleString()}</b>
              </div>
              <div>
                <small>本月成本</small>
                <b>¥{Number(finance.month_cost || 0).toLocaleString()}</b>
              </div>
            </div>
          </div>
          <div className="hero-finance-footer">
            <div className="hero-weather">
              <span>
                <CloudSun />
              </span>
              <div>
                <b>{data.weather_alerts?.region || "我的农场"}</b>
                <small>{weatherSummary.label}</small>
              </div>
            </div>
            <button
              className="hero-finance-button"
              onClick={() => onNavigate("finance")}
            >
              查看财务详情 <ArrowRight />
            </button>
          </div>
          <div className="hero-orbit" aria-hidden="true">
            <CircleDollarSign />
            <i />
          </div>
        </Card>
        <div className="metric-stack">
          <Card className="metric-card">
            <span className="metric-icon green">
              <Sprout />
            </span>
            <div className="metric-card-content">
              <div className="metric-card-heading">
                <small>进行中计划</small>
                <span>
                  覆盖 {new Set(progress.map((item) => item.crop)).size} 种作物
                </span>
              </div>
              <strong>
                {progress.filter((item) => item.status !== "已完成").length}
              </strong>
            </div>
          </Card>
          <Card className="metric-card">
            <span className="metric-icon amber">
              <CalendarClock />
            </span>
            <div className="metric-card-content">
              <div className="metric-card-heading">
                <small>待办农事</small>
                <span>
                  {
                    activeTasks.filter((item) => item.priority === "high")
                      .length
                  }{" "}
                  项高优先级
                </span>
              </div>
              <strong>{activeTasks.length}</strong>
            </div>
          </Card>
        </div>
      </div>
      <div className="dashboard-workspace-grid">
        <div className="dashboard-workspace-column dashboard-primary-column">
          <div className="dashboard-status-strip">
            <Card
              className="device-overview-card"
              title="硬件设备状态"
              action={
                <button
                  type="button"
                  className="text-button"
                  onClick={() => onNavigate("devices")}
                >
                  设备中心 <ArrowRight />
                </button>
              }
            >
              <div
                className={`device-health-banner ${
                  abnormalDevices.length ? "warning" : "healthy"
                }`}
              >
                <span>
                  <Cpu />
                </span>
                <div>
                  <b>
                    {!devices.length
                      ? "尚未注册硬件设备"
                      : abnormalDevices.length
                        ? `${abnormalDevices.length} 台设备需要检查`
                        : "当前设备工作正常"}
                  </b>
                  <small>
                    {devices.length
                      ? `${onlineDevices.length} / ${devices.length} 台设备已连接`
                      : "注册设备后将在这里显示实时状态"}
                  </small>
                </div>
                <i>
                  {!devices.length
                    ? "未接入"
                    : abnormalDevices.length
                      ? "需关注"
                      : "状态良好"}
                </i>
              </div>
              <div className="device-overview-metrics">
                <div>
                  <small>在线设备</small>
                  <strong>{onlineDevices.length}</strong>
                </div>
                <div>
                  <small>运行中</small>
                  <strong>{runningDevices.length}</strong>
                </div>
                <div className={abnormalDevices.length ? "warning" : ""}>
                  <small>异常 / 离线</small>
                  <strong>{abnormalDevices.length}</strong>
                </div>
              </div>
              <div className="device-action-summary">
                <span>
                  <Activity />
                </span>
                <div>
                  <small>{actionHeading}</small>
                  <b>{actionSummary}</b>
                  <em>
                    {pendingDeviceActions.length
                      ? `另有 ${pendingDeviceActions.length} 个动作待确认`
                      : latestDeviceAction?.timestamp
                        ? `记录于 ${String(latestDeviceAction.timestamp).slice(0, 16).replace("T", " ")}`
                        : "暂无待确认动作"}
                  </em>
                </div>
              </div>
            </Card>
          </div>
          <Card
            className="dashboard-progress-card"
            title="种植进度"
            action={
              <button
                className="text-button"
                onClick={() => onNavigate("calendar")}
              >
                全部计划 <ArrowRight />
              </button>
            }
          >
            {progress.length ? (
              <div className="progress-list">
                {progress.slice(0, 5).map((item) => (
                  <div className="progress-row" key={item.id}>
                    <div className="crop-avatar">{item.crop.slice(0, 1)}</div>
                    <div className="progress-main">
                      <div>
                        <b>{item.crop}</b>
                        <span>{item.stage || "准备期"}</span>
                        <em title="整季预计进度">{overallProgress(item)}%</em>
                      </div>
                      <div className="progress-track">
                        <i
                          style={{
                            width: `${overallProgress(item)}%`,
                          }}
                        />
                      </div>
                    </div>
                    {item.status !== "已完成" && item.status !== "待开始" && (
                      <button
                        className="small-button"
                        disabled={busy === item.id}
                        onClick={() => advance(item.id)}
                      >
                        <Check />
                        完成阶段
                      </button>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <Empty
                title="还没有种植计划"
                body="从种植向导生成第一份完整计划。"
              />
            )}
          </Card>
        </div>
        <div className="dashboard-workspace-column dashboard-secondary-column">
          <Card className="risk-summary-card">
            <span
              className={
                alerts.length || !weatherSummary.available ? "warning" : "clear"
              }
            >
              <TriangleAlert />
            </span>
            <div>
              <b>风险提醒</b>
              <small
                className={
                  alerts.length || !weatherSummary.available
                    ? "warning-text"
                    : "positive"
                }
              >
                {alerts.length
                  ? "建议及时查看"
                  : weatherSummary.available
                    ? "暂无已记录风险"
                    : "天气数据缺失"}
              </small>
            </div>
            <strong>
              {alerts.length || (weatherSummary.available ? 0 : "—")}
            </strong>
          </Card>
          <Card
            className="dashboard-task-card"
            title="近期任务"
            action={
              <button
                className="text-button"
                onClick={() => onNavigate("calendar")}
              >
                农事日历
              </button>
            }
          >
            {activeTasks.length ? (
              <div className="task-list">
                {activeTasks.slice(0, 6).map((task) => (
                  <div className="task-row" key={task.id}>
                    <button
                      className="task-check"
                      aria-label={task.device_id ? "登记现场完成" : "标记完成"}
                      title={
                        isTaskRunning(task)
                          ? "设备运行结束后自动更新"
                          : task.device_id
                            ? "登记现场完成"
                            : "标记完成"
                      }
                      onClick={() => completeTask(task.id)}
                      disabled={!!busy || isTaskRunning(task)}
                    >
                      <Check />
                    </button>
                    <div>
                      <b>{task.title}</b>
                      <span>
                        {task.crop || "通用任务"} ·{" "}
                        {task.end_date?.slice(0, 10) || "未设截止时间"}
                      </span>
                    </div>
                    <StatusPill
                      tone={
                        task.priority === "high"
                          ? "danger"
                          : task.priority === "low"
                            ? "success"
                            : "warning"
                      }
                    >
                      {task.priority === "high"
                        ? "高"
                        : task.priority === "low"
                          ? "低"
                          : "中"}
                    </StatusPill>
                  </div>
                ))}
              </div>
            ) : (
              <Empty
                title="今天没有待办"
                body="可以安心巡视田间，或创建新的农事任务。"
              />
            )}
          </Card>
        </div>
      </div>
      <div className="content-grid equal">
        <Card title="经营趋势">
          {incomeTrend.length ? (
            <>
              <MiniBars
                values={incomeTrend.map(([, value]) =>
                  Number((value as JsonMap).income || 0),
                )}
                labels={incomeTrend.map(
                  ([month]) => `${Number(month.slice(5, 7))}月`,
                )}
              />
              <div className="chart-legend">
                <span>
                  <i className="green-dot" />
                  真实收入走势
                </span>
                <b>
                  本月净利润 ¥{Number(finance.profit || 0).toLocaleString()}
                </b>
              </div>
            </>
          ) : (
            <Empty
              title="暂无经营趋势数据"
              body="在财务管理中记录收入后，这里将展示真实的月度收入趋势。"
            />
          )}
        </Card>
        <Card title="风险与建议">
          {alerts.length ? (
            <div className="alert-list">
              {alerts.slice(0, 4).map((alert: JsonMap, index: number) => (
                <div key={index}>
                  <span>
                    <TriangleAlert />
                  </span>
                  <div>
                    <b>
                      {alert.type ||
                        `${alert.crop || "作物"} ${alert.disease || "风险"}`}
                    </b>
                    <p>{alert.advice || alert.desc || "请关注后续变化"}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="all-clear">
              <span>
                <Check />
              </span>
              <div>
                <b>
                  {weatherSummary.available
                    ? "暂无已记录的风险提示"
                    : "天气数据暂不可用"}
                </b>
                <p>
                  {weatherSummary.available
                    ? "当前没有新的预警记录，请继续关注田间变化。"
                    : "尚未获取天气数据，暂时无法判断天气风险。"}
                </p>
              </div>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}

function deviceActionLabel(value: string) {
  const labels: Record<string, string> = {
    running: "运行中",
    active: "运行中",
    busy: "执行中",
    working: "工作中",
    start: "启动运行",
    stop: "停止设备",
    power_on: "设备通电",
    power_off: "设备断电",
    irrigate: "执行灌溉",
    fertigate: "执行施肥",
    ventilate: "执行通风",
    light: "执行补光",
    capture: "采集画面",
  };
  return labels[value.toLowerCase()] || value;
}
