import type { ChatCardReference, ChatMessage, JsonMap } from "./types";

export function chatCardKey(card: ChatCardReference): string {
  return card.type === "device"
    ? JSON.stringify(["device", card.device_id, card.pending_action_id || ""])
    : JSON.stringify(["task", card.task_id]);
}

export function normalizeChatCards(value: unknown): ChatCardReference[] {
  if (!Array.isArray(value)) return [];
  const result: ChatCardReference[] = [];
  const seen = new Set<string>();
  const validId = (id: unknown): id is string =>
    typeof id === "string" && !!id.trim() && id.length <= 256;
  for (const item of value) {
    if (!item || typeof item !== "object") continue;
    let card: ChatCardReference;
    if (item.type === "device" && validId(item.device_id)) {
      card = { type: "device", device_id: item.device_id };
      if (validId(item.pending_action_id))
        card.pending_action_id = item.pending_action_id;
    } else if (item.type === "task" && validId(item.task_id)) {
      card = { type: "task", task_id: item.task_id };
    } else continue;
    const key = chatCardKey(card);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(card);
    if (result.length === 8) break;
  }
  return result;
}

export function inlinePendingIds(messages: ChatMessage[]): Set<string> {
  return new Set(
    messages.flatMap((message) =>
      normalizeChatCards(message.cards).flatMap((card) =>
        card.type === "device" && card.pending_action_id
          ? [card.pending_action_id]
          : [],
      ),
    ),
  );
}

export function deviceObservation(state: JsonMap = {}, now = Date.now()) {
  const observedAt =
    typeof state._observed_at === "string" ? state._observed_at : "";
  const observed = Date.parse(observedAt);
  const age = now - observed;
  const fresh =
    state._quality === "good" &&
    !state._error &&
    !state.error &&
    state.online !== false &&
    state._fresh !== false &&
    Number.isFinite(observed) &&
    age >= 0 &&
    age <= 300_000;
  const operation =
    (
      {
        running: "运行中",
        standby: "待机",
        idle: "空闲",
        stopped: "已停止",
        powered_off: "已关机",
      } as Record<string, string>
    )[String(state.status)] || "运行状态未知";
  return {
    fresh,
    connection: fresh ? "在线" : "在线状态未知",
    operation: fresh ? operation : "运行状态未知",
    observedAt,
  };
}

export function canPrepareDevice(capabilities: string[] = []): boolean {
  return controllableCapabilities(capabilities).length > 0;
}

export function runningTimeLabel(
  state: JsonMap,
  now = Date.now(),
): string | null {
  if (!deviceObservation(state, now).fresh || state.status !== "running")
    return null;
  const deadline = state.run_until;
  if (
    typeof deadline !== "number" ||
    !Number.isFinite(deadline) ||
    deadline <= 0 ||
    !Number.isFinite(new Date(deadline * 1000).getTime())
  )
    return "结束时间未知";
  const remaining = deadline * 1000 - now;
  return remaining > 0
    ? `预计剩余 ${Math.ceil(remaining / 60_000)} 分钟`
    : "等待状态确认";
}

export function controllableCapabilities(
  capabilities: string[] = [],
): string[] {
  return capabilities.filter((capability) =>
    [
      "irrigate",
      "fertigate",
      "ventilate",
      "heat",
      "cool",
      "shade",
      "light",
    ].includes(capability),
  );
}

export function startParams(duration: string): { duration: number } {
  const value = Number(duration);
  if (!duration.trim() || !Number.isFinite(value) || value < 1)
    throw new Error("运行时长须为至少 1 分钟的有效数字。");
  return { duration: value };
}

export function prepareParams(
  capabilities: string[],
  selected: string,
  command: "start" | "stop",
  duration: string,
): JsonMap {
  const options = controllableCapabilities(capabilities);
  const action = selected || (options.length === 1 ? options[0] : "");
  if (!options.includes(action))
    throw new Error("请先选择此设备要执行的操作能力。");
  return { action, ...(command === "start" ? startParams(duration) : {}) };
}

export function taskStatusLabel(status: string): string {
  if (
    [
      "待办",
      "进行中",
      "已完成",
      "已逾期",
      "运行中",
      "执行中",
      "已中止",
      "失败",
      "已取消",
    ].includes(status)
  )
    return status;
  return (
    (
      {
        pending: "待办",
        in_progress: "进行中",
        completed: "已完成",
        cancelled: "已取消",
        overdue: "已逾期",
        scheduled: "已预约",
        failed: "失败",
      } as Record<string, string>
    )[status] || "状态未知"
  );
}

export function taskReminderDetails(
  task: JsonMap,
): { time: string; frequency: string } | null {
  const time = task.reminder_time;
  if (
    typeof time !== "string" ||
    !time.includes("T") ||
    !Number.isFinite(Date.parse(time))
  )
    return null;
  const raw = String(task.reminder_frequency || "");
  const frequency = ["单次", "每天", "每周", "每月", "自定义"].includes(raw)
    ? raw
    : (
        {
          once: "单次",
          daily: "每天",
          weekly: "每周",
          monthly: "每月",
          custom: "自定义",
        } as Record<string, string>
      )[raw] || "频率未知";
  return { time, frequency };
}

const numericParameterKeys = [
  "duration",
  "duration_minutes",
  "amount_kg",
  "target_temp",
  "brightness_percent",
  "flow_rate",
  "volume_liters",
  "target_soil_moisture",
  "concentration_percent",
  "speed_percent",
  "target_lux",
  "power_percent",
  "position_percent",
];

export function pendingNumericDrafts(params: JsonMap): Record<string, string> {
  return Object.fromEntries(
    Object.entries(params)
      .filter(([key]) => numericParameterKeys.includes(key))
      .map(([key, value]) => [key, String(value)]),
  );
}

export function mergePendingParameters(
  original: JsonMap,
  drafts: Record<string, string>,
  advanced?: string,
): JsonMap {
  let extras: JsonMap = {};
  if (advanced !== undefined) {
    try {
      extras = JSON.parse(advanced);
    } catch {
      throw new Error("高级参数须为有效的 JSON 对象。");
    }
    if (!extras || typeof extras !== "object" || Array.isArray(extras))
      throw new Error("高级参数须为 JSON 对象。");
  }
  const result = { ...original, ...extras };
  for (const key of ["action", "capability"]) {
    if (Object.hasOwn(original, key)) result[key] = original[key];
    else delete result[key];
  }
  for (const [key, raw] of Object.entries(drafts)) {
    if (!numericParameterKeys.includes(key))
      throw new Error("该参数不支持数字编辑。");
    if (["duration", "duration_minutes"].includes(key))
      result[key] = startParams(raw).duration;
    else {
      const value = Number(raw);
      if (!raw.trim() || !Number.isFinite(value))
        throw new Error("操作参数须为有效数字，不能为空。");
      result[key] = value;
    }
  }
  return result;
}

export function resolvesUnknownExecution(
  expectedId: string,
  pending: JsonMap | null | undefined,
  fresh: boolean,
  reconcileResult?: JsonMap,
): boolean {
  if (
    reconcileResult?.success === true &&
    reconcileResult.status === "reconciled"
  )
    return true;
  return (
    fresh &&
    !!expectedId &&
    pending?.id === expectedId &&
    ["executed", "reconciled", "rejected", "cancelled", "expired"].includes(
      String(pending.status),
    )
  );
}

export function createChatRequestGuard() {
  let generation = 0;
  let current: symbol | null = null;
  return {
    begin() {
      if (current) return null;
      const token = { generation, id: Symbol() };
      current = token.id;
      return token;
    },
    isCurrent(token: { generation: number; id: symbol } | null) {
      return !!token && generation === token.generation && current === token.id;
    },
    finish(token: { generation: number; id: symbol } | null) {
      if (token && generation === token.generation && current === token.id)
        current = null;
    },
    invalidate() {
      generation += 1;
      current = null;
    },
  };
}
