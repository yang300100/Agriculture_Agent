import type { JsonMap, Rule } from "./types";

export type RuleDraft = {
  name: string;
  enabled: boolean;
  logic: string;
  conditions: string;
  device_id: string;
  command: string;
  parameterName: string;
  parameterValue: string;
  duration: string;
  maxUse: string;
  maxDay: string;
  minInterval: string;
  forbiddenHours: string;
  executionMode: string;
  ai: boolean;
};

export function ruleToDraft(rule: Rule | null, defaultDevice = ""): RuleDraft {
  const params = rule?.action?.params || {};
  const parameterName = Object.keys(params)[0] || "duration";
  const value = params[parameterName] ?? 30;
  return {
    name: rule?.name || "",
    enabled: rule?.enabled ?? true,
    logic: rule?.trigger?.logic || "AND",
    conditions: JSON.stringify(
      rule?.trigger?.conditions ?? [
        { type: "sensor", field: "soil_moisture", op: "<", value: 30 },
      ],
      null,
      2,
    ),
    device_id: rule?.action?.device_id || defaultDevice,
    command: rule?.action?.command || "start",
    parameterName,
    parameterValue: typeof value === "string" ? value : JSON.stringify(value),
    duration: String(params.duration ?? 30),
    maxUse: String(rule?.constraints?.max_duration_per_use ?? 60),
    maxDay: String(rule?.constraints?.max_duration_per_day ?? 180),
    minInterval: String(rule?.constraints?.min_interval_minutes ?? 120),
    forbiddenHours: (
      rule?.constraints?.forbidden_hours ?? [22, 23, 0, 1, 2, 3, 4, 5]
    ).join(", "),
    executionMode: rule?.execution_mode || "auto",
    ai: rule?.ai_enhance?.enabled ?? false,
  };
}

export function parseRuleValue(value: string): unknown {
  try {
    return JSON.parse(value.trim());
  } catch {
    return value;
  }
}

export function parseConditions(source: string): JsonMap[] {
  let value: unknown;
  try {
    value = JSON.parse(source);
  } catch {
    throw new Error("条件 JSON 格式错误");
  }
  if (
    !Array.isArray(value) ||
    value.some(
      (item) => !item || typeof item !== "object" || Array.isArray(item),
    )
  ) {
    throw new Error("触发条件必须是条件对象组成的 JSON 数组");
  }
  return value;
}

export function updateCondition(
  source: string,
  index: number,
  changes: JsonMap,
): string {
  const conditions = parseConditions(source);
  if (!conditions[index]) throw new Error("该条件不存在");
  conditions[index] = {
    ...conditions[index],
    ...changes,
    ...(Object.hasOwn(changes, "value")
      ? { value: parseRuleValue(String(changes.value)) }
      : {}),
  };
  return JSON.stringify(conditions, null, 2);
}

function numberField(value: string, label: string, minimum: number): number {
  const number = Number(value);
  if (!value.trim() || !Number.isFinite(number) || number < minimum) {
    throw new Error(`${label}须为不小于 ${minimum} 的数字`);
  }
  return number;
}

export function buildRulePayload(
  form: RuleDraft,
  selected: Rule | null,
): JsonMap {
  if (!form.name.trim()) throw new Error("请输入规则名称");
  if (!form.device_id) throw new Error("请先在设备中心注册一个设备");
  if (!["AND", "OR"].includes(form.logic))
    throw new Error("触发逻辑只能为 AND 或 OR");
  if (!["auto", "confirm", "notify"].includes(form.executionMode))
    throw new Error("请选择有效的执行模式");
  const maxUse = numberField(form.maxUse, "单次最长时长", 1);
  const maxDay = numberField(form.maxDay, "每日时长上限", 1);
  const minInterval = numberField(form.minInterval, "最小间隔", 0);
  const hours = form.forbiddenHours.trim()
    ? form.forbiddenHours.split(/[,，]/).map((hour) => {
        const number = Number(hour.trim());
        if (
          !hour.trim() ||
          !Number.isInteger(number) ||
          number < 0 ||
          number > 23
        ) {
          throw new Error(
            "禁止时段须填写 0 至 23 的整点小时，以逗号分隔；留空表示不限制",
          );
        }
        return number;
      })
    : [];
  const action =
    selected?.action?.device_id === form.device_id ? selected.action : {};
  let params: JsonMap = {};
  if (form.command === "start") {
    const duration = numberField(form.duration, "启动时长", 0);
    if (duration === 0) throw new Error("启动时长须大于 0 分钟");
    if (duration > maxUse) throw new Error("启动时长不能超过单次最长时长");
    params = { ...(action.command === "start" ? action.params : {}), duration };
  } else if (form.command === "set_param") {
    if (!form.parameterName.trim()) throw new Error("请输入要设置的参数名");
    params = {
      [form.parameterName.trim()]: parseRuleValue(form.parameterValue),
    };
  }
  return {
    name: form.name.trim(),
    enabled: form.enabled,
    execution_mode: form.executionMode,
    trigger: {
      ...selected?.trigger,
      logic: form.logic,
      conditions: parseConditions(form.conditions),
    },
    action: {
      ...action,
      device_id: form.device_id,
      command: form.command,
      params,
    },
    constraints: {
      ...selected?.constraints,
      max_duration_per_use: maxUse,
      max_duration_per_day: maxDay,
      min_interval_minutes: minInterval,
      forbidden_hours: [...new Set(hours)],
    },
    ai_enhance: {
      ...(selected?.ai_enhance ?? {
        can_adjust: ["duration"],
        adjust_range: { duration: [-10, 10] },
      }),
      enabled: form.ai,
    },
  };
}

export async function saveRuleDraft(
  draft: RuleDraft,
  selected: Rule | null,
  transport: {
    persist: (id: string | undefined, payload: JsonMap) => Promise<JsonMap>;
    test: (id: string) => Promise<JsonMap>;
  },
  testAfterSave = false,
) {
  const payload = buildRulePayload(draft, selected);
  const saved = await transport.persist(
    selected ? String(selected.id) : undefined,
    payload,
  );
  if (!saved.success) throw new Error(saved.error || "规则保存失败");
  const id = String(selected?.id ?? saved.rule_id ?? "");
  if (!id) throw new Error("保存响应缺少规则 ID，请刷新规则列表后重试");
  let testResult: JsonMap | undefined;
  let testError = "";
  if (testAfterSave) {
    try {
      testResult = await transport.test(id);
      if (!testResult.success)
        throw new Error(testResult.error || "规则条件测试失败");
    } catch (reason) {
      testError = reason instanceof Error ? reason.message : "规则条件测试失败";
    }
  }
  return { id, payload, testResult, testError };
}
