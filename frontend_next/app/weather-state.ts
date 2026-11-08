export type FieldWeather = Record<string, unknown> & { temp: number };

export function normalizeFieldWeather(value: unknown): FieldWeather | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const data = value as Record<string, unknown>;
  const raw = data.temp;
  if (typeof raw !== "number" && typeof raw !== "string") return null;
  if (typeof raw === "string" && !raw.trim()) return null;
  const temperature = Number(raw);
  return Number.isFinite(temperature) ? { ...data, temp: temperature } : null;
}

export function summarizeWeather(value: unknown) {
  const data = value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
  if (!data || typeof data.has_alert !== "boolean") {
    return { available: false, hasAlert: false, label: "天气数据暂不可用" };
  }
  const count = Number(data.count);
  const alertCount = Number.isFinite(count) && count > 0 ? Math.floor(count) : 1;
  return {
    available: true,
    hasAlert: data.has_alert,
    label: data.has_alert ? `${alertCount} 条天气预警` : "暂无天气预警",
  };
}
