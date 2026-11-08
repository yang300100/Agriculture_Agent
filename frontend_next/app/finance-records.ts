export function formatFinanceAmount(value: unknown): string {
  const amount = Number(value ?? 0);
  return Number.isFinite(amount) ? amount.toLocaleString("zh-CN", {
    minimumFractionDigits: 2, maximumFractionDigits: 2,
  }) : "—";
}

type DatedRecord = {
  [key: string]: unknown;
  id?: unknown;
  date?: unknown;
  created_at?: unknown;
};

export function compareRecentFinanceRecords(left: DatedRecord, right: DatedRecord): number {
  return String(right.date || "").localeCompare(String(left.date || ""))
    || String(right.created_at || "").localeCompare(String(left.created_at || ""))
    || String(right.id || "").localeCompare(String(left.id || ""), undefined, { numeric: true });
}
