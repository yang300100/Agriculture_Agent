type CalendarTask = { status: string; end_date?: string };
export type TaskFilter =
  "active" | "all" | "completed" | "running" | "attention";

export function isTaskRunning(task: {
  status: string;
  device_id?: string;
}): boolean {
  return (
    ["执行中", "运行中"].includes(task.status) ||
    (task.status === "进行中" && !!task.device_id)
  );
}

export function selectCalendarTasks<T extends CalendarTask>(
  tasks: T[],
  filter: TaskFilter,
  date: string,
): T[] {
  return tasks
    .filter((task) => {
      if (date && task.end_date?.slice(0, 10) !== date) return false;
      if (filter === "completed") return task.status === "已完成";
      if (filter === "running")
        return isTaskRunning(task) || task.status === "进行中";
      if (filter === "attention")
        return ["失败", "已中止", "已逾期"].includes(task.status);
      return filter === "all" || task.status !== "已完成";
    })
    .sort((a, b) =>
      String(a.end_date || "9999").localeCompare(String(b.end_date || "9999")),
    );
}

export function overallProgress(item: Record<string, any>): number {
  if (item.status === "待开始") return 0;
  if (item.status === "已完成") return 100;
  const clamp = (value: unknown) =>
    Math.min(100, Math.max(0, Number(value) || 0));
  if (item.overall_progress != null) return clamp(item.overall_progress);
  const stage = clamp(
    item.stage_progress ?? item.progress_percent ?? item.progress ?? 0,
  );
  if (Number(item.total_stages) > 0 && Number(item.stage_number) > 0) {
    return clamp(
      Math.round(
        ((Number(item.stage_number) - 1) * 100 + stage) /
          Number(item.total_stages),
      ),
    );
  }
  return stage;
}
