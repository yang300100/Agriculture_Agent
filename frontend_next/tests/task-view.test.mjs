import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(
  new URL("../app/task-view.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { selectCalendarTasks, isTaskRunning, overallProgress } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);
const rows = [
  { id: "done", status: "已完成", end_date: "2026-11-07" },
  { id: "later", status: "待办", end_date: "2026-11-09" },
  { id: "running", status: "运行中", end_date: "2026-11-07" },
  { id: "failed", status: "失败", end_date: "2026-11-06" },
];

test("日历可回查完成历史，并能按日期筛选运行任务", () => {
  assert.deepEqual(
    selectCalendarTasks(rows, "completed", "").map((t) => t.id),
    ["done"],
  );
  assert.deepEqual(
    selectCalendarTasks(rows, "active", "2026-11-07").map((t) => t.id),
    ["running"],
  );
  assert.deepEqual(
    selectCalendarTasks(rows, "all", "").map((t) => t.id),
    ["failed", "done", "running", "later"],
  );
  assert.equal(rows[0].id, "done");
});

test("正在领取和实际运行的设备任务均不能重复执行或勾选完成", () => {
  assert.equal(isTaskRunning({ status: "执行中" }), true);
  assert.equal(isTaskRunning({ status: "进行中", device_id: "pump_01" }), true);
  assert.equal(isTaskRunning({ status: "运行中" }), true);
  assert.equal(isTaskRunning({ status: "待办" }), false);
  assert.equal(isTaskRunning({ status: "进行中" }), false);
});

test("整季进度优先使用后端值，旧卡片按阶段折算且零值不被覆盖", () => {
  assert.equal(overallProgress({ overall_progress: 0, progress: 90 }), 0);
  assert.equal(overallProgress({ overall_progress: 42 }), 42);
  assert.equal(
    overallProgress({ stage_number: 2, total_stages: 5, stage_progress: 0 }),
    20,
  );
  assert.equal(
    overallProgress({ stage_number: 2, total_stages: 5, progress: 50 }),
    30,
  );
  assert.equal(overallProgress({ status: "待开始", progress: 90 }), 0);
});
