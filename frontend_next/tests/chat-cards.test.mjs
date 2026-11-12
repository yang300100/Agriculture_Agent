import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(
  new URL("../app/chat-cards.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const {
  normalizeChatCards,
  chatCardKey,
  inlinePendingIds,
  deviceObservation,
  runningTimeLabel,
  canPrepareDevice,
  startParams,
  prepareParams,
  taskStatusLabel,
  taskReminderDetails,
  pendingNumericDrafts,
  mergePendingParameters,
  resolvesUnknownExecution,
  createChatRequestGuard,
} = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);

test("卡片引用只保留白名单标识，正文链接和旧状态不能成为操作", () => {
  assert.deepEqual(
    normalizeChatCards([
      {
        type: "device",
        device_id: "pump/a",
        pending_action_id: "p1",
        command: "start",
        params: { duration: 99 },
        state: { status: "running" },
      },
      { type: "task", task_id: "t1", device_command: "start" },
      { type: "link", url: "/api/devices/pump/command" },
      { type: "device", device_id: "" },
      null,
    ]),
    [
      { type: "device", device_id: "pump/a", pending_action_id: "p1" },
      { type: "task", task_id: "t1" },
    ],
  );
  assert.deepEqual(normalizeChatCards("[启动](device:pump/start)"), []);
});

test("多设备和不同候选保持独立，重复引用去重且最多八项", () => {
  const cards = normalizeChatCards([
    { type: "device", device_id: "a" },
    { type: "device", device_id: "b" },
    { type: "device", device_id: "a", pending_action_id: "p1" },
    { type: "device", device_id: "a", pending_action_id: "p1" },
    ...Array.from({ length: 10 }, (_, index) => ({
      type: "task",
      task_id: `t${index}`,
    })),
  ]);
  assert.equal(cards.length, 8);
  assert.notEqual(chatCardKey(cards[0]), chatCardKey(cards[1]));
  assert.notEqual(chatCardKey(cards[0]), chatCardKey(cards[2]));
  assert.deepEqual(
    [...inlinePendingIds([{ role: "assistant", content: "", cards }])],
    ["p1"],
  );
});

test("只有质量合格且五分钟内的真实观测可以显示在线和运行状态", () => {
  const now = Date.parse("2026-11-09T10:00:00Z");
  const state = {
    _quality: "good",
    _observed_at: "2026-11-09T09:59:00Z",
    status: "running",
  };
  assert.deepEqual(deviceObservation(state, now), {
    fresh: true,
    connection: "在线",
    operation: "运行中",
    observedAt: state._observed_at,
  });
  for (const change of [
    { _quality: "unknown" },
    { _error: "timeout" },
    { error: "offline" },
    { online: false },
    { _fresh: false },
    { _observed_at: "2026-11-09T09:54:59Z" },
    { _observed_at: "2026-11-09T10:00:01Z" },
    { _observed_at: "bad-date" },
  ]) {
    const result = deviceObservation({ ...state, ...change }, now);
    assert.equal(result.fresh, false);
    assert.equal(result.connection, "在线状态未知");
    assert.equal(result.operation, "运行状态未知");
  }
  assert.equal(
    deviceObservation({ ...state, status: "standby" }, now).operation,
    "待机",
  );
  assert.equal(
    deviceObservation({ ...state, status: "mystery" }, now).operation,
    "运行状态未知",
  );
});

test("多能力设备须明确选择能力，停止也保留所选能力", () => {
  assert.throws(() =>
    prepareParams(["irrigate", "fertigate"], "", "start", "10"),
  );
  assert.throws(() =>
    prepareParams(["read_sensor"], "read_sensor", "start", "10"),
  );
  assert.deepEqual(
    prepareParams(["irrigate", "fertigate"], "fertigate", "start", "10"),
    { action: "fertigate", duration: 10 },
  );
  assert.deepEqual(
    prepareParams(["irrigate", "fertigate"], "fertigate", "stop", ""),
    { action: "fertigate" },
  );
  assert.deepEqual(prepareParams(["irrigate"], "", "start", "2"), {
    action: "irrigate",
    duration: 2,
  });
});

test("传感器不能准备启动，时长拒绝空值零负数和非有限数", () => {
  assert.equal(canPrepareDevice(["read_sensor"]), false);
  assert.equal(canPrepareDevice(["capture"]), false);
  assert.equal(canPrepareDevice(["irrigate"]), true);
  assert.deepEqual(startParams("2.5"), { duration: 2.5 });
  for (const value of ["", " ", "0", "0.5", "-1", "Infinity", "NaN", "abc"]) {
    assert.throws(() => startParams(value));
  }
});

test("本地未知结果只接受原操作的新鲜终态或后端明确核对证据", () => {
  assert.equal(
    resolvesUnknownExecution("p1", { id: "p1", status: "executed" }, true),
    true,
  );
  assert.equal(
    resolvesUnknownExecution("p1", { id: "p1", status: "reconciled" }, true),
    true,
  );
  for (const status of ["rejected", "cancelled", "expired"]) {
    assert.equal(
      resolvesUnknownExecution("p1", { id: "p1", status }, true),
      true,
    );
    assert.equal(
      resolvesUnknownExecution("p1", { id: "p1", status }, false),
      false,
    );
    assert.equal(
      resolvesUnknownExecution("p1", { id: "p2", status }, true),
      false,
    );
  }
  assert.equal(
    resolvesUnknownExecution("p1", { id: "p1", status: "executed" }, false),
    false,
  );
  assert.equal(
    resolvesUnknownExecution("p1", { id: "p2", status: "executed" }, true),
    false,
  );
  assert.equal(
    resolvesUnknownExecution("p1", { id: "p1", status: "pending" }, true),
    false,
  );
  assert.equal(
    resolvesUnknownExecution("p1", { id: "p1", status: "failed" }, true),
    false,
  );
  assert.equal(
    resolvesUnknownExecution("p1", null, true, {
      success: true,
      message: "已核对",
    }),
    false,
  );
  assert.equal(
    resolvesUnknownExecution("p1", null, true, {
      success: false,
      status: "reconciled",
    }),
    false,
  );
  assert.equal(
    resolvesUnknownExecution("p1", null, false, {
      success: true,
      status: "reconciled",
    }),
    true,
  );
});

test("同一提交同步防重，切换会话后旧请求不能写回新会话", () => {
  const guard = createChatRequestGuard();
  const request = guard.begin();
  assert.notEqual(request, null);
  assert.equal(guard.begin(), null);
  guard.invalidate();
  const next = guard.begin();
  assert.equal(guard.isCurrent(request), false);
  guard.finish(request);
  assert.equal(guard.begin(), null);
  assert.equal(guard.isCurrent(next), true);
  guard.finish(next);
  assert.notEqual(guard.begin(), null);
});

test("预约任务保留后端真实中文状态，未知枚举不能伪装成待办", () => {
  assert.equal(taskStatusLabel("待办"), "待办");
  assert.equal(taskStatusLabel("运行中"), "运行中");
  assert.equal(taskStatusLabel("已中止"), "已中止");
  assert.equal(taskStatusLabel("completed"), "已完成");
  assert.equal(taskStatusLabel("unexpected"), "状态未知");
});

test("预约提醒使用真实绝对时间，缺少记录时不从正文猜测", () => {
  assert.deepEqual(
    taskReminderDetails({
      reminder_time: "2026-11-10T15:00:00",
      reminder_frequency: "单次",
    }),
    { time: "2026-11-10T15:00:00", frequency: "单次" },
  );
  assert.deepEqual(
    taskReminderDetails({
      reminder_time: "2026-11-10T15:00:00",
      reminder_frequency: "weekly",
    }),
    { time: "2026-11-10T15:00:00", frequency: "每周" },
  );
  for (const reminder_frequency of ["每月", "monthly"]) {
    assert.deepEqual(
      taskReminderDetails({
        reminder_time: "2026-11-10T15:00:00",
        reminder_frequency,
      }),
      { time: "2026-11-10T15:00:00", frequency: "每月" },
    );
  }
  assert.equal(
    taskReminderDetails({
      start_date: "2026-11-10",
      description: "明天下午3点灌溉",
    }),
    null,
  );
  assert.equal(
    taskReminderDetails({
      reminder_time: "invalid",
      reminder_frequency: "单次",
    }),
    null,
  );
});

test("可信运行快照才显示预计剩余，期限经过不能推断设备停止", () => {
  const now = Date.parse("2026-11-09T10:00:00Z");
  const state = {
    _quality: "good",
    _observed_at: "2026-11-09T09:59:59Z",
    status: "running",
    run_until: now / 1000 + 90,
  };
  assert.equal(runningTimeLabel(state, now), "预计剩余 2 分钟");
  assert.equal(
    runningTimeLabel({ ...state, run_until: now / 1000 - 1 }, now),
    "等待状态确认",
  );
  for (const run_until of [
    null,
    undefined,
    0,
    -1,
    Infinity,
    NaN,
    "invalid",
    1e20,
  ]) {
    assert.equal(
      runningTimeLabel({ ...state, run_until }, now),
      "结束时间未知",
    );
  }
  assert.equal(runningTimeLabel({ ...state, status: "standby" }, now), null);
  assert.equal(runningTimeLabel({ ...state, _quality: "unknown" }, now), null);
  assert.equal(
    runningTimeLabel({ ...state, _observed_at: "2026-11-09T09:54:59Z" }, now),
    null,
  );
});

test("常用数字参数编辑保留未知字段与原能力，非法数字不能保存", () => {
  const original = {
    duration: 1,
    amount_kg: 2,
    action: "fertigate",
    capability: "fertigate",
    custom: { mode: "gentle" },
  };
  assert.deepEqual(pendingNumericDrafts(original), {
    duration: "1",
    amount_kg: "2",
  });
  assert.deepEqual(
    mergePendingParameters(original, { duration: "2", amount_kg: "3.5" }),
    {
      duration: 2,
      amount_kg: 3.5,
      action: "fertigate",
      capability: "fertigate",
      custom: { mode: "gentle" },
    },
  );
  assert.deepEqual(
    mergePendingParameters(
      original,
      { duration: "2" },
      '{"action":"irrigate","capability":"irrigate","extra":42}',
    ),
    {
      duration: 2,
      amount_kg: 2,
      action: "fertigate",
      capability: "fertigate",
      custom: { mode: "gentle" },
      extra: 42,
    },
  );
  assert.equal(original.duration, 1);
  for (const value of ["", " ", "0.5", "NaN", "Infinity", "abc"])
    assert.throws(() => mergePendingParameters(original, { duration: value }));
  assert.throws(() => mergePendingParameters(original, {}, "[]"));
  assert.throws(() => mergePendingParameters(original, {}, "invalid"));
});
