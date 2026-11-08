import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(
  new URL("../app/rule-editor.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { ruleToDraft, buildRulePayload, updateCondition, saveRuleDraft } =
  await import(
    `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
  );

const rule = {
  id: "12",
  name: "北田灌溉",
  enabled: true,
  execution_mode: "confirm",
  trigger: {
    logic: "AND",
    conditions: [
      {
        type: "sensor",
        field: "soil_moisture",
        op: "<",
        value: 30,
        device_id: "sensor-1",
      },
    ],
  },
  action: {
    device_id: "pump-1",
    command: "start",
    capability: "irrigate",
    params: { duration: 18, pressure: 2.5 },
  },
  constraints: {
    max_duration_per_use: 60,
    max_duration_per_day: 180,
    min_interval_minutes: 0,
    forbidden_hours: [],
    custom_limit: 8,
  },
  ai_enhance: {
    enabled: true,
    can_adjust: ["duration"],
    adjust_range: { duration: [-2, 3] },
    provider: "existing",
  },
};

test("编辑启动时长保留未知动作参数、零冷却、自定义约束和AI配置", () => {
  const draft = ruleToDraft(rule);
  assert.equal(draft.minInterval, "0");
  assert.equal(draft.forbiddenHours, "");
  const payload = buildRulePayload(
    { ...draft, duration: "22", executionMode: "notify" },
    rule,
  );
  assert.deepEqual(payload.action.params, { duration: 22, pressure: 2.5 });
  assert.equal(payload.action.capability, "irrigate");
  assert.deepEqual(payload.constraints, rule.constraints);
  assert.deepEqual(payload.ai_enhance, rule.ai_enhance);
  assert.equal(payload.execution_mode, "notify");
  assert.equal(rule.action.params.duration, 18);
});

test("set_param保持后端单参数格式并正确还原对象值", () => {
  const selected = {
    ...rule,
    action: {
      device_id: "pump-1",
      command: "set_param",
      params: { target: { low: 3, high: 8 } },
    },
  };
  const payload = buildRulePayload(ruleToDraft(selected), selected);
  assert.deepEqual(payload.action.params, { target: { low: 3, high: 8 } });
});

test("条件可视编辑保留传感器来源及高级字段", () => {
  const result = updateCondition(JSON.stringify(rule.trigger.conditions), 0, {
    field: "temperature",
    op: ">=",
    value: "24.5",
  });
  assert.deepEqual(JSON.parse(result), [
    {
      type: "sensor",
      field: "temperature",
      op: ">=",
      value: 24.5,
      device_id: "sensor-1",
    },
  ]);
});

test("拒绝非法条件数组、时长、禁止时段和执行模式", () => {
  const draft = ruleToDraft(rule);
  for (const update of [
    { conditions: "{}" },
    { conditions: "[null]" },
    { duration: "0" },
    { duration: "abc" },
    { duration: "61" },
    { maxUse: "" },
    { minInterval: "-1" },
    { forbiddenHours: "24" },
    { forbiddenHours: "3.5" },
    { forbiddenHours: "1,,2" },
    { executionMode: "unknown" },
  ])
    assert.throws(() => buildRulePayload({ ...draft, ...update }, rule));
});

test("保存并测试先保存当前草稿再按返回ID测试，匹配值来自新草稿", async () => {
  const draft = { ...ruleToDraft(rule), name: "修改后的名称", duration: "26" };
  let stored;
  const result = await saveRuleDraft(
    draft,
    null,
    {
      persist: async (id, payload) => {
        assert.equal(id, undefined);
        stored = payload;
        return { success: true, rule_id: "91" };
      },
      test: async (id) => {
        assert.equal(id, "91");
        assert.equal(stored.action.params.duration, 26);
        assert.equal(stored.name, "修改后的名称");
        return { success: true, rule_matched: true };
      },
    },
    true,
  );
  assert.equal(result.id, "91");
  assert.equal(result.testResult.rule_matched, true);
});

test("保存失败不能继续测试旧版本", async () => {
  let tested = false;
  await assert.rejects(
    saveRuleDraft(
      ruleToDraft(rule),
      rule,
      {
        persist: async () => ({ success: false, error: "保存失败" }),
        test: async () => {
          tested = true;
          return { success: true };
        },
      },
      true,
    ),
    /保存失败/,
  );
  assert.equal(tested, false);
});
