import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(
  new URL("../app/weather-state.ts", import.meta.url), "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { normalizeFieldWeather, summarizeWeather } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);

test("空响应和非法温度保持不可用，不能成为天气正常的依据", () => {
  for (const payload of [null, {}, [], { temp: null }, { temp: "" },
    { temp: "unknown" }, { temp: Infinity }, { temp: NaN }]) {
    assert.equal(normalizeFieldWeather(payload), null);
  }
});

test("零度和负温度都是有效天气，接口中的数字字符串可以显示", () => {
  assert.deepEqual(normalizeFieldWeather({ temp: 0, text: "晴" }), { temp: 0, text: "晴" });
  assert.deepEqual(normalizeFieldWeather({ temp: "-3.5" }), { temp: -3.5 });
});

test("没有预警数据时明确提示未知，不能宣称总体稳定", () => {
  for (const payload of [null, undefined, {}]) {
    assert.equal(summarizeWeather(payload).available, false);
    assert.equal(summarizeWeather(payload).label, "天气数据暂不可用");
  }
});

test("已获取的无预警结果与确有预警明确区分", () => {
  assert.deepEqual(summarizeWeather({ has_alert: false, count: 0 }), {
    available: true, hasAlert: false, label: "暂无天气预警",
  });
  assert.deepEqual(summarizeWeather({ has_alert: true, count: 2 }), {
    available: true, hasAlert: true, label: "2 条天气预警",
  });
});
