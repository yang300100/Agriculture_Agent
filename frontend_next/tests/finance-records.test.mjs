import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../app/finance-records.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { formatFinanceAmount, compareRecentFinanceRecords } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);

test("经营表格保留分币，不能把1097.50元显示成1098元", () => {
  assert.equal(formatFinanceAmount(1097.5), "1,097.50");
  assert.equal(formatFinanceAmount(38.5), "38.50");
  assert.equal(formatFinanceAmount(0), "0.00");
  assert.equal(formatFinanceAmount(-0.5), "-0.50");
});

test("同一天的新流水按记录时间和ID排前，不会被旧流水挤出最近列表", () => {
  const rows = [
    { id: 1, date: "2026-11-05", created_at: "2026-11-05 08:00:00" },
    { id: 2, date: "2026-11-05", created_at: "2026-11-05 09:00:00" },
    { id: 3, date: "2026-11-05", created_at: "2026-11-05 09:00:00" },
    { id: 4, date: "2026-11-04", created_at: "2026-11-05 10:00:00" },
  ];
  assert.deepEqual(rows.sort(compareRecentFinanceRecords).map((row) => row.id), [3, 2, 1, 4]);
});
