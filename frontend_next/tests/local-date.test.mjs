import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(
  new URL("../app/local-date.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { localDateKey } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);

test("香港凌晨和跨年日期按浏览器本地日历保存", () => {
  const previousTimezone = process.env.TZ;
  process.env.TZ = "Asia/Hong_Kong";
  try {
    assert.equal(localDateKey(new Date("2026-11-05T16:30:00Z")), "2026-11-06");
    assert.equal(localDateKey(new Date("2026-12-31T16:30:00Z")), "2027-01-01");
  } finally {
    if (previousTimezone === undefined) delete process.env.TZ;
    else process.env.TZ = previousTimezone;
  }
});
