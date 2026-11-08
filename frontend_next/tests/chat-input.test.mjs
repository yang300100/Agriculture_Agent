import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(
  new URL("../app/chat-input.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { shouldSendChatKey } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);

test("回车发送，Shift回车保持换行", () => {
  assert.equal(shouldSendChatKey({ key: "Enter" }), true);
  assert.equal(shouldSendChatKey({ key: "Enter", shiftKey: true }), false);
  assert.equal(shouldSendChatKey({ key: "a" }), false);
});

test("中文选词以及浏览器组合输入结束后的同次回车均不发送", () => {
  assert.equal(shouldSendChatKey({ key: "Enter", isComposing: true }), false);
  assert.equal(shouldSendChatKey({ key: "Enter", keyCode: 229 }), false);
  assert.equal(shouldSendChatKey({ key: "Enter" }, true), false);
  assert.equal(shouldSendChatKey({ key: "Enter" }, false, 20), false);
  assert.equal(shouldSendChatKey({ key: "Enter" }, false, 200), true);
});
