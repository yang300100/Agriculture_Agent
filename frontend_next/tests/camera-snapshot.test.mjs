import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { hookHarness } from "./hook-harness.mjs";
import ts from "typescript";

const source = await readFile(
  new URL("../app/components/Automation.tsx", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(
  `${source}\nexports.CameraLiveView = CameraLiveView;`,
  {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
    },
  },
).outputText;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

const cameraHarness = (request) =>
  hookHarness(compiled, "CameraLiveView", { "../api": { get: request } });

const camera = (device_id = "camera_a") => ({ device_id, name: device_id });
const snapshot = (image = "Zmlyc3Q=") => ({
  success: true,
  image_base64: image,
  mime_type: "image/jpeg",
  timestamp: "2026-11-09T10:00:00",
});

test("离线摄像头可手动抓拍恢复，且不会后台轮询", async () => {
  let calls = 0;
  const harness = cameraHarness(async () => {
    calls++;
    return snapshot();
  });
  harness.render({ device: camera(), offline: true });
  const button = harness.nodes("button")[0];
  assert.equal(button.props.disabled, false);
  assert.equal(calls, 0);
  assert.equal(harness.timers.size, 0);
  button.props.onClick();
  await harness.settle();
  assert.equal(
    harness.nodes("img")[0].props.src,
    "data:image/jpeg;base64,Zmlyc3Q=",
  );
  assert.equal(harness.timers.size, 0);
});

test("在线每五秒继续抓拍，转离线后保留最后快照且停止后台刷新", async () => {
  let calls = 0;
  const harness = cameraHarness(async () => {
    calls++;
    return snapshot();
  });
  harness.render({ device: camera(), offline: false });
  await harness.settle();
  const [timerId, timer] = [...harness.timers][0];
  assert.equal(timer.delay, 5000);
  harness.timers.delete(timerId);
  timer.callback();
  await harness.settle();
  assert.equal(calls, 2);
  harness.render({ device: camera(), offline: true });
  assert.equal(harness.nodes("img").length, 1);
  assert.equal(harness.nodes("alert").length, 1);
  assert.equal(harness.timers.size, 0);
  harness.nodes("button")[0].props.onClick();
  await harness.settle();
  assert.equal(calls, 3);
  assert.equal(harness.nodes("alert").length, 0);
  assert.equal(harness.timers.size, 0);
});

test("慢抓拍未完成时手动刷新不会发出并发请求", async () => {
  const pending = deferred();
  let calls = 0;
  const harness = cameraHarness(() => {
    calls++;
    return pending.promise;
  });
  harness.render({ device: camera(), offline: false });
  harness.nodes("button")[0].props.onClick();
  assert.equal(calls, 1);
  pending.resolve(snapshot());
  await harness.settle();
  assert.equal(harness.nodes("button")[0].props.disabled, false);
  assert.equal(harness.timers.size, 1);
});

test("切换摄像头后旧设备的晚到响应不会覆盖新设备快照", async () => {
  const old = deferred();
  const harness = cameraHarness((path) =>
    path.includes("camera_a") ? old.promise : Promise.resolve(snapshot("bmV3")),
  );
  harness.render({ device: camera(), offline: false });
  harness.render({ device: camera("camera_b"), offline: false });
  await harness.settle();
  old.resolve(snapshot("b2xk"));
  await harness.settle();
  assert.equal(
    harness.nodes("img")[0].props.src,
    "data:image/jpeg;base64,bmV3",
  );
  assert.equal(harness.timers.size, 1);
});

test("抓拍失败保留最后画面并显示接口错误", async () => {
  let calls = 0;
  const harness = cameraHarness(async () =>
    ++calls === 1 ? snapshot() : { success: false, error: "摄像头连接超时" },
  );
  harness.render({ device: camera(), offline: false });
  await harness.settle();
  harness.nodes("button")[0].props.onClick();
  await harness.settle();
  assert.equal(harness.nodes("img").length, 1);
  const errors = harness.nodes("alert");
  assert.equal(errors.length, 1);
  assert.ok(JSON.stringify(errors[0]).includes("摄像头连接超时"));
});

test("卸载后的抓拍响应不会再更新组件或安排轮询", async () => {
  const pending = deferred();
  const harness = cameraHarness(() => pending.promise);
  harness.render({ device: camera(), offline: false });
  harness.unmount();
  pending.resolve(snapshot());
  for (let i = 0; i < 6; i++) await Promise.resolve();
  assert.equal(harness.writesAfterUnmount, 0);
  assert.equal(harness.timers.size, 0);
});

test("本机摄像头驱动使用连续预览而不会启动五秒抓拍", async () => {
  let snapshots = 0;
  const harness = cameraHarness(async () => {
    snapshots++;
    return snapshot();
  });
  harness.render({ device: { ...camera(), driver: "camera" }, offline: false });
  await harness.settle();
  assert.equal(snapshots, 0);
  assert.equal(harness.timers.size, 0);
});
