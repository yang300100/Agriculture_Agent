import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
import { hookHarness } from "./hook-harness.mjs";

const source = await readFile(
  new URL("../app/components/CameraStreamView.tsx", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
    jsx: ts.JsxEmit.ReactJSX,
  },
}).outputText;
const device = (device_id = "camera_a") => ({
  device_id,
  driver: "camera",
  name: device_id,
});
const jpeg = Uint8Array.of(255, 216, 0, 255, 217);

function previewHarness() {
  let identity = { username: "test-user", token: "test-token" };
  const connections = [];
  const revoked = [];
  const blobs = [];
  const listeners = new Map();
  const harness = hookHarness(
    compiled,
    "CameraStreamView",
    {
      "../api": {
        getIdentity: () => identity,
        getApiBase: () => "http://localhost:18001",
      },
      "../camera-stream": {
        receiveCameraStream(deviceId, base, user, signal, onFrame) {
          let reject;
          const promise = new Promise((_resolve, no) => {
            reject = no;
          });
          connections.push({ deviceId, base, user, signal, onFrame, reject });
          return promise;
        },
      },
    },
    {
      btoa,
      URL: {
        createObjectURL(blob) {
          blobs.push(blob);
          return `blob:frame-${blobs.length}`;
        },
        revokeObjectURL(url) {
          revoked.push(url);
        },
      },
      window: {
        addEventListener(name, callback) {
          listeners.set(name, callback);
        },
        removeEventListener(name) {
          listeners.delete(name);
        },
      },
    },
  );
  return Object.assign(harness, {
    connections,
    revoked,
    blobs,
    listeners,
    switchIdentity() {
      identity = { username: "another-user", token: "another-token" };
      listeners.get("storage")?.({ key: "agri_username" });
    },
  });
}

test("本机摄像头自动连接且收到首帧后才显示实时画面，重入不会并发", async () => {
  const harness = previewHarness();
  harness.render({ device: device() });
  assert.equal(harness.connections.length, 1);
  assert.equal(harness.nodes("img").length, 0);
  assert.equal(harness.nodes("button")[0].props.disabled, true);
  harness.nodes("button")[0].props.onClick();
  assert.equal(harness.connections.length, 1);
  harness.connections[0].onFrame(jpeg);
  await harness.settle();
  assert.equal(harness.nodes("img")[0].props.src, "blob:frame-1");
  assert.equal(harness.nodes("img")[0].props["data-live"], true);
  harness.connections[0].onFrame(jpeg);
  await harness.settle();
  assert.equal(harness.nodes("img")[0].props.src, "blob:frame-2");
  assert.deepEqual(harness.revoked, ["blob:frame-1"]);
  assert.equal(harness.blobs[0].type, "image/jpeg");
});

test("视频流错误保留最后画面并标记过期，手动重连后首帧恢复", async () => {
  const harness = previewHarness();
  harness.render({ device: device() });
  assert.equal(harness.connections.length, 1);
  harness.connections[0].onFrame(jpeg);
  await harness.settle();
  harness.connections[0].reject(new Error("USB 摄像头断开"));
  await harness.settle();
  assert.equal(harness.connections[0].signal.aborted, true);
  assert.ok(harness.revoked.includes("blob:frame-1"));
  assert.equal(
    harness.nodes("img")[0].props.src,
    "data:image/jpeg;base64,/9gA/9k=",
  );
  assert.equal(harness.nodes("img")[0].props["data-live"], false);
  assert.equal(harness.nodes("alert").length, 1);
  assert.ok(
    JSON.stringify(harness.nodes("alert")[0]).includes("USB 摄像头断开"),
  );
  assert.equal(harness.connections.length, 1);
  assert.equal(harness.timers.size, 0);
  harness.nodes("button")[0].props.onClick();
  await harness.settle();
  assert.equal(harness.connections.length, 2);
  harness.connections[1].onFrame(jpeg);
  await harness.settle();
  assert.equal(harness.nodes("alert").length, 0);
  assert.equal(harness.nodes("img")[0].props["data-live"], true);
});

test("切换设备与卸载会取消旧流并释放URL，晚到帧不能写回", async () => {
  const harness = previewHarness();
  harness.render({ device: device() });
  assert.equal(harness.connections.length, 1);
  harness.connections[0].onFrame(jpeg);
  await harness.settle();
  harness.render({ device: device("camera_b") });
  assert.equal(harness.connections[0].signal.aborted, true);
  assert.deepEqual(harness.revoked, ["blob:frame-1"]);
  harness.connections[0].onFrame(jpeg);
  harness.connections[1].onFrame(jpeg);
  await harness.settle();
  assert.equal(harness.nodes("img")[0].props.src, "blob:frame-2");
  harness.unmount();
  assert.equal(harness.connections[1].signal.aborted, true);
  assert.deepEqual(harness.revoked, ["blob:frame-1", "blob:frame-2"]);
  harness.connections[1].onFrame(jpeg);
  assert.equal(harness.writesAfterUnmount, 0);
  assert.equal(harness.timers.size, 0);
});

test("跨账号切换立即取消视频且清除旧账号画面", async () => {
  const harness = previewHarness();
  harness.render({ device: device() });
  assert.equal(harness.connections.length, 1);
  harness.connections[0].onFrame(jpeg);
  await harness.settle();
  harness.switchIdentity();
  await harness.settle();
  assert.equal(harness.connections[0].signal.aborted, true);
  assert.ok(harness.revoked.includes("blob:frame-1"));
  assert.equal(harness.nodes("img").length, 0);
  harness.connections[0].onFrame(jpeg);
  await harness.settle();
  assert.equal(harness.nodes("img").length, 0);
});

test("长时间没有新帧会停止LIVE并取消连接，不自动重连", async () => {
  const harness = previewHarness();
  harness.render({ device: device() });
  harness.connections[0].onFrame(jpeg);
  await harness.settle();
  const [timerId, timer] = [...harness.timers][0];
  assert.equal(timer.delay, 15000);
  harness.timers.delete(timerId);
  timer.callback();
  await harness.settle();
  assert.equal(harness.connections[0].signal.aborted, true);
  assert.equal(harness.nodes("img")[0].props["data-live"], false);
  assert.equal(harness.nodes("alert").length, 1);
  assert.equal(harness.nodes("button")[0].props.disabled, false);
  assert.equal(harness.connections.length, 1);
  assert.equal(harness.timers.size, 0);
});

test("浏览器不能解码画面时停止LIVE并保留可重连的错误状态", async () => {
  const harness = previewHarness();
  harness.render({ device: device() });
  harness.connections[0].onFrame(jpeg);
  await harness.settle();
  harness.nodes("img")[0].props.onError();
  await harness.settle();
  assert.equal(harness.connections[0].signal.aborted, true);
  assert.equal(harness.nodes("img")[0].props["data-live"], false);
  assert.equal(harness.nodes("alert").length, 1);
  assert.equal(harness.nodes("button")[0].props.disabled, false);
});
