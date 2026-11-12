import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(
  new URL("../app/camera-stream.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { MjpegParser, multipartBoundary, receiveCameraStream } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);
const encoder = new TextEncoder();
const jpeg = Uint8Array.from([255, 216, 21, 42, 255, 217]);
const second = Uint8Array.from([
  255,
  216,
  ...encoder.encode("--frame\r\n"),
  255,
  217,
]);
function part(image) {
  return Buffer.concat([
    Buffer.from(
      `--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${image.length}\r\n\r\n`,
    ),
    image,
    Buffer.from("\r\n"),
  ]);
}

test("multipart JPEG可以按任意单字节拆分且画面内部的分界文字不会截断帧", () => {
  const parser = new MjpegParser("frame");
  const frames = [];
  for (const byte of Buffer.concat([
    part(jpeg),
    part(second),
    Buffer.from("--frame--\r\n"),
  ])) {
    frames.push(...parser.push(Uint8Array.of(byte)));
  }
  parser.finish();
  assert.deepEqual(
    frames.map((frame) => [...frame]),
    [[...jpeg], [...second]],
  );
});

test("单个网络分块含多帧时全部按Content-Length还原", () => {
  const parser = new MjpegParser("frame");
  const frames = parser.push(Buffer.concat([part(jpeg), part(second)]));
  parser.finish();
  assert.deepEqual(
    frames.map((frame) => [...frame]),
    [[...jpeg], [...second]],
  );
});

test("拒绝缺失非法或过大的帧长度并限制头部缓存", () => {
  for (const length of ["", "-1", "4oops", "4194305", "9007199254740992"]) {
    const parser = new MjpegParser("frame");
    assert.throws(() =>
      parser.push(
        encoder.encode(
          `--frame\r\nContent-Type: image/jpeg\r\n${length ? `Content-Length: ${length}\r\n` : ""}\r\n`,
        ),
      ),
    );
  }
  const parser = new MjpegParser("frame");
  assert.throws(() =>
    parser.push(encoder.encode(`--frame\r\n${"a".repeat(9000)}`)),
  );
});

test("损坏JPEG和半截帧不会被当作可显示画面", () => {
  assert.throws(() =>
    new MjpegParser("frame").push(part(Uint8Array.of(1, 2, 3, 4))),
  );
  const parser = new MjpegParser("frame");
  parser.push(part(jpeg).subarray(0, -4));
  assert.throws(() => parser.finish());
});

test("响应必须是合法带boundary的multipart JPEG流", () => {
  assert.equal(
    multipartBoundary('multipart/x-mixed-replace; boundary="camera-123"'),
    "camera-123",
  );
  for (const value of [
    "image/jpeg",
    "multipart/x-mixed-replace",
    'multipart/x-mixed-replace; boundary="\r\nx"',
  ]) {
    assert.throws(() => multipartBoundary(value));
  }
});

test("流请求保留用户名隔离并只在Authorization中传令牌，结束后释放reader", async (context) => {
  let observed;
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(part(jpeg));
      controller.enqueue(part(second));
      controller.close();
    },
  });
  context.mock.method(globalThis, "fetch", async (url, options) => {
    observed = { url: new URL(url), options };
    return new Response(body, {
      headers: { "Content-Type": "multipart/x-mixed-replace; boundary=frame" },
    });
  });
  const controller = new AbortController();
  const frames = [];
  await assert.rejects(
    receiveCameraStream(
      "camera/a",
      "http://localhost:18001",
      {
        username: "农场 A",
        token: "test-only-token",
      },
      controller.signal,
      (frame) => frames.push([...frame]),
    ),
    /已结束/,
  );
  assert.equal(observed.url.pathname, "/api/devices/camera%2Fa/stream");
  assert.equal(observed.url.searchParams.get("username"), "农场 A");
  assert.equal(observed.url.searchParams.has("token"), false);
  assert.equal(
    observed.options.headers.get("Authorization"),
    "Bearer test-only-token",
  );
  assert.equal(observed.options.signal, controller.signal);
  assert.deepEqual(frames, [[...jpeg], [...second]]);
  assert.equal(body.locked, false);
});

test("取消连接会取消待读取并释放reader，不接收后续画面", async (context) => {
  let cancelled = false;
  const body = new ReadableStream({
    cancel() {
      cancelled = true;
    },
  });
  context.mock.method(
    globalThis,
    "fetch",
    async () =>
      new Response(body, {
        headers: {
          "Content-Type": "multipart/x-mixed-replace; boundary=frame",
        },
      }),
  );
  const controller = new AbortController();
  const frames = [];
  const pending = receiveCameraStream(
    "camera_a",
    "http://localhost:18001",
    {
      username: "test-user",
      token: "",
    },
    controller.signal,
    (frame) => frames.push(frame),
  );
  await Promise.resolve();
  controller.abort();
  await assert.rejects(pending);
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
  assert.deepEqual(frames, []);
});
