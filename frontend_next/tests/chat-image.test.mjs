import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
import { hookHarness } from "./hook-harness.mjs";

const source = await readFile(
  new URL("../app/components/Chat.tsx", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
    jsx: ts.JsxEmit.ReactJSX,
  },
}).outputText;

async function realModule(path) {
  const text = await readFile(new URL(path, import.meta.url), "utf8");
  const output = ts.transpileModule(text, {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  return import(
    `data:text/javascript;base64,${Buffer.from(output).toString("base64")}`
  );
}

const cards = await realModule("../app/chat-cards.ts");
const input = await realModule("../app/chat-input.ts");
const autonomy = await realModule("../app/autonomy-status.ts");
const imageA = "data:image/png;base64,YQ==";
const imageB = "data:image/png;base64,Yg==";
const file = (name = "叶片A.png", change = {}) => ({
  name,
  type: "image/png",
  size: 20,
  ...change,
});

function allNodes(node, predicate) {
  if (Array.isArray(node))
    return node.flatMap((child) => allNodes(child, predicate));
  if (!node || typeof node !== "object") return [];
  return [
    ...(predicate(node) ? [node] : []),
    ...allNodes(node.props?.children, predicate),
  ];
}

function chatHarness() {
  const readers = [];
  const requests = [];
  class ControlledFileReader {
    result = null;
    readyState = 0;
    readAsDataURL(selected) {
      this.file = selected;
      this.readyState = 1;
    }
    abort() {
      this.readyState = 2;
      this.onabort?.();
    }
    complete(result) {
      this.result = result;
      this.readyState = 2;
      this.onload?.();
    }
    fail() {
      this.readyState = 2;
      this.onerror?.();
    }
    constructor() {
      readers.push(this);
    }
  }
  const harness = hookHarness(
    compiled,
    "ChatPage",
    {
      "../api": {
        async get(url) {
          if (url === "/api/profile") return { name: "隔离测试" };
          if (url === "/api/chat/sessions?limit=30")
            return [{ id: "history-1", title: "历史对话", message_count: 0 }];
          if (url.startsWith("/api/chat/sessions/")) return { messages: [] };
          return [];
        },
        post(url, body) {
          if (url !== "/api/chat")
            return Promise.resolve({ id: "test-session" });
          return new Promise((resolve, reject) =>
            requests.push({ body, resolve, reject }),
          );
        },
      },
      "../chat-cards": cards,
      "../chat-input": input,
      "../autonomy-status": autonomy,
      "../use-visible-polling": { useVisiblePolling() {} },
      "./Common": {
        Card: "Card",
        Empty: "Empty",
        ErrorState: "ErrorState",
        PageHeader: "PageHeader",
      },
      "./MarkdownContent": { MarkdownContent: "MarkdownContent" },
      "./ChatCards": { ChatCards: "ChatCards" },
    },
    { FileReader: ControlledFileReader },
  );
  const nodes = (predicate) => allNodes(harness.render(), predicate);
  return Object.assign(harness, {
    readers,
    requests,
    choose(selected) {
      const target = {
        files: [selected],
        value: `C:\\fakepath\\${selected.name}`,
      };
      harness.nodes("input")[0].props.onChange({ target });
      harness.render();
      return target;
    },
    type(text) {
      harness.nodes("textarea")[0].props.onChange({ target: { value: text } });
      harness.render();
    },
    submit() {
      const promise = harness
        .nodes("form")[0]
        .props.onSubmit({ preventDefault() {} });
      harness.render();
      return promise;
    },
    sendDisabled() {
      return nodes((node) => node.props?.["aria-label"] === "发送消息")[0].props
        .disabled;
    },
    attachmentName() {
      const chip = nodes((node) => node.props?.className === "image-chip")[0];
      return (
        chip &&
        allNodes(chip, (node) => node.type === "span")[0]?.props.children
      );
    },
    messageImages() {
      return nodes((node) => node.type === "img").map((node) => ({
        src: node.props.src,
        alt: node.props.alt,
      }));
    },
    cancelAttachment() {
      const chip = nodes((node) => node.props?.className === "image-chip")[0];
      assert.ok(chip, "读取中的图片也应允许取消");
      allNodes(chip, (node) => node.type === "button")[0].props.onClick();
      harness.render();
    },
    newSession() {
      harness.nodes("PageHeader")[0].props.actions.props.onClick();
      harness.render();
    },
    openHistory() {
      const button = nodes(
        (node) =>
          node.type === "button" &&
          allNodes(
            node,
            (child) =>
              child.type === "b" && child.props.children === "历史对话",
          ).length,
      )[0];
      const promise = button.props.onClick();
      harness.render();
      return promise;
    },
    errors() {
      return harness
        .nodes("ErrorState")
        .map((node) => node.props.message)
        .join(" ");
    },
  });
}

test("先选A后选B，即使A读取更晚完成也只发送B", async () => {
  const chat = chatHarness();
  chat.render();
  chat.choose(file());
  chat.choose(file("叶片B.png"));
  chat.readers[1].complete(imageB);
  chat.readers[0].complete(imageA);
  chat.readers[0].fail();
  await chat.settle();
  assert.equal(chat.attachmentName(), "叶片B.png");
  assert.equal(chat.errors(), "");
  chat.submit();
  assert.equal(chat.requests[0].body.image_data, "Yg==");
});

test("A请求等待期间选好的B不会被A成功响应清除", async () => {
  const chat = chatHarness();
  chat.render();
  chat.choose(file());
  chat.readers[0].complete(imageA);
  await chat.settle();
  const first = chat.submit();
  assert.equal(chat.requests[0].body.image_data, "YQ==");
  chat.choose(file("叶片B.png"));
  chat.readers[1].complete(imageB);
  await chat.settle();
  chat.requests[0].resolve({ final_answer: "A的识别结果" });
  await first;
  await chat.settle();
  assert.equal(chat.attachmentName(), "叶片B.png");
  assert.equal(chat.sendDisabled(), false);
  chat.submit();
  assert.equal(chat.requests[1].body.image_data, "Yg==");
});

test("发送后的图片预览保留原始src和文件名，替换下一张不改变旧消息", async () => {
  const chat = chatHarness();
  chat.render();
  chat.choose(file());
  chat.readers[0].complete(imageA);
  await chat.settle();
  const first = chat.submit();
  assert.deepEqual(chat.messageImages(), [
    { src: "data:image/png;base64,YQ==", alt: "已上传：叶片A.png" },
  ]);
  chat.choose(file("叶片B.png"));
  chat.readers[1].complete(imageB);
  await chat.settle();
  assert.deepEqual(chat.messageImages(), [
    { src: "data:image/png;base64,YQ==", alt: "已上传：叶片A.png" },
  ]);
  chat.requests[0].resolve({ final_answer: "A的识别结果" });
  await first;
  await chat.settle();
  chat.submit();
  assert.deepEqual(chat.messageImages(), [
    { src: "data:image/png;base64,YQ==", alt: "已上传：叶片A.png" },
    { src: "data:image/png;base64,Yg==", alt: "已上传：叶片B.png" },
  ]);
});

test("选新图读取时禁用发送，表单和回车都不能发送旧图", async () => {
  const chat = chatHarness();
  chat.render();
  chat.choose(file());
  chat.readers[0].complete(imageA);
  await chat.settle();
  chat.type("只返回识别结果");
  chat.choose(file("叶片B.png"));
  assert.equal(chat.sendDisabled(), true);
  await chat.submit();
  chat.nodes("textarea")[0].props.onKeyDown({
    nativeEvent: { key: "Enter" },
    preventDefault() {},
  });
  assert.equal(chat.requests.length, 0);
});

test("选图后清空原生文件输入，使取消后可以重选同一文件", async () => {
  const chat = chatHarness();
  chat.render();
  const selected = file();
  const target = chat.choose(selected);
  assert.equal(target.value, "");
  chat.readers[0].complete(imageA);
  await chat.settle();
  chat.cancelAttachment();
  assert.equal(chat.attachmentName(), undefined);
  chat.choose(selected);
  chat.readers[1].complete(imageA);
  await chat.settle();
  assert.equal(chat.attachmentName(), "叶片A.png");
});

test("取消读取后晚到结果不会重新挂上附件", async () => {
  const chat = chatHarness();
  chat.render();
  chat.choose(file());
  chat.cancelAttachment();
  chat.readers[0].complete(imageA);
  await chat.settle();
  assert.equal(chat.attachmentName(), undefined);
  assert.equal(chat.sendDisabled(), true);
  assert.equal(chat.errors(), "");
});

for (const [event, description] of [
  ["fail", "失败"],
  ["abort", "中断"],
]) {
  test(`读图${description}显示错误并且不能提交先前附件`, async () => {
    const chat = chatHarness();
    chat.render();
    chat.choose(file());
    chat.readers[0].complete(imageA);
    await chat.settle();
    chat.choose(file("叶片B.png"));
    chat.readers[1][event]();
    await chat.settle();
    assert.match(chat.errors(), /图片.*(失败|中断)/);
    assert.equal(chat.attachmentName(), undefined);
    assert.equal(chat.sendDisabled(), true);
    await chat.submit();
    assert.equal(chat.requests.length, 0);
  });
}

for (const [description, selected, result] of [
  ["非图片", file("说明.txt", { type: "text/plain" }), undefined],
  ["空文件", file("空图.png", { size: 0 }), undefined],
  ["无图片数据", file(), "data:image/png;base64,"],
  ["错误读取结果", file(), "data:text/plain;base64,YQ=="],
]) {
  test(`${description}提供可见提示且不会发送空图或旧图`, async () => {
    const chat = chatHarness();
    chat.render();
    chat.choose(selected);
    if (result !== undefined) chat.readers[0].complete(result);
    await chat.settle();
    assert.match(chat.errors(), /图片|文件/);
    assert.equal(chat.attachmentName(), undefined);
    assert.equal(chat.sendDisabled(), true);
    await chat.submit();
    assert.equal(chat.requests.length, 0);
  });
}

test("切换新会话会清除读图状态，旧读取不能回填", async () => {
  const chat = chatHarness();
  chat.render();
  chat.choose(file());
  chat.newSession();
  chat.readers[0].complete(imageA);
  await chat.settle();
  assert.equal(chat.attachmentName(), undefined);
  chat.type("新会话的问题");
  assert.equal(chat.sendDisabled(), false);
  chat.submit();
  assert.equal(chat.requests[0].body.image_data, undefined);
});

test("打开历史会话后不残留读取状态或旧附件", async () => {
  const chat = chatHarness();
  chat.render();
  await chat.settle();
  chat.choose(file());
  await chat.openHistory();
  chat.readers[0].complete(imageA);
  await chat.settle();
  assert.equal(chat.attachmentName(), undefined);
  chat.type("历史会话的新问题");
  assert.equal(chat.sendDisabled(), false);
  chat.submit();
  assert.equal(chat.requests[0].body.image_data, undefined);
});

test("请求失败时保留原图供重试，不需要再次选择", async () => {
  const chat = chatHarness();
  chat.render();
  chat.choose(file());
  chat.readers[0].complete(imageA);
  await chat.settle();
  const first = chat.submit();
  chat.requests[0].reject(new Error("识别服务暂不可用"));
  await first;
  await chat.settle();
  assert.equal(chat.attachmentName(), "叶片A.png");
  assert.equal(chat.sendDisabled(), false);
  chat.submit();
  assert.equal(chat.requests[1].body.image_data, "YQ==");
});

test("页面卸载后晚到读取和错误均不能写入状态", async () => {
  const chat = chatHarness();
  chat.render();
  chat.choose(file());
  await chat.settle();
  chat.unmount();
  chat.readers[0].complete(imageA);
  chat.readers[0].fail();
  assert.equal(chat.writesAfterUnmount, 0);
});
