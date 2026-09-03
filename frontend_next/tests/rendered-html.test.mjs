import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("服务端渲染青禾智能农场登录入口", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>青禾智能农场<\/title>/);
  assert.match(html, /青禾智能农场/);
  assert.match(html, /欢迎回来/);
  assert.match(html, /进入农场/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
  assert.doesNotMatch(html, /codex-preview/);
});

test("完整保留原前端业务入口并移除临时骨架", async () => {
  const [page, layout, app, data, styles, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/AgricultureApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/data.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  const labels = [
    "农场概览",
    "智能对话",
    "基本信息",
    "地块管理",
    "财务管理",
    "农事日历",
    "政策补贴",
    "作物百科",
    "农资计算",
    "种植向导",
    "设备中心",
    "规则管理",
    "文档中心",
  ];
  for (const label of labels) assert.match(data, new RegExp(label));

  const pageComponents = [
    "DashboardPage",
    "ChatPage",
    "ProfilePage",
    "FieldsPage",
    "FinancePage",
    "CalendarPage",
    "PolicyPage",
    "EncyclopediaPage",
    "CalculatorPage",
    "WizardPage",
    "DevicesPage",
    "RulesPage",
    "DocsPage",
  ];
  for (const component of pageComponents)
    assert.match(app, new RegExp(`<${component}`));

  assert.match(page, /<AgricultureApp\s*\/>/);
  assert.match(layout, /lang="zh-CN"/);
  assert.match(styles, /@media \(max-width: 760px\)/);
  assert.match(styles, /\.nav-collapsed \.side-nav \.nav-item span/);
  assert.match(styles, /\.nav-collapsed \.side-nav[\s\S]*?display:\s*none/);
  assert.doesNotMatch(styles, /\.nav-collapsed \.nav-item span/);
  assert.match(packageJson, /"name": "qinghe-smart-farm"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(
    access(
      new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url),
    ),
  );
});

test("关键交互均通过 FastAPI 契约连接", async () => {
  const sources = await Promise.all(
    [
      "Auth.tsx",
      "Dashboard.tsx",
      "Chat.tsx",
      "Records.tsx",
      "Knowledge.tsx",
      "Automation.tsx",
    ].map((name) =>
      readFile(new URL(`../app/components/${name}`, import.meta.url), "utf8"),
    ),
  );
  const source = sources.join("\n");
  const contracts = [
    "/api/auth/",
    "/api/dashboard",
    "/api/chat",
    "/api/profile",
    "/api/fields",
    "/api/finance/",
    "/api/tasks",
    "/api/progress",
    "/api/policy/search",
    "/api/encyclopedia",
    "/api/plan",
    "/api/devices",
    "/api/actions/",
    "/api/rules",
  ];
  for (const contract of contracts)
    assert.ok(source.includes(contract), `缺少接口契约：${contract}`);
});

test("本地后端使用独立端口并迁移旧地址", async () => {
  const [apiSource, authSource] = await Promise.all([
    readFile(new URL("../app/api.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/components/Auth.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(apiSource, /http:\/\/localhost:18001/);
  assert.match(apiSource, /LEGACY_LOCAL_API_BASES/);
  assert.match(apiSource, /http:\/\/localhost:8000/);
  assert.match(apiSource, /无法连接农业后端/);
  assert.match(apiSource, /url\.origin/);
  assert.match(authSource, /http:\/\/localhost:18001/);
});

test("设备配置支持地块内作业分区", async () => {
  const [automation, types] = await Promise.all([
    readFile(new URL("../app/components/Automation.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/types.ts", import.meta.url), "utf8"),
  ]);
  assert.match(automation, /zone_id/);
  assert.match(automation, /作业分区 ID/);
  assert.match(types, /zone_id\?: string/);
});

test("设备弹窗支持键盘关闭且提交期间防止重复保存", async () => {
  const automation = await readFile(
    new URL("../app/components/Automation.tsx", import.meta.url),
    "utf8",
  );
  assert.match(automation, /function useEscapeClose/);
  assert.match(automation, /event\.key === "Escape"/);
  assert.match(automation, /role="dialog"/);
  assert.match(automation, /if \(saving\) return/);
  assert.match(automation, /disabled=\{saving\}/);
});

test("规则表单为空或未绑定设备时给出明确校验", async () => {
  const automation = await readFile(
    new URL("../app/components/Automation.tsx", import.meta.url),
    "utf8",
  );
  assert.match(automation, /请输入规则名称/);
  assert.match(automation, /请先在设备中心注册一个设备/);
  assert.match(automation, /placeholder="小麦自动灌溉"[\s\S]*?required/);
});

test("Leaflet 默认资源使用本地静态路径", async () => {
  const [styles, leafletStyles] = await Promise.all([
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/leaflet.css", import.meta.url), "utf8"),
  ]);
  assert.match(styles, /@import\s+"\.\/leaflet\.css"/);
  assert.match(leafletStyles, /url\("\/images\/layers\.png"\)/);
  assert.match(leafletStyles, /url\("\/images\/marker-icon\.png"\)/);
  await access(new URL("../public/images/layers.png", import.meta.url));
  await access(new URL("../public/images/marker-icon.png", import.meta.url));
});

test("经营趋势只使用真实财务汇总且无数据时保持为空", async () => {
  const dashboard = await readFile(
    new URL("../app/components/Dashboard.tsx", import.meta.url),
    "utf8",
  );
  assert.match(dashboard, /\/api\/finance\/summary/);
  assert.match(dashboard, /暂无经营趋势数据/);
  assert.match(dashboard, /真实收入走势/);
  assert.doesNotMatch(dashboard, /42,\s*58,\s*47,\s*72,\s*66/);
  assert.doesNotMatch(dashboard, /Math\.max\(16,\s*finance\.month_income/);
});

test("经营概况卡片具有清晰的财务信息层级", async () => {
  const [dashboard, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Dashboard.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  for (const className of [
    "hero-finance-head",
    "hero-finance-body",
    "hero-profit",
    "hero-finance-footer",
    "hero-finance-button",
  ]) {
    assert.match(dashboard, new RegExp(`className="${className}"`));
    assert.match(styles, new RegExp(`\\.${className}`));
  }
  assert.match(dashboard, /本月净利润/);
  const hero = dashboard.slice(
    dashboard.indexOf('className="farm-hero"'),
    dashboard.indexOf('className="metric-stack"'),
  );
  assert.match(hero, /className="hero-breakdown"/);
  assert.match(hero, /本月收入/);
  assert.match(hero, /本月成本/);
  const lowerMetrics = dashboard.slice(
    dashboard.indexOf('<div className="dashboard-workspace-grid">'),
    dashboard.indexOf('className="content-grid equal"'),
  );
  assert.doesNotMatch(lowerMetrics, /本月收入/);
  assert.doesNotMatch(lowerMetrics, /本月成本/);
  assert.match(lowerMetrics, /风险提醒/);
});

test("概览风险提醒使用紧凑卡片并保留真实风险计数", async () => {
  const [dashboard, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Dashboard.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(dashboard, /data\.weather_persistence\?\.alerts/);
  assert.match(dashboard, /data\.disease_risks/);
  assert.match(dashboard, /className="dashboard-workspace-grid"/);
  assert.match(dashboard, /className="risk-summary-card"/);
  assert.match(
    styles,
    /\.dashboard-workspace-grid\s*\{[\s\S]*?minmax\(0, 1\.55fr\) minmax\(290px, 0\.75fr\)/,
  );
  assert.match(
    dashboard,
    /className="dashboard-workspace-column dashboard-secondary-column"[\s\S]*?className="risk-summary-card"[\s\S]*?className="dashboard-task-card"/,
  );
  assert.match(
    styles,
    /\.dashboard-workspace-column\s*\{[\s\S]*?align-content:\s*start;[\s\S]*?gap:\s*16px/,
  );
  assert.match(styles, /\.risk-summary-card\s*\{[\s\S]*?width:\s*100%/);
  assert.match(styles, /\.risk-summary-card\s*\{[\s\S]*?min-height:\s*96px/);
});

test("风险提醒左侧仅保留硬件设备状态，避免重复种植进度", async () => {
  const [dashboard, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Dashboard.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(dashboard, /get<Device\[]>\("\/api\/devices"\)/);
  assert.match(dashboard, /get<JsonMap\[]>\("\/api\/actions\/log\?limit=12"\)/);
  assert.match(dashboard, /get<JsonMap\[]>\("\/api\/actions\/pending"\)/);
  assert.match(dashboard, /className="dashboard-status-strip"/);
  assert.match(dashboard, /硬件设备状态/);
  assert.match(dashboard, /onlineDevices\.length/);
  assert.match(dashboard, /当前设备工作正常/);
  assert.match(dashboard, /异常 \/ 离线/);
  assert.match(dashboard, /当前动作/);
  assert.match(dashboard, /最近动作/);
  assert.match(dashboard, /deviceActionLabel/);
  assert.match(dashboard, /onNavigate\("devices"\)/);
  const statusStrip = dashboard.slice(
    dashboard.indexOf('<div className="dashboard-status-strip">'),
    dashboard.indexOf('className="dashboard-progress-card"'),
  );
  assert.doesNotMatch(statusStrip, /种植进度/);
  assert.match(styles, /\.dashboard-status-strip\s*\{[\s\S]*?min-width:\s*0/);
  assert.match(
    styles,
    /\.device-overview-card,[\s\S]*?\.dashboard-progress-card\s*\{[\s\S]*?min-height:\s*360px/,
  );
  assert.match(styles, /\.device-overview-metrics\s*\{/);
  assert.match(styles, /\.device-action-summary\s*\{/);
});

test("财务统计标题放大且所有流水金额右对齐", async () => {
  const [records, styles] = await Promise.all([
    readFile(new URL("../app/components/Records.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  for (const label of ["累计收入", "累计成本", "净收益"]) {
    assert.match(records, new RegExp(label));
  }
  assert.match(records, /className="finance-record-row"/);
  assert.match(records, /className=\{`record-amount/);
  assert.match(styles, /\.finance-metric small\s*\{[\s\S]*?font-size:\s*15px/);
  assert.match(
    styles,
    /\.finance-metric strong\s*\{[\s\S]*?text-align:\s*right/,
  );
  assert.match(styles, /\.record-amount\s*\{[\s\S]*?text-align:\s*right/);
});

test("智能对话的图片与语音按钮使用统一尺寸", async () => {
  const [chat, styles] = await Promise.all([
    readFile(new URL("../app/components/Chat.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(chat, /className="composer-tools"/);
  assert.match(chat, /className="tool-button"[\s\S]*?aria-label="选择图片"/);
  assert.match(chat, /className="tool-button"[\s\S]*?title="语音输入"/);
  assert.match(styles, /\.tool-button\s*\{[\s\S]*?width:\s*36px/);
  assert.match(styles, /\.tool-button input\[type="file"\]/);
  assert.match(styles, /\.composer-tools\s*\{[\s\S]*?align-items:\s*center/);
});

test("智能对话欢迎语背景会随文字宽度自适应", async () => {
  const [chat, styles] = await Promise.all([
    readFile(new URL("../app/components/Chat.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(chat, /className="welcome-kicker">你好，我是青禾/);
  assert.match(styles, /\.welcome-kicker\s*\{[\s\S]*?width:\s*fit-content/);
  assert.match(styles, /\.welcome-kicker\s*\{[\s\S]*?max-width:\s*100%/);
  assert.match(
    styles,
    /\.welcome-kicker\s*\{[\s\S]*?overflow-wrap:\s*anywhere/,
  );
  assert.doesNotMatch(styles, /\.chat-welcome\s*>\s*span\s*\{/);
});

test("智能对话顶部身份、状态与清空按钮分区对齐", async () => {
  const [chat, styles] = await Promise.all([
    readFile(new URL("../app/components/Chat.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(chat, /className="chat-assistant-identity"/);
  assert.match(chat, /className="chat-assistant-icon"/);
  assert.match(chat, /className="chat-assistant-copy"/);
  assert.match(chat, /className="chat-clear-button"/);
  assert.match(chat, /<span>清空对话<\/span>/);
  assert.match(
    styles,
    /\.chat-top\s*\{[\s\S]*?grid-template-columns:\s*minmax\(0, 1fr\) auto/,
  );
  assert.match(styles, /\.chat-clear-button\s*\{/);
  assert.doesNotMatch(styles, /\.chat-top > div:nth-child\(2\)/);
});

test("基本信息使用背景色表达目标与自主权选中状态", async () => {
  const [records, styles] = await Promise.all([
    readFile(new URL("../app/components/Records.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(
    records,
    /aria-pressed=\{profile\.user_goals\.includes\(goal\)\}/,
  );
  assert.match(records, /className="autonomy-current"/);
  assert.match(records, /aria-pressed=\{autonomy === item\.id\}/);
  assert.doesNotMatch(
    records,
    /profile\.user_goals\.includes\(goal\) && <Check/,
  );
  assert.match(styles, /\.choice-group button\.selected[\s\S]*?background:/);
  assert.match(
    styles,
    /\.autonomy-options button\.selected[\s\S]*?background:/,
  );
});

test("顶部搜索栏直接执行作物与政策搜索", async () => {
  const app = await readFile(
    new URL("../app/AgricultureApp.tsx", import.meta.url),
    "utf8",
  );
  assert.match(app, /className="topbar-search" onSubmit=\{runGlobalSearch\}/);
  assert.match(app, /\/api\/encyclopedia/);
  assert.match(app, /\/api\/policy\/search\?q=/);
  assert.match(app, /className="global-search-panel"/);
  assert.doesNotMatch(
    app,
    /className="search-button"[\s\S]*?navigate\("encyclopedia"\)/,
  );
});

test("地块管理以大地图为主并隔离地图标记与列表卡片", async () => {
  const [records, fieldMap, styles] = await Promise.all([
    readFile(new URL("../app/components/Records.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/components/FieldMap.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(records, /<FieldOverviewMap/);
  assert.match(records, /className="selected-field-metric"/);
  assert.match(
    fieldMap,
    /tileLayer\("https:\/\/\{s\}\.tile\.openstreetmap\.org/,
  );
  assert.match(fieldMap, /L\.polygon\(coordinates/);
  assert.match(fieldMap, /fitBounds\(bounds/);
  assert.match(styles, /\.field-layout\s*\{[\s\S]*?minmax\(0, 1fr\) 320px/);
  assert.match(styles, /\.farm-map\s*\{[\s\S]*?min-height:\s*520px/);
  assert.match(styles, /\.leaflet-field-map\s*\{[\s\S]*?position:\s*absolute/);
  assert.match(styles, /\.field-index\s*\{[\s\S]*?display:\s*grid/);
});

test("创建地块使用真实地图绘制并支持本机定位", async () => {
  const [records, createMap, styles] = await Promise.all([
    readFile(new URL("../app/components/Records.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/components/FieldCreateMap.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(
    records,
    /<FieldCreateMap points=\{points\} setPoints=\{setPoints\}/,
  );
  assert.doesNotMatch(records, /<canvas/);
  assert.match(
    createMap,
    /tileLayer\("https:\/\/\{s\}\.tile\.openstreetmap\.org/,
  );
  assert.match(createMap, /instance\.on\("click"/);
  assert.match(createMap, /navigator\.geolocation\.getCurrentPosition/);
  assert.match(createMap, /instance\.setView\(current, 17/);
  assert.match(createMap, /定位到本机/);
  assert.match(styles, /\.field-create-map\s*\{[\s\S]*?height:\s*360px/);
});

test("设备卡片统一信息层级、状态与操作对齐", async () => {
  const [automation, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Automation.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(automation, /function DeviceCard\(/);
  assert.match(automation, /className="device-status"/);
  assert.match(automation, /deviceStatusLabel\(device\.status\)/);
  assert.match(automation, /className="device-details"/);
  assert.match(automation, /className="sensor-empty"/);
  assert.match(automation, /function deviceParameterEntries\(/);
  assert.match(automation, /formatDeviceValue\(key, value\)/);
  assert.match(automation, /function DeviceRunControls\(/);
  assert.match(automation, /key:\s*"flow_rate"/);
  assert.match(automation, /key:\s*"duration"/);
  assert.match(automation, /onCommand\("start", params\)/);
  assert.match(automation, /function DeviceEditor\(/);
  assert.match(
    automation,
    /post<JsonMap>\([\s\S]*?`\/api\/devices\/\$\{device\.device_id\}\/config`/,
  );
  assert.match(automation, /后端尚未加载设备配置接口，请重启 FastAPI 后再保存/);
  const abilitySections = automation.match(
    /<span>设备能力<\/span>[\s\S]*?<\/div>\s*<\/div>/g,
  );
  assert.equal(abilitySections?.length, 2);
  abilitySections?.forEach((section) =>
    assert.doesNotMatch(section, /<Check\s*\/>/),
  );
  assert.match(automation, /保存并重新连接/);
  assert.match(automation, /<span>删除<\/span>/);
  assert.match(
    styles,
    /\.device-card\s*\{[\s\S]*?grid-template-rows:\s*auto auto auto auto auto/,
  );
  assert.match(
    styles,
    /\.device-status \.pill\s*\{[\s\S]*?justify-content:\s*center/,
  );
  assert.match(
    styles,
    /\.sensor-grid\s*\{[\s\S]*?repeat\(2, minmax\(0, 1fr\)\)[\s\S]*?min-height:\s*136px/,
  );
  assert.match(styles, /\.sensor-grid\.compact\s*\{[\s\S]*?min-height:\s*64px/);
  assert.match(styles, /\.device-run-panel\s*\{/);
  assert.match(styles, /\.device-edit-modal\s*\{/);
  assert.match(
    styles,
    /\.device-actions\s*\{[\s\S]*?repeat\(3, minmax\(0, 1fr\)\)[\s\S]*?min-height:\s*83px/,
  );
  assert.match(
    styles,
    /\.device-actions button\s*\{[\s\S]*?align-items:\s*center[\s\S]*?justify-content:\s*center/,
  );
});

test("设备统计卡可打开详情且待确认动作支持弹窗处理", async () => {
  const [automation, pendingEditor, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Automation.tsx", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL("../app/components/PendingActionEditor.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  for (const kind of ["all", "online", "pending", "today"]) {
    assert.match(automation, new RegExp(`setSummary\\("${kind}"\\)`));
  }
  assert.match(automation, /function DeviceSummaryModal\(/);
  assert.match(automation, /aria-modal="true"/);
  assert.match(automation, /onDecide\(actionId, true\)/);
  assert.match(automation, /onDecide\(actionId, false\)/);
  assert.match(automation, /<PendingActionEditor/);
  assert.match(pendingEditor, /function PendingActionEditor\(/);
  assert.match(pendingEditor, /put\(`\/api\/actions\/\$\{action\.id\}`/);
  assert.match(pendingEditor, /保存不会立即执行/);
  assert.match(automation, /item\.status === "failed" \? "重试执行"/);
  assert.match(automation, /setBusy\(`action:\$\{id\}`\)/);
  assert.match(
    styles,
    /\.device-metric\s*\{[\s\S]*?grid-template-columns:\s*44px minmax\(0, 1fr\) 18px/,
  );
  assert.match(styles, /\.device-summary-modal\s*\{/);
  assert.match(styles, /\.summary-action-buttons\s*\{/);
  assert.match(styles, /\.pending-action-editor textarea\s*\{/);
});

test("摄像头设备卡片展示定时刷新的当前画面", async () => {
  const [automation, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Automation.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(automation, /function CameraLiveView\(/);
  assert.match(automation, /isCamera && <CameraLiveView/);
  assert.match(
    automation,
    /`\/api\/devices\/\$\{device\.device_id\}\/snapshot`/,
  );
  assert.match(automation, /setTimeout\(poll, 5000\)/);
  assert.match(automation, /className="camera-live-frame"/);
  assert.match(automation, /当前画面/);
  assert.match(
    styles,
    /\.camera-live-frame\s*\{[\s\S]*?aspect-ratio:\s*16 \/ 9/,
  );
  assert.match(
    styles,
    /\.camera-live-frame img\s*\{[\s\S]*?object-fit:\s*cover/,
  );
});

test("农事日历使用独立日期面板与结构化进度时间线", async () => {
  const records = await readFile(
    new URL("../app/components/Records.tsx", import.meta.url),
    "utf8",
  );
  const calendar = records.slice(
    records.indexOf("export function CalendarPage"),
  );
  for (const className of [
    "calendar-workspace",
    "calendar-board",
    "growth-timeline-card",
    "calendar-plan-timeline",
    "stage-progress",
    "calendar-task-actions",
  ]) {
    assert.match(calendar, new RegExp(`className="${className}"`));
  }
  assert.doesNotMatch(calendar, /className="timeline-track"/);
  assert.doesNotMatch(calendar, /left:\s*`\$\{/);
  assert.match(calendar, /post<JsonMap>\("\/api\/tasks"/);
  assert.match(calendar, /if \(saving\) return/);
  assert.match(calendar, /formError && <ErrorState message=\{formError\}/);
  assert.match(calendar, /saving \? "正在保存" : "保存"/);
});

test("作物百科按真实知识库字段生成标签与中文对比", async () => {
  const [knowledge, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Knowledge.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  for (const field of [
    "planting_seasons",
    "growth_stages",
    "soil_requirements",
    "yield_info",
    "fertilization_guide",
    "common_diseases",
  ]) {
    assert.match(knowledge, new RegExp(field));
  }
  assert.doesNotMatch(knowledge, /暂无结构化资料/);
  assert.match(
    styles,
    /\.crop-directory\s*\{[\s\S]*?grid-auto-rows:\s*min-content/,
  );
  assert.match(styles, /\.compare-table\s*>\s*div/);
});

test("农资计算全部直列且每项都有独立计算按钮", async () => {
  const knowledge = await readFile(
    new URL("../app/components/Knowledge.tsx", import.meta.url),
    "utf8",
  );
  assert.match(knowledge, /className="calculator-stack"/);
  for (const label of ["计算播种量", "计算肥料用量", "计算稀释结果"]) {
    assert.match(knowledge, new RegExp(label));
  }
  assert.doesNotMatch(knowledge, /setTab\(/);
  assert.doesNotMatch(knowledge, /className="calculator-tabs"/);
  assert.equal((knowledge.match(/noValidate/g) || []).length, 3);
  assert.ok((knowledge.match(/step="any"/g) || []).length >= 8);
  assert.match(knowledge, /safeCalculationValue/);
  assert.match(knowledge, /输入框中的原值未被修改/);
});

test("种植向导支持自定义作物并采用紧凑背景选中布局", async () => {
  const [knowledge, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Knowledge.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(knowledge, /list="wizard-crop-options"/);
  assert.match(knowledge, /也可直接输入其他作物/);
  assert.match(knowledge, /className="wizard-form-panel"/);
  assert.match(knowledge, /生成种植报告/);
  assert.doesNotMatch(knowledge, /selectedGoals\.includes\(goal\) && <Check/);
  assert.match(styles, /\.crop-picker button\.selected[\s\S]*?background:/);
  assert.match(
    styles,
    /\.wizard-submit\s*\{[\s\S]*?justify-content:\s*space-between/,
  );
});

test("标题栏提醒按任务、天气与硬件真实数据分组", async () => {
  const [app, styles] = await Promise.all([
    readFile(new URL("../app/AgricultureApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  for (const endpoint of [
    "/api/tasks",
    "/api/dashboard",
    "/api/devices",
    "/api/actions/pending",
  ]) {
    assert.match(app, new RegExp(endpoint.replaceAll("/", "\\/")));
  }
  for (const label of ["当前任务", "天气与风险", "硬件通知"]) {
    assert.match(app, new RegExp(label));
  }
  assert.match(app, /className="notification-groups"/);
  assert.match(styles, /\.notification-groups\s*\{/);
  assert.match(styles, /\.notification-item\.urgent/);
});

test("政策检索会立即执行热门词并明确展示真实来源", async () => {
  const knowledge = await readFile(
    new URL("../app/components/Knowledge.tsx", import.meta.url),
    "utf8",
  );
  assert.match(knowledge, /runPolicySearch\(item\)/);
  assert.match(knowledge, /searchError \? \(/);
  assert.match(knowledge, /中国政府网|官方来源/);
  assert.doesNotMatch(knowledge, /item\.score \|\| item\.relevance \|\| 0\.86/);
});

test("规则页独立加载设备并在删除后同步本地与后端状态", async () => {
  const automation = await readFile(
    new URL("../app/components/Automation.tsx", import.meta.url),
    "utf8",
  );
  assert.match(automation, /get<Device\[]>\("\/api\/devices"\)/);
  assert.match(automation, /devicesLoading \? \(/);
  assert.match(automation, /remove<JsonMap>\(`\/api\/rules\/\$\{rule\.id\}`\)/);
  assert.match(automation, /current\.filter\(/);
  assert.match(automation, /setMessage\("规则已彻底删除"\)/);
  assert.match(automation, /await load\(\)/);
});

test("农资计算结果使用放大数值与分行对齐布局", async () => {
  const [knowledge, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Knowledge.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(knowledge, /className="calculation-detail-row"/);
  assert.match(
    styles,
    /\.calculation-result strong\s*\{[\s\S]*?font-size:\s*clamp\(30px, 3vw, 40px\)/,
  );
  assert.match(styles, /\.calculation-detail-row\s*\{/);
  assert.match(
    styles,
    /\.calculation-detail-row b\s*\{[\s\S]*?text-align:\s*right/,
  );
});

test("页面恢复 Geist 字体且不依赖 vinext 本机字体缓存", async () => {
  const [layout, styles] = await Promise.all([
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.doesNotMatch(layout, /next\/font|Geist_Mono|Geist\(/);
  assert.doesNotMatch(styles, /--font-geist/);
  assert.match(styles, /@font-face[\s\S]*?font-family:\s*"Geist"/);
  assert.match(styles, /url\("\/fonts\/geist-latin\.woff2"\)/);
  assert.match(styles, /font-family:[\s\S]*?"Geist", "Microsoft YaHei"/);
  assert.match(styles, /"Microsoft YaHei"/);
});

test("移动端收起导航会关闭抽屉而非切换桌面宽度", async () => {
  const app = await readFile(
    new URL("../app/AgricultureApp.tsx", import.meta.url),
    "utf8",
  );
  assert.match(app, /function collapseNavigation\(\)/);
  assert.match(app, /window\.matchMedia\("\(max-width: 760px\)"\)\.matches/);
  assert.match(app, /setMobileOpen\(false\)/);
  assert.match(
    app,
    /className="collapse-button"[\s\S]*?onClick=\{collapseNavigation\}/,
  );
});

test("文档中心包含各驱动协议的连接参数和切换步骤", async () => {
  const [automation, styles] = await Promise.all([
    readFile(
      new URL("../app/components/Automation.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  for (const label of [
    "更换驱动协议与连接参数",
    "Modbus TCP",
    "Modbus RTU",
    "CoAP",
    "OPC UA",
    "USB 摄像头",
    "IP / ESP32 摄像头",
    "保存并重新连接",
  ]) {
    assert.match(automation, new RegExp(label));
  }
  assert.match(automation, /JSON\.stringify\(example\.config, null, 2\)/);
  assert.match(styles, /\.connection-example-grid\s*\{/);
  assert.match(styles, /\.protocol-config-steps\s*\{/);
});

test("模型输出使用统一安全 Markdown 渲染", async () => {
  const [chat, knowledge, markdown, styles] = await Promise.all([
    readFile(new URL("../app/components/Chat.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/components/Knowledge.tsx", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL("../app/components/MarkdownContent.tsx", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(chat, /<MarkdownContent content=\{message\.content\}/);
  assert.match(knowledge, /<MarkdownContent[\s\S]*?result\.plan_text/);
  assert.match(markdown, /ReactMarkdown/);
  assert.match(markdown, /remarkGfm/);
  assert.match(styles, /\.markdown-content table\s*\{/);
});

test("图片发送后会在当前消息中显示附件预览", async () => {
  const chat = await readFile(
    new URL("../app/components/Chat.tsx", import.meta.url),
    "utf8",
  );
  assert.match(chat, /image:\s*image/);
  assert.match(chat, /className="message-image"/);
  assert.match(chat, /image_data:\s*image\?\.data/);
});

test("设备注册可直接填写连接 JSON 且规则设置参数有名称和值", async () => {
  const automation = await readFile(
    new URL("../app/components/Automation.tsx", import.meta.url),
    "utf8",
  );
  const creator = automation.slice(
    automation.indexOf("function DeviceCreator("),
    automation.indexOf("function DeviceEditor("),
  );
  assert.match(automation, /可在注册时直接填写/);
  assert.doesNotMatch(creator, /form\.driver !== "simulator"/);
  assert.match(automation, /form\.command === "set_param"/);
  assert.match(automation, /参数名[\s\S]*?form\.parameterName/);
  assert.match(automation, /参数值[\s\S]*?form\.parameterValue/);
  assert.match(automation, /params:\s*actionParams/);
});
