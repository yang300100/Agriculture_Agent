# 青禾智能农场新前端

这是 `Agriculture_Agent` 的独立 React / TypeScript 前端，不替换、不依赖原有 Streamlit 页面文件。业务数据统一通过 FastAPI 接口交换。

## 功能范围

- 用户登录、注册与后端地址配置
- 农场概览、智能对话、种植档案与地块管理
- 财务、农事日历、政策、作物百科与农资计算
- 种植向导、设备中心、自动规则和文档中心
- 桌面、平板和手机响应式布局

## 本地启动

要求 Node.js `>=22.13.0`。

```powershell
npm ci
npm run dev
```

前端默认访问 `http://localhost:18001`，也可在登录页的“后端连接设置”中填写已部署的 HTTPS API 地址。

## 独立构建配置

项目使用仓库内的 `vite.config.ts` 和 `wrangler.jsonc` 构建，不需要 `.openai/hosting.json` 或 Sites 专用插件。本地开发、构建和测试不需要 Cloudflare 账号或云端密钥。

`wrangler.jsonc` 配置 Worker 入口、静态资源和图片处理绑定。业务数据由 FastAPI 管理，默认不绑定 D1 数据库或 R2 存储。若启用 `examples/d1` 示例，需要在 `wrangler.jsonc` 的 `d1_databases` 中添加名为 `DB` 的真实绑定。

```powershell
npm run build
npm start
```

构建结果写入 `dist/client` 和 `dist/server`。`npm start` 使用 Cloudflare Workers 本地运行时预览构建结果，并提供字体、地图图片等静态资源。正式部署时另行配置自己的 Cloudflare 账号与所需资源；不要使用占位数据库 ID。

## 验证

```powershell
npm test
npm run lint
```

生产环境只需向前端提供公开的 FastAPI 地址。数据库密码、LLM 密钥和硬件连接凭据必须保留在后端，不能写入前端环境变量或浏览器存储。
