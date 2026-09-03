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
npm install
npm run dev
```

前端默认访问 `http://localhost:18001`，也可在登录页的“后端连接设置”中填写已部署的 HTTPS API 地址。

## 验证

```powershell
npm test
npm run lint
```

生产环境只需向前端提供公开的 FastAPI 地址。数据库密码、LLM 密钥和硬件连接凭据必须保留在后端，不能写入前端环境变量或浏览器存储。
