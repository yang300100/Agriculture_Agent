"use client";

import { FormEvent, useState } from "react";
import {
  ArrowRight,
  Leaf,
  LockKeyhole,
  ServerCog,
  ShieldCheck,
} from "lucide-react";
import { getApiBase, post, saveIdentity, setApiBase } from "../api";

export function AuthScreen({
  onAuthenticated,
}: {
  onAuthenticated: (username: string) => void;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [apiBase, setBase] = useState(getApiBase());
  const [showServer, setShowServer] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    if (!username.trim() || !password) return setError("请输入用户名和密码");
    if (mode === "register" && password !== confirm)
      return setError("两次输入的密码不一致");
    setLoading(true);
    try {
      setApiBase(apiBase);
      const result = await post<{
        success: boolean;
        username: string;
        token: string;
      }>(`/api/auth/${mode}`, { username: username.trim(), password }, true);
      saveIdentity(result.username, result.token || "");
      onAuthenticated(result.username);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "连接后端失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-story">
        <div className="brand-lockup">
          <span>
            <Leaf size={21} />
          </span>
          <b>青禾智能农场</b>
        </div>
        <div className="auth-copy">
          <span className="auth-kicker">AGRICULTURE INTELLIGENCE</span>
          <h1>
            把每一块土地，
            <br />
            照料得更明白。
          </h1>
          <p>
            从种植计划到设备联动，从天气风险到经营账目，一套真正懂农业现场的智能工作台。
          </p>
        </div>
        <div className="auth-proof">
          <div>
            <ShieldCheck />
            <span>
              <b>数据隔离</b>
              <small>每位农场主独立存储</small>
            </span>
          </div>
          <div>
            <LockKeyhole />
            <span>
              <b>安全控制</b>
              <small>硬件操作保留确认边界</small>
            </span>
          </div>
        </div>
        <div className="field-lines" aria-hidden="true">
          <i />
          <i />
          <i />
          <i />
          <i />
        </div>
      </section>
      <section className="auth-panel">
        <div className="auth-card">
          <div className="mobile-brand">
            <Leaf size={19} />
            青禾智能农场
          </div>
          <h2>{mode === "login" ? "欢迎回来" : "创建农场账户"}</h2>
          <p>
            {mode === "login"
              ? "登录后继续管理今天的农事。"
              : "几秒钟建立你的数字农场。"}
          </p>
          <div className="segmented auth-tabs">
            <button
              className={mode === "login" ? "active" : ""}
              onClick={() => setMode("login")}
            >
              登录
            </button>
            <button
              className={mode === "register" ? "active" : ""}
              onClick={() => setMode("register")}
            >
              注册
            </button>
          </div>
          <form onSubmit={submit}>
            <label>
              用户名
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="请输入用户名"
                autoComplete="username"
              />
            </label>
            <label>
              密码
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="请输入密码"
                autoComplete={
                  mode === "login" ? "current-password" : "new-password"
                }
              />
            </label>
            {mode === "register" && (
              <label>
                确认密码
                <input
                  type="password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="再次输入密码"
                  autoComplete="new-password"
                />
              </label>
            )}
            {error && <div className="form-error">{error}</div>}
            <button className="primary-button auth-submit" disabled={loading}>
              {loading
                ? "正在连接…"
                : mode === "login"
                  ? "进入农场"
                  : "创建并进入"}
              <ArrowRight size={17} />
            </button>
          </form>
          <button
            className="server-toggle"
            onClick={() => setShowServer(!showServer)}
          >
            <ServerCog size={15} />
            后端连接设置
          </button>
          {showServer && (
            <label className="server-field">
              API 地址
              <input
                value={apiBase}
                onChange={(e) => setBase(e.target.value)}
                placeholder="https://api.example.com"
              />
              <small>本地开发默认使用 http://localhost:18001</small>
            </label>
          )}
        </div>
      </section>
    </main>
  );
}
