"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  Bell,
  CalendarClock,
  ChevronDown,
  CloudSun,
  Cpu,
  Leaf,
  LogOut,
  Menu,
  PanelLeftClose,
  RefreshCw,
  Search,
  ServerCog,
  Settings2,
  X,
} from "lucide-react";
import { clearIdentity, get, getApiBase, getIdentity, setApiBase } from "./api";
import { navItems } from "./data";
import type { Device, JsonMap, Task } from "./types";
import { AuthScreen } from "./components/Auth";
import { Modal } from "./components/Common";
import { DashboardPage } from "./components/Dashboard";
import { ChatPage } from "./components/Chat";
import {
  CalendarPage,
  FieldsPage,
  FinancePage,
  ProfilePage,
} from "./components/Records";
import {
  CalculatorPage,
  EncyclopediaPage,
  PolicyPage,
  WizardPage,
} from "./components/Knowledge";
import { DevicesPage, DocsPage, RulesPage } from "./components/Automation";

type FarmNotification = {
  id: string;
  category: "task" | "weather" | "hardware";
  title: string;
  detail: string;
  destination: "dashboard" | "calendar" | "devices";
  urgent?: boolean;
};

export default function AgricultureApp() {
  const [username, setUsername] = useState("");
  const [page, setPage] = useState("dashboard");
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [apiBase, setBase] = useState("");
  const [notifications, setNotifications] = useState(false);
  const [notificationItems, setNotificationItems] = useState<
    FarmNotification[]
  >([]);
  const [notificationLoading, setNotificationLoading] = useState(false);
  const [notificationError, setNotificationError] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [searchResults, setSearchResults] = useState<{
    crops: string[];
    policies: JsonMap[];
  }>({ crops: [], policies: [] });
  const [encyclopediaCrop, setEncyclopediaCrop] = useState("");
  const searchInput = useRef<HTMLInputElement>(null);

  const loadNotifications = useCallback(async () => {
    setNotificationLoading(true);
    setNotificationError("");
    try {
      const [taskRows, dashboard, deviceRows, pendingRows] = await Promise.all([
        get<Task[]>("/api/tasks"),
        get<JsonMap>("/api/dashboard"),
        get<Device[]>("/api/devices"),
        get<JsonMap[]>("/api/actions/pending"),
      ]);
      const today = new Date().toISOString().slice(0, 10);
      const taskItems: FarmNotification[] = taskRows
        .filter((task) => task.status !== "已完成")
        .sort((left, right) =>
          String(left.end_date || "9999").localeCompare(
            String(right.end_date || "9999"),
          ),
        )
        .map((task) => {
          const overdue =
            task.status === "已逾期" ||
            Boolean(task.end_date && String(task.end_date).slice(0, 10) < today);
          return {
            id: `task:${task.id}`,
            category: "task" as const,
            title: `${overdue ? "任务逾期" : "待办任务"} · ${task.title}`,
            detail: `${task.crop || "通用农事"} · ${
              task.end_date
                ? `截止 ${String(task.end_date).slice(0, 10)}`
                : "未设置截止日期"
            }`,
            destination: "calendar" as const,
            urgent: overdue || task.priority === "high",
          };
        });

      const weatherRows = [
        ...(Array.isArray(dashboard.weather_persistence?.alerts)
          ? dashboard.weather_persistence.alerts
          : []),
        ...(Array.isArray(dashboard.disease_risks)
          ? dashboard.disease_risks
          : []),
      ] as JsonMap[];
      const weatherItems: FarmNotification[] = weatherRows.map(
        (alert, index) => ({
          id: `weather:${index}:${alert.type || alert.disease || "alert"}`,
          category: "weather" as const,
          title:
            alert.type ||
            (alert.disease
              ? `${alert.crop || "作物"} · ${alert.disease}`
              : "天气风险提醒"),
          detail: alert.advice || alert.desc || alert.message || "请及时关注变化",
          destination: "dashboard" as const,
          urgent: true,
        }),
      );
      if (dashboard.weather_alerts?.has_alert && !weatherItems.length) {
        weatherItems.push({
          id: "weather:summary",
          category: "weather",
          title: "天气预警",
          detail: `${dashboard.weather_alerts.region || "当前农场"} · ${
            dashboard.weather_alerts.count || 1
          } 条预警`,
          destination: "dashboard",
          urgent: true,
        });
      }

      const offlineItems: FarmNotification[] = deviceRows
        .filter((device) => device.status !== "online")
        .map((device) => ({
          id: `device:${device.device_id}`,
          category: "hardware" as const,
          title: `设备离线 · ${device.name}`,
          detail: device.location || device.plot_name || device.driver || "请检查连接",
          destination: "devices" as const,
          urgent: true,
        }));
      const pendingItems: FarmNotification[] = pendingRows
        .filter((item) => item.status === "pending")
        .map((item, index) => ({
          id: `action:${item.id || item.action_id || index}`,
          category: "hardware" as const,
          title: `待确认动作 · ${item.command || item.action || "设备操作"}`,
          detail: item.device_name || item.device_id || "请进入设备中心确认",
          destination: "devices" as const,
          urgent: true,
        }));

      setNotificationItems([
        ...taskItems,
        ...weatherItems,
        ...pendingItems,
        ...offlineItems,
      ]);
    } catch (reason) {
      setNotificationError(
        reason instanceof Error ? reason.message : "提醒信息读取失败",
      );
    } finally {
      setNotificationLoading(false);
    }
  }, []);

  useEffect(() => {
    setUsername(getIdentity().username);
    setBase(getApiBase());
  }, []);

  useEffect(() => {
    if (username) loadNotifications();
  }, [loadNotifications, username]);

  useEffect(() => {
    function handleShortcut(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSearchOpen(true);
        requestAnimationFrame(() => searchInput.current?.focus());
      }
      if (event.key === "Escape") setSearchOpen(false);
    }
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, []);

  if (!username) return <AuthScreen onAuthenticated={setUsername} />;

  const current = navItems.find((item) => item.id === page) || navItems[0];
  const PageIcon = current.icon;

  function navigate(id: string) {
    setPage(id);
    setMobileOpen(false);
    setSearchOpen(false);
    setNotifications(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function runGlobalSearch(event: FormEvent) {
    event.preventDefault();
    const keyword = searchQuery.trim();
    if (!keyword) {
      setSearchOpen(true);
      searchInput.current?.focus();
      return;
    }
    setSearchOpen(true);
    setSearching(true);
    setSearchError("");
    try {
      const [cropData, policyData] = await Promise.all([
        get<Record<string, JsonMap>>("/api/encyclopedia"),
        get<JsonMap | JsonMap[]>(
          `/api/policy/search?q=${encodeURIComponent(keyword)}`,
        ),
      ]);
      const crops = Object.entries(cropData)
        .filter(([name, data]) => {
          const aliases = Array.isArray(data.aliases) ? data.aliases : [];
          return (
            name.includes(keyword) ||
            aliases.some((alias) => String(alias).includes(keyword))
          );
        })
        .map(([name]) => name);
      const policies = Array.isArray(policyData)
        ? policyData
        : Array.isArray(policyData.results)
          ? policyData.results
          : [];
      setSearchResults({ crops, policies: policies.slice(0, 5) });
    } catch (reason) {
      setSearchResults({ crops: [], policies: [] });
      setSearchError(
        reason instanceof Error ? reason.message : "搜索服务暂时不可用",
      );
    } finally {
      setSearching(false);
    }
  }

  function logout() {
    clearIdentity();
    setUsername("");
  }

  function collapseNavigation() {
    if (
      typeof window !== "undefined" &&
      window.matchMedia("(max-width: 760px)").matches
    ) {
      setMobileOpen(false);
      return;
    }
    setCollapsed((current) => !current);
  }

  return (
    <div className={`app-shell ${collapsed ? "nav-collapsed" : ""}`}>
      <aside className={`side-nav ${mobileOpen ? "mobile-open" : ""}`}>
        <div className="nav-brand">
          <span>
            <Leaf />
          </span>
          <div>
            <b>青禾</b>
            <small>智能农场</small>
          </div>
          <button className="mobile-close" onClick={() => setMobileOpen(false)}>
            <X />
          </button>
        </div>
        <nav aria-label="主导航">
          <span className="nav-section">农场工作台</span>
          {navItems.slice(0, 6).map((item) => (
            <NavButton
              key={item.id}
              item={item}
              active={page === item.id}
              onClick={() => navigate(item.id)}
            />
          ))}
          <span className="nav-section">知识与自动化</span>
          {navItems.slice(6).map((item) => (
            <NavButton
              key={item.id}
              item={item}
              active={page === item.id}
              onClick={() => navigate(item.id)}
            />
          ))}
        </nav>
        <div className="nav-footer">
          <button onClick={() => setSettingsOpen(true)}>
            <Settings2 />
            <span>系统设置</span>
          </button>
          <button onClick={logout}>
            <LogOut />
            <span>退出登录</span>
          </button>
          <button
            className="collapse-button"
            onClick={collapseNavigation}
            aria-label="收起导航"
          >
            <PanelLeftClose />
            <span>收起导航</span>
          </button>
        </div>
      </aside>
      {mobileOpen && (
        <div className="nav-scrim" onClick={() => setMobileOpen(false)} />
      )}
      <main className="main-shell">
        <header className="topbar">
          <div className="topbar-context">
            <button className="menu-button" onClick={() => setMobileOpen(true)}>
              <Menu />
            </button>
            <span className="context-icon">
              <PageIcon />
            </span>
            <div>
              <small>青禾智能农场</small>
              <b>{current.label}</b>
            </div>
          </div>
          <div className="topbar-actions">
            <div className="global-search-wrap">
              <form className="topbar-search" onSubmit={runGlobalSearch}>
                <Search />
                <input
                  ref={searchInput}
                  value={searchQuery}
                  onFocus={() => setSearchOpen(true)}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  placeholder="搜索作物与政策"
                  aria-label="搜索作物与政策"
                />
                <button className="search-submit" type="submit">
                  搜索
                </button>
                <kbd>Enter</kbd>
              </form>
              {searchOpen && (
                <div className="global-search-panel">
                  <div className="global-search-head">
                    <div>
                      <b>全站搜索</b>
                      <span>直接查找作物知识和农业政策</span>
                    </div>
                    <button
                      className="icon-button"
                      onClick={() => setSearchOpen(false)}
                      aria-label="关闭搜索"
                    >
                      <X />
                    </button>
                  </div>
                  {searching ? (
                    <p className="search-state">正在搜索…</p>
                  ) : searchError ? (
                    <p className="search-state error">{searchError}</p>
                  ) : searchQuery.trim() &&
                    !searchResults.crops.length &&
                    !searchResults.policies.length ? (
                    <p className="search-state">没有找到相关内容</p>
                  ) : searchResults.crops.length ||
                    searchResults.policies.length ? (
                    <div className="global-search-results">
                      {!!searchResults.crops.length && (
                        <section>
                          <span>作物百科</span>
                          {searchResults.crops.map((name) => (
                            <button
                              key={name}
                              onClick={() => {
                                setEncyclopediaCrop(name);
                                navigate("encyclopedia");
                              }}
                            >
                              <Leaf />
                              <b>{name}</b>
                              <small>打开完整种植资料</small>
                            </button>
                          ))}
                        </section>
                      )}
                      {!!searchResults.policies.length && (
                        <section>
                          <span>政策结果</span>
                          {searchResults.policies.map((item, index) => (
                            <article key={index}>
                              <b>
                                {item.title ||
                                  item.policy_name ||
                                  `政策结果 ${index + 1}`}
                              </b>
                              <p>{item.summary || item.content || item.text}</p>
                            </article>
                          ))}
                        </section>
                      )}
                    </div>
                  ) : (
                    <p className="search-state">
                      输入关键词后按 Enter，可在当前页面直接查看结果。
                    </p>
                  )}
                </div>
              )}
            </div>
            <div className="notification-wrap">
              <button
                className="icon-button"
                aria-label="通知"
                aria-expanded={notifications}
                onClick={() => {
                  const next = !notifications;
                  setNotifications(next);
                  if (next) loadNotifications();
                }}
              >
                <Bell />
                {!!notificationItems.length && (
                  <i aria-label={`${notificationItems.length} 条提醒`}>
                    {notificationItems.length > 99
                      ? "99+"
                      : notificationItems.length}
                  </i>
                )}
              </button>
              {notifications && (
                <div className="notification-pop">
                  <div className="notification-head">
                    <div>
                      <strong>农场提醒</strong>
                      <span>{notificationItems.length} 条需要关注的信息</span>
                    </div>
                    <button
                      className="notification-refresh"
                      onClick={loadNotifications}
                      disabled={notificationLoading}
                      aria-label="刷新提醒"
                    >
                      <RefreshCw
                        className={notificationLoading ? "spinning" : ""}
                      />
                    </button>
                  </div>
                  {notificationLoading && !notificationItems.length ? (
                    <p className="notification-state">正在读取农场信息…</p>
                  ) : notificationError ? (
                    <p className="notification-state error">
                      {notificationError}
                    </p>
                  ) : notificationItems.length ? (
                    <div className="notification-groups">
                      {[
                        {
                          id: "task" as const,
                          label: "当前任务",
                          icon: <CalendarClock />,
                        },
                        {
                          id: "weather" as const,
                          label: "天气与风险",
                          icon: <CloudSun />,
                        },
                        {
                          id: "hardware" as const,
                          label: "硬件通知",
                          icon: <Cpu />,
                        },
                      ].map((group) => {
                        const items = notificationItems.filter(
                          (item) => item.category === group.id,
                        );
                        if (!items.length) return null;
                        return (
                          <section key={group.id}>
                            <div className="notification-group-title">
                              <span>{group.icon}</span>
                              <b>{group.label}</b>
                              <small>{items.length}</small>
                            </div>
                            {items.map((item) => (
                              <button
                                className={`notification-item ${
                                  item.urgent ? "urgent" : ""
                                }`}
                                key={item.id}
                                onClick={() => navigate(item.destination)}
                              >
                                <i />
                                <span>
                                  <b>{item.title}</b>
                                  <small>{item.detail}</small>
                                </span>
                              </button>
                            ))}
                          </section>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="notification-empty">
                      <Bell />
                      <b>暂无需要处理的提醒</b>
                      <span>任务、天气和设备状态都很平稳。</span>
                    </div>
                  )}
                </div>
              )}
            </div>
            <button className="user-chip" onClick={() => navigate("profile")}>
              <span>{username.slice(0, 1).toUpperCase()}</span>
              <div>
                <b>{username}</b>
                <small>农场管理员</small>
              </div>
              <ChevronDown />
            </button>
          </div>
        </header>
        <div className="page-content">
          {page === "dashboard" && <DashboardPage onNavigate={navigate} />}
          {page === "chat" && <ChatPage />}
          {page === "profile" && <ProfilePage />}
          {page === "fields" && <FieldsPage />}
          {page === "finance" && <FinancePage />}
          {page === "calendar" && <CalendarPage />}
          {page === "policy" && <PolicyPage />}
          {page === "encyclopedia" && (
            <EncyclopediaPage initialCrop={encyclopediaCrop} />
          )}
          {page === "calculator" && <CalculatorPage />}
          {page === "wizard" && <WizardPage onNavigate={navigate} />}
          {page === "devices" && <DevicesPage />}
          {page === "rules" && <RulesPage />}
          {page === "docs" && <DocsPage />}
        </div>
      </main>
      <nav className="mobile-tabs">
        {navItems.slice(0, 4).map((item) => (
          <NavButton
            key={item.id}
            item={item}
            active={page === item.id}
            onClick={() => navigate(item.id)}
          />
        ))}
        <button onClick={() => setMobileOpen(true)}>
          <Menu />
          <span>全部</span>
        </button>
      </nav>
      {settingsOpen && (
        <Modal title="系统设置" onClose={() => setSettingsOpen(false)}>
          <div className="settings-panel">
            <div className="setting-row">
              <ServerCog />
              <div>
                <b>FastAPI 后端</b>
                <small>新前端所有业务数据都通过此地址交换</small>
              </div>
            </div>
            <label>
              API 地址
              <input
                value={apiBase}
                onChange={(event) => setBase(event.target.value)}
              />
            </label>
            <button
              className="primary-button"
              onClick={() => {
                setApiBase(apiBase);
                setSettingsOpen(false);
                location.reload();
              }}
            >
              保存并重新连接
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function NavButton({
  item,
  active,
  onClick,
}: {
  item: (typeof navItems)[number];
  active: boolean;
  onClick: () => void;
}) {
  const Icon = item.icon;
  return (
    <button
      className={`nav-item ${active ? "active" : ""}`}
      onClick={onClick}
      title={item.label}
    >
      <Icon />
      <span>{item.label}</span>
      {active && <i />}
    </button>
  );
}
