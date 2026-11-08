"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  CalendarCheck,
  Check,
  CircleDollarSign,
  Download,
  Edit3,
  Plus,
  Play,
  RefreshCw,
  Save,
  Sprout,
  Trash2,
  TrendingDown,
  TrendingUp,
  UserRound,
} from "lucide-react";
import { get, post, put, remove } from "../api";
import { experienceLevels, goals, soils } from "../data";
import type { Field, JsonMap, Profile, Progress, Task } from "../types";
import {
  Card,
  Empty,
  ErrorState,
  Loading,
  Notice,
  PageHeader,
  StatusPill,
} from "./Common";
import { FieldCreateMap } from "./FieldCreateMap";
import { FieldOverviewMap } from "./FieldMap";
import { normalizeFieldWeather } from "../weather-state";
import { localDateKey } from "../local-date";
import {
  isTaskRunning,
  overallProgress,
  selectCalendarTasks,
  type TaskFilter,
} from "../task-view";
import { useVisiblePolling } from "../use-visible-polling";
import {
  compareRecentFinanceRecords,
  formatFinanceAmount,
} from "../finance-records";

const blankProfile: Profile = {
  user_region: "",
  user_soil_type: "",
  user_farm_size: 1,
  user_experience: "",
  user_goals: [],
  user_phone: "",
};

export function ProfilePage() {
  const [profile, setProfile] = useState<Profile>(blankProfile);
  const [autonomy, setAutonomy] = useState("medium");
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [profileError, setProfileError] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [effectiveAutonomy, setEffectiveAutonomy] = useState("medium");
  const [profileLoaded, setProfileLoaded] = useState(false);
  useEffect(() => {
    get<Profile>("/api/profile")
      .then((data) => {
        setProfile({ ...blankProfile, ...data });
        setProfileLoaded(true);
        const next = data.autonomy_level;
        if (next === "low" || next === "medium" || next === "high") {
          setAutonomy(next);
          setEffectiveAutonomy(next);
        }
      })
      .catch((reason) =>
        setProfileError(
          reason instanceof Error ? reason.message : "档案读取失败",
        ),
      )
      .finally(() => setLoading(false));
  }, []);
  async function save(event: FormEvent) {
    event.preventDefault();
    if (savingProfile || !profileLoaded) return;
    setSavingProfile(true);
    setMessage("");
    setProfileError("");
    try {
      const result = await post<JsonMap>("/api/profile", {
        ...profile,
        autonomy_level: autonomy,
      });
      if (result.success === false)
        throw new Error(result.error || "档案保存失败");
      const saved = await get<Profile>("/api/profile");
      setProfile({ ...blankProfile, ...saved });
      const next = saved.autonomy_level || "medium";
      setAutonomy(next);
      setEffectiveAutonomy(next);
      setMessage("种植档案与设备操作模式已更新");
    } catch (reason) {
      setProfileError(
        reason instanceof Error ? reason.message : "档案保存失败",
      );
    } finally {
      setSavingProfile(false);
    }
  }
  if (loading) return <Loading label="正在读取种植档案" />;
  return (
    <>
      <PageHeader
        eyebrow="FARMER PROFILE"
        title="基本信息"
        description="这些资料会参与种植计划、天气建议和智能体决策。"
      />
      {message && <Notice>{message}</Notice>}
      {profileError && (
        <ErrorState
          message={profileError}
          retry={() => {
            get<Profile>("/api/profile")
              .then((data) => {
                setProfile({ ...blankProfile, ...data });
                setProfileLoaded(true);
                setAutonomy(data.autonomy_level || "medium");
                setEffectiveAutonomy(data.autonomy_level || "medium");
                setProfileError("");
              })
              .catch((reason) => setProfileError(reason.message));
          }}
        />
      )}
      <form onSubmit={save} className="profile-layout">
        <Card className="profile-summary">
          <div className="profile-avatar">
            <UserRound />
          </div>
          <h2>我的种植档案</h2>
          <p>
            {profile.user_region || "尚未设置地区"} · {profile.user_farm_size}{" "}
            亩
          </p>
          <div className="profile-facts">
            <div>
              <span>土壤</span>
              <b>{profile.user_soil_type || "未设置"}</b>
            </div>
            <div>
              <span>经验</span>
              <b>{profile.user_experience || "未设置"}</b>
            </div>
            <div>
              <span>目标</span>
              <b>{profile.user_goals.length} 项</b>
            </div>
          </div>
        </Card>
        <div className="profile-forms">
          <Card title="农场资料">
            <div className="form-grid two">
              <label>
                所在地区
                <input
                  value={profile.user_region}
                  onChange={(e) =>
                    setProfile({ ...profile, user_region: e.target.value })
                  }
                  placeholder="如：河北保定"
                />
              </label>
              <label>
                土壤类型
                <select
                  value={profile.user_soil_type}
                  onChange={(e) =>
                    setProfile({ ...profile, user_soil_type: e.target.value })
                  }
                >
                  <option value="">请选择</option>
                  {soils.map((item) => (
                    <option key={item}>{item}</option>
                  ))}
                </select>
              </label>
              <label>
                种植面积（亩）
                <input
                  type="number"
                  min="0"
                  step="0.5"
                  value={profile.user_farm_size}
                  onChange={(e) =>
                    setProfile({
                      ...profile,
                      user_farm_size: Number(e.target.value),
                    })
                  }
                />
              </label>
              <label>
                种植经验
                <select
                  value={profile.user_experience}
                  onChange={(e) =>
                    setProfile({ ...profile, user_experience: e.target.value })
                  }
                >
                  <option value="">请选择</option>
                  {experienceLevels.map((item) => (
                    <option key={item}>{item}</option>
                  ))}
                </select>
              </label>
              <label className="full">
                手机号码
                <input
                  value={profile.user_phone}
                  onChange={(e) =>
                    setProfile({ ...profile, user_phone: e.target.value })
                  }
                  placeholder="用于农事短信提醒"
                />
              </label>
            </div>
            <div className="choice-group">
              <span>种植目标</span>
              <div>
                {goals.map((goal) => (
                  <button
                    type="button"
                    className={
                      profile.user_goals.includes(goal) ? "selected" : ""
                    }
                    key={goal}
                    onClick={() =>
                      setProfile({
                        ...profile,
                        user_goals: profile.user_goals.includes(goal)
                          ? profile.user_goals.filter((item) => item !== goal)
                          : [...profile.user_goals, goal],
                      })
                    }
                    aria-pressed={profile.user_goals.includes(goal)}
                  >
                    {goal}
                  </button>
                ))}
              </div>
            </div>
          </Card>
          <Card title="智能体自主权">
            <p className="section-hint">
              硬安全限制始终生效，你可以决定正常操作需要多少人工确认。
            </p>
            <div className="autonomy-current" aria-live="polite">
              <span>
                {autonomy === effectiveAutonomy
                  ? "当前生效"
                  : "当前选择（尚未保存）"}
              </span>
              <b>
                {autonomy === "low"
                  ? "谨慎模式"
                  : autonomy === "high"
                    ? "自主模式"
                    : "协作模式"}
              </b>
            </div>
            <div className="autonomy-options">
              {[
                {
                  id: "low",
                  title: "谨慎模式",
                  body: "启动与参数调整均需确认，停止操作立即执行",
                  tag: "最安全",
                },
                {
                  id: "medium",
                  title: "协作模式",
                  body: "规则范围内自动执行，超限确认",
                  tag: "推荐",
                },
                {
                  id: "high",
                  title: "自主模式",
                  body: "在安全边界内完全自主决策",
                  tag: "高效率",
                },
              ].map((item) => (
                <button
                  type="button"
                  key={item.id}
                  className={autonomy === item.id ? "selected" : ""}
                  onClick={() => setAutonomy(item.id)}
                  aria-pressed={autonomy === item.id}
                >
                  <span>
                    {item.title}
                    <em>{item.tag}</em>
                  </span>
                  <p>{item.body}</p>
                </button>
              ))}
            </div>
          </Card>
          <button
            className="primary-button save-profile"
            disabled={savingProfile || !profileLoaded}
          >
            <Save />
            {savingProfile ? "正在保存" : "保存全部修改"}
          </button>
        </div>
      </form>
    </>
  );
}

export function FieldsPage() {
  const [fields, setFields] = useState<Field[]>([]);
  const [weather, setWeather] = useState<Record<string, JsonMap | null>>({});
  const [showCreate, setShowCreate] = useState(false);
  const [copySource, setCopySource] = useState<Field | null>(null);
  const [selected, setSelected] = useState<Field | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    try {
      const rows = await get<Field[]>("/api/fields");
      setFields(rows);
      setWeather({});
      setSelected(
        (current) =>
          rows.find((item) => item.id === current?.id) || rows[0] || null,
      );
      setError("");
      Promise.all(
        rows.map(async (field) => {
          if (!field.center_lat || !field.center_lon) return;
          try {
            const data = await get(
              `/api/weather-by-coordinates?lon=${field.center_lon}&lat=${field.center_lat}`,
            );
            setWeather((current) => ({
              ...current,
              [field.id]: normalizeFieldWeather(data),
            }));
          } catch {
            setWeather((current) => ({ ...current, [field.id]: null }));
          }
        }),
      ).catch(() => null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "地块读取失败");
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);
  async function deleteField(id: string) {
    if (!confirm("确定删除这个地块吗？")) return;
    await remove(`/api/fields/${id}`);
    await load();
  }
  if (loading) return <Loading label="正在加载地块" />;
  return (
    <>
      <PageHeader
        eyebrow="FIELD OPERATIONS"
        title="地块管理"
        description="管理边界、作物、土壤与每块田的实时天气。"
        actions={
          <button
            className="primary-button"
            onClick={() => {
              setCopySource(null);
              setShowCreate(true);
            }}
          >
            <Plus />
            添加地块
          </button>
        }
      />
      {error && <ErrorState message={error} retry={load} />}
      <div className="field-layout">
        <Card className="field-map-card">
          <div className="field-map-toolbar">
            <div>
              <b>地块总览</b>
              <span>
                {fields.length} 块 ·{" "}
                {fields
                  .reduce((sum, field) => sum + Number(field.area_mu || 0), 0)
                  .toFixed(1)}{" "}
                亩
              </span>
            </div>
            <button
              className="icon-button"
              onClick={load}
              aria-label="刷新地块与天气"
            >
              <RefreshCw />
            </button>
          </div>
          <FieldOverviewMap
            fields={fields}
            selectedId={selected?.id}
            onSelect={setSelected}
          />
          {selected && (
            <div className="selected-field">
              <div className="selected-field-name">
                <span className="field-symbol">
                  {selected.current_crop ? "禾" : "田"}
                </span>
                <div>
                  <b>{selected.name}</b>
                  <span>
                    {selected.soil_type || "未设置土壤"} ·{" "}
                    {selected.current_crop || "暂未种植"}
                  </span>
                </div>
              </div>
              <div className="selected-field-metric">
                <strong>{selected.area_mu.toFixed(2)}</strong>
                <span>亩</span>
              </div>
              <div className="selected-field-metric">
                <strong>{weather[selected.id]?.temp ?? "—"}°</strong>
                <span>实时温度</span>
              </div>
            </div>
          )}
        </Card>
        <Card title="我的地块" className="field-list-card">
          {fields.length ? (
            <div className="field-list">
              {fields.map((field) => (
                <button
                  className={selected?.id === field.id ? "selected" : ""}
                  onClick={() => setSelected(field)}
                  key={field.id}
                  aria-pressed={selected?.id === field.id}
                >
                  <span className="field-index">{field.name.slice(0, 1)}</span>
                  <div>
                    <b>{field.name}</b>
                    <small>
                      {field.current_crop || "未种植"} ·{" "}
                      {field.area_mu.toFixed(2)}亩
                    </small>
                  </div>
                  <StatusPill tone={weather[field.id] ? "success" : "neutral"}>
                    {weather[field.id]
                      ? `${weather[field.id]?.temp}°C`
                      : weather[field.id] === null
                        ? "暂不可用"
                        : "待同步"}
                  </StatusPill>
                </button>
              ))}
            </div>
          ) : (
            <Empty title="暂无地块" body="添加第一块田，开始建立数字农场。" />
          )}
          {selected && (
            <div className="field-actions">
              <button
                className="secondary-button"
                onClick={() => {
                  setCopySource(selected);
                  setShowCreate(true);
                }}
              >
                <Edit3 />
                复制新建
              </button>
              <button
                className="danger-button"
                onClick={() => deleteField(selected.id)}
              >
                <Trash2 />
                删除
              </button>
            </div>
          )}
        </Card>
      </div>
      {selected && <FieldHistory field={selected} onSaved={load} />}
      {showCreate && (
        <FieldCreator
          source={copySource}
          onClose={() => setShowCreate(false)}
          onSaved={async () => {
            setShowCreate(false);
            await load();
          }}
        />
      )}
    </>
  );
}

function FieldCreator({
  source,
  onClose,
  onSaved,
}: {
  source: Field | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState(source ? `${source.name}（副本）` : "");
  const [soil, setSoil] = useState(source?.soil_type || soils[0] || "壤土");
  const [crop, setCrop] = useState(source?.current_crop || "");
  const [points, setPoints] = useState<number[][]>(
    () => source?.coordinates.map((point) => point.slice(0, 2)) || [],
  );
  const [busy, setBusy] = useState(false);
  async function save(event: FormEvent) {
    event.preventDefault();
    if (!name || points.length < 3) return;
    setBusy(true);
    try {
      await post("/api/fields", {
        name,
        soil_type: soil,
        current_crop: crop,
        coordinates: points.map((point) => point.slice(0, 2)),
      });
      onSaved();
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <div className="modal wide-modal">
        <div className="modal-head">
          <h2>{source ? "复制地块" : "绘制新地块"}</h2>
          <button className="icon-button" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="field-create-grid">
          <div>
            <FieldCreateMap points={points} setPoints={setPoints} />
            <small className="canvas-caption">
              已标记 {points.length} 个点 · 至少需要 3
              个边界点，保存后自动计算面积和中心位置
            </small>
          </div>
          <form onSubmit={save}>
            <label>
              地块名称
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="如：东侧麦田"
              />
            </label>
            <label>
              土壤类型
              <select value={soil} onChange={(e) => setSoil(e.target.value)}>
                {soils.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label>
              当前作物
              <input
                value={crop}
                onChange={(e) => setCrop(e.target.value)}
                placeholder="可稍后设置"
              />
            </label>
            <div className="coordinate-list">
              {points.slice(0, 5).map((point, index) => (
                <span key={index}>
                  P{index + 1} {point[1].toFixed(4)}, {point[0].toFixed(4)}
                </span>
              ))}
            </div>
            <button
              className="primary-button"
              disabled={!name || points.length < 3 || busy}
            >
              <Save />
              {busy ? "正在保存" : "保存地块"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

function FieldHistory({
  field,
  onSaved,
}: {
  field: Field;
  onSaved: () => void;
}) {
  const [crop, setCrop] = useState("");
  const [season, setSeason] = useState("");
  const [yieldAmount, setYield] = useState(0);
  const [notes, setNotes] = useState("");
  async function save(event: FormEvent) {
    event.preventDefault();
    await post(`/api/fields/${field.id}/history`, {
      crop,
      season,
      yield_amount: yieldAmount,
      notes,
    });
    setCrop("");
    setSeason("");
    setYield(0);
    setNotes("");
    onSaved();
  }
  return (
    <Card title={`${field.name} · 种植历史`}>
      <div className="history-layout">
        <div>
          {field.history?.length ? (
            field.history.map((item, index) => (
              <div className="history-item" key={index}>
                <span>{item.season || "往季"}</span>
                <div>
                  <b>{item.crop}</b>
                  <small>
                    产量 {item.yield_amount || 0} kg · {item.notes || "无备注"}
                  </small>
                </div>
              </div>
            ))
          ) : (
            <Empty title="暂无历史记录" body="记录轮作和产量，帮助后续决策。" />
          )}
        </div>
        <form className="compact-form" onSubmit={save}>
          <b>添加历史记录</b>
          <input
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
            placeholder="作物名称"
            required
          />
          <input
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            placeholder="季节，如 2026春"
          />
          <input
            type="number"
            min="0"
            value={yieldAmount}
            onChange={(e) => setYield(Number(e.target.value))}
            placeholder="产量 kg"
          />
          <input
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="备注"
          />
          <button className="secondary-button">
            <Plus />
            添加记录
          </button>
        </form>
      </div>
    </Card>
  );
}

export function FinancePage() {
  const [summary, setSummary] = useState<JsonMap>({});
  const [costs, setCosts] = useState<JsonMap[]>([]);
  const [income, setIncome] = useState<JsonMap[]>([]);
  const [type, setType] = useState<"cost" | "income">("cost");
  const [crop, setCrop] = useState("");
  const [amount, setAmount] = useState(0);
  const [category, setCategory] = useState("种子");
  const [item, setItem] = useState("");
  const [message, setMessage] = useState("");
  const [financeError, setFinanceError] = useState("");
  const [saving, setSaving] = useState(false);
  const load = useCallback(async () => {
    const [report, costRows, incomeRows] = await Promise.all([
      get("/api/finance/summary"),
      get<JsonMap[]>("/api/finance/costs"),
      get<JsonMap[]>("/api/finance/income"),
    ]);
    setSummary(report);
    setCosts(costRows);
    setIncome(incomeRows);
  }, []);
  useEffect(() => {
    load().catch((reason) =>
      setFinanceError(
        reason instanceof Error ? reason.message : "财务数据读取失败",
      ),
    );
  }, [load]);
  async function save(event: FormEvent) {
    event.preventDefault();
    if (saving) return;
    setMessage("");
    setFinanceError("");
    if (!crop.trim()) return setFinanceError("请输入作物名称");
    if (!Number.isFinite(amount) || amount <= 0)
      return setFinanceError("请输入大于零的有效金额");
    setSaving(true);
    try {
      if (type === "cost")
        await post("/api/finance/costs", {
          crop: crop.trim(),
          cost_type: category,
          item_name: item.trim() || category,
          unit_price: amount,
        });
      else
        await post("/api/finance/income", {
          crop: crop.trim(),
          quantity: 1,
          unit_price: amount,
        });
      setMessage(
        `已记录${type === "cost" ? "成本" : "收入"} ¥${amount.toFixed(2)}`,
      );
      setAmount(0);
      try {
        await load();
      } catch {
        setFinanceError("记录已保存，统计更新失败，请刷新页面重试");
      }
    } catch (reason) {
      setFinanceError(
        reason instanceof Error ? reason.message : "记账失败，请稍后重试",
      );
    } finally {
      setSaving(false);
    }
  }
  async function exportCsv() {
    const data = await get<{ csv: string }>("/api/finance/export");
    const blob = new Blob([data.csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `finance_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }
  const totalIncome = income.reduce(
    (sum, row) => sum + Number(row.total_amount || 0),
    0,
  );
  const totalCost = costs.reduce(
    (sum, row) => sum + Number(row.total_amount || 0),
    0,
  );
  const reports = (summary.crop_reports || []) as JsonMap[];
  const records: Array<JsonMap & { recordType: string }> = [
    ...costs.map((row): JsonMap & { recordType: string } => ({
      ...row,
      recordType: "成本",
    })),
    ...income.map((row): JsonMap & { recordType: string } => ({
      ...row,
      recordType: "收入",
    })),
  ]
    .sort(compareRecentFinanceRecords)
    .slice(0, 10);
  return (
    <>
      <PageHeader
        eyebrow="FARM FINANCE"
        title="财务管理"
        description="记录投入与收成，看清每种作物真正带来的收益。"
        actions={
          <button className="secondary-button" onClick={exportCsv}>
            <Download />
            导出 CSV
          </button>
        }
      />
      {message && <Notice>{message}</Notice>}
      {financeError && <ErrorState message={financeError} />}
      <div className="metric-grid three finance-metrics">
        <Card className="finance-metric income">
          <span>
            <TrendingUp />
          </span>
          <div>
            <small>累计收入</small>
            <strong>¥{formatFinanceAmount(totalIncome)}</strong>
          </div>
        </Card>
        <Card className="finance-metric cost">
          <span>
            <TrendingDown />
          </span>
          <div>
            <small>累计成本</small>
            <strong>¥{formatFinanceAmount(totalCost)}</strong>
          </div>
        </Card>
        <Card className="finance-metric profit">
          <span>
            <CircleDollarSign />
          </span>
          <div>
            <small>净收益</small>
            <strong>¥{formatFinanceAmount(totalIncome - totalCost)}</strong>
          </div>
        </Card>
      </div>
      <div className="content-grid finance-grid">
        <Card title="快速记账">
          <form onSubmit={save} className="finance-form">
            <div className="segmented">
              <button
                type="button"
                className={type === "cost" ? "active" : ""}
                onClick={() => setType("cost")}
                disabled={saving}
              >
                成本支出
              </button>
              <button
                type="button"
                className={type === "income" ? "active" : ""}
                onClick={() => setType("income")}
                disabled={saving}
              >
                销售收入
              </button>
            </div>
            <label>
              作物
              <input
                value={crop}
                onChange={(e) => setCrop(e.target.value)}
                placeholder="小麦"
                disabled={saving}
                required
              />
            </label>
            {type === "cost" && (
              <>
                <label>
                  成本类型
                  <select
                    value={category}
                    disabled={saving}
                    onChange={(e) => setCategory(e.target.value)}
                  >
                    {["种子", "肥料", "农药", "人工", "农机", "其他"].map(
                      (item) => (
                        <option key={item}>{item}</option>
                      ),
                    )}
                  </select>
                </label>
                <label>
                  项目说明
                  <input
                    value={item}
                    disabled={saving}
                    onChange={(e) => setItem(e.target.value)}
                    placeholder="如：春季底肥"
                  />
                </label>
              </>
            )}
            <label>
              {type === "cost" ? "支出金额" : "销售总额"}
              <div className="money-input">
                <span>¥</span>
                <input
                  type="number"
                  min="0.01"
                  step="0.01"
                  disabled={saving}
                  value={amount || ""}
                  onChange={(e) => setAmount(Number(e.target.value))}
                  required
                />
              </div>
            </label>
            <button className="primary-button" disabled={saving}>
              <Save />
              {saving ? "正在保存" : "保存记录"}
            </button>
          </form>
        </Card>
        <Card title="按作物经营表现">
          {reports.length ? (
            <div className="finance-table">
              <div className="table-head">
                <span>作物</span>
                <span>收入</span>
                <span>成本</span>
                <span>净利润</span>
              </div>
              {reports.map((row: JsonMap, index: number) => (
                <div className="table-row" key={index}>
                  <b>{row.crop || "其他"}</b>
                  <span>¥{formatFinanceAmount(row.total_income)}</span>
                  <span>¥{formatFinanceAmount(row.total_cost)}</span>
                  <strong
                    className={
                      Number(row.net_profit || 0) >= 0 ? "positive" : "negative"
                    }
                  >
                    ¥{formatFinanceAmount(row.net_profit)}
                  </strong>
                </div>
              ))}
            </div>
          ) : (
            <Empty
              title="暂无财务数据"
              body="保存第一笔收支后，这里会形成经营分析。"
            />
          )}
        </Card>
      </div>
      <Card title="最近流水">
        <div className="record-list">
          {records.map((row, index) => (
            <div className="finance-record-row" key={index}>
              <span
                className={`record-icon ${
                  row.recordType === "收入" ? "income" : "cost"
                }`}
              >
                {row.recordType === "收入" ? <TrendingUp /> : <TrendingDown />}
              </span>
              <div className="finance-record-copy">
                <b>{row.crop || row.item_name || "未分类"}</b>
                <small>
                  {row.date || "今天"} · {row.category || row.recordType}
                </small>
              </div>
              <strong
                className={`record-amount ${
                  row.recordType === "收入" ? "positive" : "negative"
                }`}
              >
                {row.recordType === "收入" ? "+" : "-"}¥
                {Number(row.total_amount || 0).toFixed(2)}
              </strong>
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}

export function CalendarPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [progress, setProgress] = useState<Progress[]>([]);
  const [calendarFields, setCalendarFields] = useState<Field[]>([]);
  const [bindingProgress, setBindingProgress] = useState("");
  const [showForm, setShowForm] = useState<"task" | "progress" | null>(null);
  const [formError, setFormError] = useState("");
  const [saving, setSaving] = useState(false);
  const [calendarError, setCalendarError] = useState("");
  const [calendarNotice, setCalendarNotice] = useState("");
  const [busyTask, setBusyTask] = useState("");
  const [editingTask, setEditingTask] = useState("");
  const [detailTaskId, setDetailTaskId] = useState("");
  const [taskFilter, setTaskFilter] = useState<TaskFilter>("active");
  const [selectedDate, setSelectedDate] = useState("");
  const [updatedAt, setUpdatedAt] = useState("");
  const loadSequence = useRef(0);
  const [form, setForm] = useState<JsonMap>({
    crop: "",
    title: "",
    task_type: "浇水",
    priority: "medium",
    stage: "准备期",
    total_stages: 5,
    start_date: localDateKey(),
  });
  const load = useCallback(async () => {
    const sequence = ++loadSequence.current;
    const [taskRows, progressRows, fieldRows] = await Promise.all([
      get<Task[]>("/api/tasks"),
      get<Progress[]>("/api/progress"),
      get<Field[]>("/api/fields"),
    ]);
    if (sequence === loadSequence.current) {
      setTasks(taskRows);
      setProgress(progressRows);
      setCalendarFields(fieldRows);
      setCalendarError("");
      setUpdatedAt(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
    }
  }, []);
  useEffect(() => {
    load().catch((reason) =>
      setCalendarError(
        reason instanceof Error ? reason.message : "农事数据读取失败",
      ),
    );
  }, [load]);
  useVisiblePolling(
    async () => {
      try {
        await load();
      } catch (reason) {
        setCalendarError(
          reason instanceof Error ? reason.message : "农事数据刷新失败",
        );
      }
    },
    5000,
    !saving && !busyTask && !bindingProgress,
  );
  useEffect(() => {
    if (!showForm && !detailTaskId) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !saving) {
        setShowForm(null);
        setDetailTaskId("");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showForm, detailTaskId, saving]);
  function openTask(task?: Task) {
    setEditingTask(task?.id || "");
    setForm({
      crop: task?.crop || "",
      title: task?.title || "",
      task_type: task?.task_type || "浇水",
      description: task?.description || "",
      priority: task?.priority || "medium",
      start_date: task?.start_date || "",
      end_date: task?.end_date?.slice(0, 10) || selectedDate || localDateKey(),
    });
    setDetailTaskId("");
    setFormError("");
    setShowForm("task");
  }
  async function create(event: FormEvent) {
    event.preventDefault();
    if (saving) return;
    setFormError("");
    setSaving(true);
    try {
      const taskPayload = {
        crop: form.crop,
        title: String(form.title || "").trim(),
        task_type: form.task_type,
        description: form.description || `${form.task_type}任务`,
        priority: form.priority,
        start_date: form.start_date || "",
        end_date: form.end_date || "",
      };
      if (
        showForm === "task" &&
        taskPayload.start_date &&
        taskPayload.end_date &&
        taskPayload.start_date > taskPayload.end_date
      ) {
        throw new Error("截止日期不能早于开始日期");
      }
      const result =
        showForm === "task"
          ? editingTask
            ? await put<JsonMap>(`/api/tasks/${editingTask}`, taskPayload)
            : await post<JsonMap>("/api/tasks", {
                ...taskPayload,
                status: "待办",
                progress_percent: 0,
              })
          : await post<JsonMap>("/api/progress", {
              crop: form.crop,
              plot_id: form.plot_id ? Number(form.plot_id) : null,
              stage: form.stage,
              total_stages: Number(form.total_stages),
              status: "进行中",
              start_date: form.start_date || localDateKey(),
            });
      if (result.success === false) {
        throw new Error(result.error || "保存失败，请稍后重试");
      }
      await load();
      setShowForm(null);
      setCalendarNotice(
        editingTask ? "任务已更新，关联提醒已同步" : "已添加到农事日历",
      );
    } catch (reason) {
      setFormError(
        reason instanceof Error ? reason.message : "保存失败，请稍后重试",
      );
    } finally {
      setSaving(false);
    }
  }
  async function bindProgress(id: string, plotId: string) {
    if (bindingProgress) return;
    setBindingProgress(id);
    setCalendarError("");
    setCalendarNotice("");
    try {
      const result = await put<JsonMap>(`/api/progress/${encodeURIComponent(id)}/plot`, {
        plot_id: plotId ? Number(plotId) : null,
      });
      if (result.success === false) throw new Error(result.error || result.message || "地块绑定失败");
      await load();
      setCalendarNotice(plotId ? "计划已绑定地块" : "计划已解除地块绑定");
    } catch (reason) {
      setCalendarError(reason instanceof Error ? reason.message : "地块绑定失败");
    } finally {
      setBindingProgress("");
    }
  }
  async function actOnTask(
    id: string,
    action: "complete" | "execute" | "delete",
  ) {
    if (busyTask) return;
    const task = tasks.find((item) => item.id === id);
    if (!task) return;
    if (
      action === "delete" &&
      !window.confirm(
        `确定删除任务“${task.title}”吗？关联的单次提醒也会取消，删除后无法恢复。`,
      )
    )
      return;
    if (
      action === "complete" &&
      task.device_id &&
      !window.confirm(
        `确认“${task.title}”已在现场完成吗？此按钮仅登记作业结果，启动设备请使用执行按钮。`,
      )
    )
      return;
    setBusyTask(id);
    setCalendarError("");
    setCalendarNotice("");
    try {
      const result =
        action === "delete"
          ? await remove<JsonMap>(`/api/tasks/${id}`)
          : await post<JsonMap>(`/api/tasks/${id}/${action}`);
      if (result.decision === "need_confirm") {
        setCalendarNotice(
          result.message || "任务已提交，请到设备中心确认操作。",
        );
      } else if (result.success === false) {
        throw new Error(result.error || result.message || "任务操作失败");
      } else {
        setCalendarNotice(
          action === "execute"
            ? result.message || "指令已下发，任务状态会随设备运行结果更新"
            : action === "complete"
              ? "任务已完成"
              : "任务已删除",
        );
      }
      await load();
    } catch (reason) {
      setCalendarError(
        reason instanceof Error ? reason.message : "任务操作失败",
      );
    } finally {
      setBusyTask("");
    }
  }
  const visibleTasks = selectCalendarTasks(tasks, taskFilter, selectedDate);
  const detailTask = tasks.find((task) => task.id === detailTaskId);
  const days = Array.from({ length: 14 }, (_, index) => {
    const date = new Date();
    date.setDate(date.getDate() + index);
    return date;
  });
  const dateKey = localDateKey;
  const todayKey = dateKey(new Date());
  const progressValue = overallProgress;
  return (
    <>
      <PageHeader
        eyebrow="FARM CALENDAR"
        title="农事日历"
        description="把种植周期、任务截止时间和每日执行安排放在一条时间线上。"
        actions={
          <>
            <button
              className="secondary-button"
              onClick={() => {
                setFormError("");
                setEditingTask("");
                setForm({
                  crop: "",
                  stage: "准备期",
                  total_stages: 5,
                  start_date: localDateKey(),
                  plot_id: "",
                });
                setShowForm("progress");
              }}
            >
              <Sprout />
              添加进度
            </button>
            <button className="primary-button" onClick={() => openTask()}>
              <Plus />
              添加任务
            </button>
          </>
        }
      />
      {calendarError && (
        <ErrorState
          message={calendarError}
          retry={() =>
            load()
              .then(() => setCalendarError(""))
              .catch((reason) => setCalendarError(reason.message))
          }
        />
      )}
      {calendarNotice && <Notice>{calendarNotice}</Notice>}
      <div className="calendar-workspace">
        <Card
          className="calendar-board"
          title="未来 14 天"
          action={<span className="calendar-range">点击日期查看任务</span>}
        >
          <div className="calendar-days">
            {days.map((day) => {
              const currentDateKey = dateKey(day);
              const dayTasks = selectCalendarTasks(
                tasks,
                taskFilter,
                "",
              ).filter(
                (task) => task.end_date?.slice(0, 10) === currentDateKey,
              );
              return (
                <button
                  type="button"
                  aria-pressed={selectedDate === currentDateKey}
                  aria-label={`${currentDateKey}，${dayTasks.length}项任务`}
                  onClick={() =>
                    setSelectedDate(
                      selectedDate === currentDateKey ? "" : currentDateKey,
                    )
                  }
                  className={`${currentDateKey === todayKey ? "today" : ""} ${selectedDate === currentDateKey ? "selected" : ""}`}
                  key={currentDateKey}
                >
                  <span className="calendar-weekday">
                    周{["日", "一", "二", "三", "四", "五", "六"][day.getDay()]}
                  </span>
                  <b>{day.getDate()}</b>
                  <small>{day.getMonth() + 1} 月</small>
                  {dayTasks.length ? (
                    <em>{dayTasks.length} 项任务</em>
                  ) : (
                    <em className="quiet">无任务</em>
                  )}
                </button>
              );
            })}
          </div>
        </Card>
        <Card className="growth-timeline-card" title="本季种植时间线">
          <p className="section-hint">绑定地块后才能按地块分析生长阶段；未绑定的进度仍可作为个人计划。</p>
          {progress.length ? (
            <div className="calendar-plan-timeline">
              {progress.map((item) => {
                const value = progressValue(item);
                return (
                  <article className="plan-line" key={item.id}>
                    <div className="plan-line-head">
                      <span className="crop-avatar">
                        {item.crop.slice(0, 1)}
                      </span>
                      <div>
                        <b>{item.crop}</b>
                        <small>{item.stage || "阶段待更新"}</small>
                      </div>
                      <StatusPill
                        tone={item.status === "已完成" ? "success" : "info"}
                      >
                        {item.status || "进行中"}
                      </StatusPill>
                    </div>
                    <label>
                      关联地块
                      <select aria-label={`${item.crop}关联地块`} value={item.plot_id ?? ""}
                        disabled={Boolean(bindingProgress)}
                        onChange={(event) => bindProgress(item.id, event.target.value)}>
                        <option value="">未绑定（个人计划）</option>
                        {item.plot_id && !calendarFields.some((field) => String(field.id) === String(item.plot_id)) && (
                          <option value={item.plot_id}>已绑定地块不可用</option>
                        )}
                        {calendarFields.map((field) => <option key={field.id} value={field.id}>{field.name}</option>)}
                      </select>
                      {bindingProgress === item.id && <small>正在保存…</small>}
                    </label>
                    <div className="plan-line-progress">
                      <div>
                        <span>整季预计进度</span>
                        <b>{value}%</b>
                      </div>
                      <small className="section-hint">
                        {item.status === "待开始"
                          ? `计划 ${item.start_date || "日期待定"} 开始`
                          : `开始于 ${item.start_date || "未设置"}`}
                        {item.status !== "待开始" &&
                          ` · 本阶段 ${item.stage_progress ?? item.progress_percent ?? item.progress ?? 0}%`}
                      </small>
                      <div
                        className="stage-progress"
                        role="progressbar"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={value}
                      >
                        <i style={{ width: `${value}%` }} />
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          ) : (
            <Empty
              title="暂无种植时间线"
              body="添加种植进度后，这里会按作物分别展示当前阶段和完成度。"
            />
          )}
        </Card>
      </div>
      <div className="content-grid equal">
        <Card
          title="农事任务"
          action={<small>{updatedAt ? `${updatedAt} 更新` : "读取中"}</small>}
        >
          <div className="calendar-filters">
            <label>
              任务状态
              <select
                value={taskFilter}
                onChange={(event) =>
                  setTaskFilter(event.target.value as TaskFilter)
                }
              >
                <option value="active">待处理</option>
                <option value="running">执行中</option>
                <option value="completed">已完成</option>
                <option value="attention">异常／逾期</option>
                <option value="all">全部状态</option>
              </select>
            </label>
            <label>
              截止日期
              <input
                type="date"
                value={selectedDate}
                onChange={(event) => setSelectedDate(event.target.value)}
              />
            </label>
            {selectedDate && (
              <button
                className="text-button"
                onClick={() => setSelectedDate("")}
              >
                全部日期
              </button>
            )}
          </div>
          {visibleTasks.length ? (
            <div className="calendar-task-list">
              {visibleTasks.map((task) => (
                <div className="calendar-task-row" key={task.id}>
                  <span className={`priority ${task.priority}`} />
                  <div className="calendar-task-copy">
                    <button
                      className="calendar-task-title"
                      onClick={() => setDetailTaskId(task.id)}
                    >
                      {task.title}
                    </button>
                    <small>
                      {task.crop} ·{" "}
                      {task.end_date?.slice(0, 10) || "未设置日期"}
                    </small>
                  </div>
                  <StatusPill
                    tone={
                      task.status === "已完成"
                        ? "success"
                        : ["已逾期", "失败", "已中止"].includes(task.status)
                          ? "danger"
                          : "info"
                    }
                  >
                    {task.status}
                  </StatusPill>
                  <div className="calendar-task-actions">
                    {task.device_id && task.status !== "已完成" && (
                      <button
                        onClick={() => actOnTask(task.id, "execute")}
                        title={
                          task.retry_available ? "恢复中断任务" : "执行设备任务"
                        }
                        aria-label={`${task.retry_available ? "恢复中断任务" : "执行设备任务"}：${task.title}`}
                        disabled={
                          !!busyTask ||
                          (isTaskRunning(task) && !task.retry_available)
                        }
                      >
                        <Play />
                      </button>
                    )}
                    <button
                      onClick={() => openTask(task)}
                      disabled={!!busyTask || isTaskRunning(task)}
                      title="编辑与改期"
                      aria-label={`编辑任务：${task.title}`}
                    >
                      <Edit3 />
                    </button>
                    {task.status !== "已完成" && (
                      <button
                        onClick={() => actOnTask(task.id, "complete")}
                        disabled={!!busyTask || isTaskRunning(task)}
                        title={task.device_id ? "登记现场完成" : "标记完成"}
                        aria-label={`完成任务：${task.title}`}
                      >
                        <Check />
                      </button>
                    )}
                    <button
                      onClick={() => actOnTask(task.id, "delete")}
                      disabled={!!busyTask || isTaskRunning(task)}
                      title="删除"
                      aria-label={`删除任务：${task.title}`}
                    >
                      <Trash2 />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Empty
              title="没有符合条件的任务"
              body="可以切换任务状态或清除日期筛选，回查其他任务。"
            />
          )}
        </Card>
        <Card title="本季进度摘要">
          <div className="season-summary">
            <div>
              <span>
                <CalendarCheck />
              </span>
              <strong>{progress.length}</strong>
              <small>种植计划</small>
            </div>
            <div>
              <span>
                <Check />
              </span>
              <strong>
                {tasks.filter((task) => task.status === "已完成").length}
              </strong>
              <small>已完成任务</small>
            </div>
            <div>
              <span>
                <Sprout />
              </span>
              <strong>{new Set(progress.map((item) => item.crop)).size}</strong>
              <small>在种作物</small>
            </div>
          </div>
        </Card>
      </div>
      {showForm && (
        <div className="modal-backdrop">
          <form className="modal compact-modal" onSubmit={create}>
            <div className="modal-head">
              <h2>
                {showForm === "task"
                  ? editingTask
                    ? "编辑农事任务"
                    : "添加农事任务"
                  : "添加种植进度"}
              </h2>
              <button
                type="button"
                className="icon-button"
                onClick={() => setShowForm(null)}
                disabled={saving}
              >
                ×
              </button>
            </div>
            <label>
              作物
              <input
                value={form.crop}
                onChange={(e) => setForm({ ...form, crop: e.target.value })}
                required
              />
            </label>
            {showForm === "task" ? (
              <>
                <label>
                  任务标题
                  <input
                    value={form.title}
                    onChange={(e) =>
                      setForm({ ...form, title: e.target.value })
                    }
                    required
                  />
                </label>
                <label>
                  任务类型
                  <select
                    value={form.task_type}
                    onChange={(e) =>
                      setForm({ ...form, task_type: e.target.value })
                    }
                  >
                    {[
                      "浇水",
                      "施肥",
                      "除草",
                      "病虫害防治",
                      "修剪",
                      "播种",
                      "收获",
                      "其他",
                    ].map((item) => (
                      <option key={item}>{item}</option>
                    ))}
                  </select>
                </label>
                <label>
                  优先级
                  <select
                    value={form.priority}
                    onChange={(e) =>
                      setForm({ ...form, priority: e.target.value })
                    }
                  >
                    <option value="high">高</option>
                    <option value="medium">中</option>
                    <option value="low">低</option>
                  </select>
                </label>
                <label>
                  任务说明
                  <textarea
                    value={form.description || ""}
                    onChange={(event) =>
                      setForm({ ...form, description: event.target.value })
                    }
                    rows={3}
                  />
                </label>
                <label>
                  开始日期（可选）
                  <input
                    type="date"
                    value={form.start_date || ""}
                    onChange={(event) =>
                      setForm({ ...form, start_date: event.target.value })
                    }
                  />
                </label>
                <label>
                  截止日期
                  <input
                    type="date"
                    value={form.end_date || ""}
                    onChange={(e) =>
                      setForm({ ...form, end_date: e.target.value })
                    }
                  />
                </label>
              </>
            ) : (
              <>
                <label>
                  关联地块（可选）
                  <select value={form.plot_id || ""} onChange={(event) => setForm({ ...form, plot_id: event.target.value })}>
                    <option value="">未绑定（个人计划）</option>
                    {calendarFields.map((field) => <option key={field.id} value={field.id}>{field.name}</option>)}
                  </select>
                </label>
                <p className="section-hint">绑定后才能按地块分析生长阶段，未绑定仍可作为个人计划。</p>
                <label>
                  播种／开始日期
                  <input
                    type="date"
                    value={form.start_date || ""}
                    onChange={(event) =>
                      setForm({ ...form, start_date: event.target.value })
                    }
                    required
                  />
                </label>
                <label>
                  当前阶段
                  <input
                    value={form.stage}
                    onChange={(e) =>
                      setForm({ ...form, stage: e.target.value })
                    }
                  />
                </label>
                <label>
                  总阶段数
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={form.total_stages}
                    onChange={(e) =>
                      setForm({ ...form, total_stages: e.target.value })
                    }
                  />
                </label>
              </>
            )}
            {formError && <ErrorState message={formError} />}
            <button className="primary-button" disabled={saving}>
              <Save />
              {saving ? "正在保存" : "保存"}
            </button>
          </form>
        </div>
      )}
      {detailTask && !showForm && (
        <div className="modal-backdrop">
          <section
            className="modal compact-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="task-detail-title"
          >
            <div className="modal-head">
              <h2 id="task-detail-title">任务详情</h2>
              <button
                className="icon-button"
                onClick={() => setDetailTaskId("")}
                aria-label="关闭任务详情"
              >
                ×
              </button>
            </div>
            <h3>{detailTask.title}</h3>
            <p>{detailTask.description || "暂无补充说明"}</p>
            <dl className="task-detail-list">
              <dt>作物</dt>
              <dd>{detailTask.crop}</dd>
              <dt>状态</dt>
              <dd>{detailTask.status}</dd>
              <dt>开始日期</dt>
              <dd>{detailTask.start_date || "未设置"}</dd>
              <dt>截止日期</dt>
              <dd>{detailTask.end_date?.slice(0, 10) || "未设置"}</dd>
              <dt>优先级</dt>
              <dd>
                {{ high: "高", medium: "中", low: "低" }[detailTask.priority] ||
                  detailTask.priority}
              </dd>
              {detailTask.device_id && (
                <>
                  <dt>关联设备</dt>
                  <dd>{detailTask.device_name || detailTask.device_id}</dd>
                  <dt>设备动作</dt>
                  <dd>
                    {{ start: "启动", stop: "停止", set_param: "调整参数" }[
                      String(detailTask.device_command)
                    ] || detailTask.device_command}
                  </dd>
                  {detailTask.device_params?.duration != null && (
                    <>
                      <dt>运行时长</dt>
                      <dd>{detailTask.device_params.duration} 分钟</dd>
                    </>
                  )}
                </>
              )}
            </dl>
            {detailTask.retry_available ? (
              <>
                <Notice tone="danger">
                  上次下发未取得结果，可核对设备后恢复；系统会沿用原执行编号防止重复启动。
                </Notice>
                <button
                  className="primary-button"
                  disabled={!!busyTask}
                  onClick={() => actOnTask(detailTask.id, "execute")}
                >
                  <Play />
                  恢复中断任务
                </button>
              </>
            ) : isTaskRunning(detailTask) ? (
              <Notice>
                作业正在执行，结束后会自动更新状态；需要提前停止时请到设备中心操作。
              </Notice>
            ) : (
              <button
                className="primary-button"
                onClick={() => openTask(detailTask)}
              >
                <Edit3 />
                编辑与改期
              </button>
            )}
          </section>
        </div>
      )}
    </>
  );
}
