"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  CalendarCheck,
  Check,
  CircleDollarSign,
  Download,
  Edit3,
  Plus,
  RefreshCw,
  Save,
  Sprout,
  Trash2,
  TrendingDown,
  TrendingUp,
  UserRound,
} from "lucide-react";
import { get, post, remove } from "../api";
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
  useEffect(() => {
    get<Profile>("/api/profile")
      .then((data) => {
        setProfile({ ...blankProfile, ...data });
        const stored = localStorage.getItem("agri_autonomy_level");
        const next = data.autonomy_level || stored;
        if (next === "low" || next === "medium" || next === "high") {
          setAutonomy(next);
        }
      })
      .finally(() => setLoading(false));
  }, []);
  async function save(event: FormEvent) {
    event.preventDefault();
    await post("/api/profile", { ...profile, autonomy_level: autonomy });
    localStorage.setItem("agri_autonomy_level", autonomy);
    setMessage("种植档案已更新");
    setTimeout(() => setMessage(""), 2500);
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
              <span>当前选择</span>
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
                  body: "所有设备操作都需要确认",
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
          <button className="primary-button save-profile">
            <Save />
            保存全部修改
          </button>
        </div>
      </form>
    </>
  );
}

export function FieldsPage() {
  const [fields, setFields] = useState<Field[]>([]);
  const [weather, setWeather] = useState<Record<string, JsonMap>>({});
  const [showCreate, setShowCreate] = useState(false);
  const [selected, setSelected] = useState<Field | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    try {
      const rows = await get<Field[]>("/api/fields");
      setFields(rows);
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
            setWeather((current) => ({ ...current, [field.id]: data }));
          } catch {}
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
            onClick={() => setShowCreate(true)}
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
                      ? `${weather[field.id].temp}°C`
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
                onClick={() => setShowCreate(true)}
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
  onClose,
  onSaved,
}: {
  onClose: () => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState("");
  const [soil, setSoil] = useState(soils[0] || "壤土");
  const [crop, setCrop] = useState("");
  const [points, setPoints] = useState<number[][]>([]);
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
          <h2>绘制新地块</h2>
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
    load().catch(() => null);
  }, [load]);
  async function save(event: FormEvent) {
    event.preventDefault();
    if (type === "cost")
      await post("/api/finance/costs", {
        crop,
        cost_type: category,
        item_name: item || category,
        unit_price: amount,
      });
    else
      await post("/api/finance/income", {
        crop,
        quantity: 1,
        unit_price: amount,
      });
    setMessage(`已记录${type === "cost" ? "成本" : "收入"} ¥${amount}`);
    setAmount(0);
    await load();
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
    .sort((a, b) => String(b.date || "").localeCompare(String(a.date || "")))
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
      <div className="metric-grid three">
        <Card className="finance-metric income">
          <span>
            <TrendingUp />
          </span>
          <div>
            <small>累计收入</small>
            <strong>¥{totalIncome.toLocaleString()}</strong>
          </div>
        </Card>
        <Card className="finance-metric cost">
          <span>
            <TrendingDown />
          </span>
          <div>
            <small>累计成本</small>
            <strong>¥{totalCost.toLocaleString()}</strong>
          </div>
        </Card>
        <Card className="finance-metric profit">
          <span>
            <CircleDollarSign />
          </span>
          <div>
            <small>净收益</small>
            <strong>¥{(totalIncome - totalCost).toLocaleString()}</strong>
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
              >
                成本支出
              </button>
              <button
                type="button"
                className={type === "income" ? "active" : ""}
                onClick={() => setType("income")}
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
                required
              />
            </label>
            {type === "cost" && (
              <>
                <label>
                  成本类型
                  <select
                    value={category}
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
                  min="0"
                  value={amount || ""}
                  onChange={(e) => setAmount(Number(e.target.value))}
                  required
                />
              </div>
            </label>
            <button className="primary-button">
              <Save />
              保存记录
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
                  <span>¥{Number(row.total_income || 0).toFixed(0)}</span>
                  <span>¥{Number(row.total_cost || 0).toFixed(0)}</span>
                  <strong
                    className={
                      Number(row.net_profit || 0) >= 0 ? "positive" : "negative"
                    }
                  >
                    ¥{Number(row.net_profit || 0).toFixed(0)}
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
  const [showForm, setShowForm] = useState<"task" | "progress" | null>(null);
  const [formError, setFormError] = useState("");
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<JsonMap>({
    crop: "",
    title: "",
    task_type: "浇水",
    priority: "medium",
    stage: "准备期",
    total_stages: 5,
  });
  const load = useCallback(async () => {
    const [taskRows, progressRows] = await Promise.all([
      get<Task[]>("/api/tasks"),
      get<Progress[]>("/api/progress"),
    ]);
    setTasks(taskRows);
    setProgress(progressRows);
  }, []);
  useEffect(() => {
    load().catch(() => null);
  }, [load]);
  async function create(event: FormEvent) {
    event.preventDefault();
    if (saving) return;
    setFormError("");
    setSaving(true);
    try {
      const result =
        showForm === "task"
          ? await post<JsonMap>("/api/tasks", {
              crop: form.crop,
              title: form.title,
              task_type: form.task_type,
              description: `${form.task_type}任务`,
              priority: form.priority,
              status: "待办",
              end_date: form.end_date || "",
              progress_percent: 0,
            })
          : await post<JsonMap>("/api/progress", {
              crop: form.crop,
              stage: form.stage,
              total_stages: Number(form.total_stages),
              status: "进行中",
              start_date: new Date().toISOString().slice(0, 10),
            });
      if (result.success === false) {
        throw new Error(result.error || "保存失败，请稍后重试");
      }
      await load();
      setShowForm(null);
    } catch (reason) {
      setFormError(
        reason instanceof Error ? reason.message : "保存失败，请稍后重试",
      );
    } finally {
      setSaving(false);
    }
  }
  async function complete(id: string) {
    await post(`/api/tasks/${id}/complete`);
    await load();
  }
  async function deleteTask(id: string) {
    await remove(`/api/tasks/${id}`);
    await load();
  }
  const active = tasks
    .filter((task) => task.status !== "已完成")
    .sort((a, b) =>
      String(a.end_date || "9999").localeCompare(String(b.end_date || "9999")),
    );
  const days = Array.from({ length: 14 }, (_, index) => {
    const date = new Date();
    date.setDate(date.getDate() + index);
    return date;
  });
  const dateKey = (date: Date) =>
    `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
  const todayKey = dateKey(new Date());
  const progressValue = (item: Progress) =>
    Math.min(
      100,
      Math.max(0, Number(item.progress_percent || item.progress || 0)),
    );
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
                setShowForm("progress");
              }}
            >
              <Sprout />
              添加进度
            </button>
            <button
              className="primary-button"
              onClick={() => {
                setFormError("");
                setShowForm("task");
              }}
            >
              <Plus />
              添加任务
            </button>
          </>
        }
      />
      <div className="calendar-workspace">
        <Card
          className="calendar-board"
          title="未来 14 天"
          action={<span className="calendar-range">任务截止日一览</span>}
        >
          <div className="calendar-days">
            {days.map((day) => {
              const currentDateKey = dateKey(day);
              const dayTasks = active.filter(
                (task) => task.end_date?.slice(0, 10) === currentDateKey,
              );
              return (
                <div
                  className={currentDateKey === todayKey ? "today" : ""}
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
                </div>
              );
            })}
          </div>
        </Card>
        <Card className="growth-timeline-card" title="本季种植时间线">
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
                    <div className="plan-line-progress">
                      <div>
                        <span>计划完成度</span>
                        <b>{value}%</b>
                      </div>
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
        <Card title="近期任务">
          {active.length ? (
            <div className="calendar-task-list">
              {active.map((task) => (
                <div className="calendar-task-row" key={task.id}>
                  <span className={`priority ${task.priority}`} />
                  <div className="calendar-task-copy">
                    <b>{task.title}</b>
                    <small>
                      {task.crop} ·{" "}
                      {task.end_date?.slice(0, 10) || "未设置日期"}
                    </small>
                  </div>
                  <StatusPill
                    tone={task.status === "已逾期" ? "danger" : "info"}
                  >
                    {task.status}
                  </StatusPill>
                  <div className="calendar-task-actions">
                    <button
                      onClick={() => complete(task.id)}
                      title="标记完成"
                      aria-label={`完成任务：${task.title}`}
                    >
                      <Check />
                    </button>
                    <button
                      onClick={() => deleteTask(task.id)}
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
            <Empty title="没有待办任务" body="今天可以专注巡视和观察。" />
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
              <h2>{showForm === "task" ? "添加农事任务" : "添加种植进度"}</h2>
              <button
                type="button"
                className="icon-button"
                onClick={() => setShowForm(null)}
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
    </>
  );
}
