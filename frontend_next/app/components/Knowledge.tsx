"use client";

import { FormEvent, useEffect, useState } from "react";
import {
  ArrowRight,
  Calculator,
  ChevronRight,
  FlaskConical,
  Leaf,
  Search,
  Sparkles,
  Sprout,
} from "lucide-react";
import { get, post } from "../api";
import {
  fertilizerContent,
  fertilizerNeed,
  goals,
  seedData,
  soils,
} from "../data";
import type { JsonMap, Profile } from "../types";
import {
  Card,
  Empty,
  ErrorState,
  Loading,
  Notice,
  PageHeader,
  StatusPill,
} from "./Common";
import { MarkdownContent } from "./MarkdownContent";

export function PolicyPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<JsonMap[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [searchError, setSearchError] = useState("");
  async function runPolicySearch(keyword: string) {
    const term = keyword.trim();
    if (!term) return;
    setQuery(term);
    setSearchError("");
    setLoading(true);
    setSearched(true);
    try {
      const data = await get<JsonMap | JsonMap[]>(
        `/api/policy/search?q=${encodeURIComponent(term)}`,
      );
      setResults(Array.isArray(data) ? data : data.results || []);
    } catch (reason) {
      setResults([]);
      setSearchError(
        reason instanceof Error ? reason.message : "政策检索暂时不可用",
      );
    } finally {
      setLoading(false);
    }
  }
  async function search(event: FormEvent) {
    event?.preventDefault();
    await runPolicySearch(query);
  }
  const quick = [
    "耕地地力保护补贴",
    "农机购置补贴",
    "高标准农田",
    "绿色种植",
    "农业保险",
    "种粮一次性补贴",
  ];
  return (
    <>
      <PageHeader
        eyebrow="POLICY INTELLIGENCE"
        title="政策补贴"
        description="用自然语言检索农业政策，快速找到申报条件、补贴对象和办理线索。"
      />
      <Card className="policy-search-card">
        <form className="knowledge-search" onSubmit={search}>
          <Search />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="例如：河北种小麦今年有哪些补贴？"
          />
          <button className="primary-button">搜索政策</button>
        </form>
        <div className="quick-tags">
          <span>热门搜索</span>
          {quick.map((item) => (
            <button key={item} onClick={() => runPolicySearch(item)}>
              {item}
            </button>
          ))}
        </div>
      </Card>
      {loading ? (
        <Loading label="正在检索政策知识库" />
      ) : searchError ? (
        <ErrorState message={searchError} />
      ) : results.length ? (
        <div className="policy-results">
          {results.map((item, index) => (
            <Card key={index}>
              <div className="result-meta">
                {item.score != null || item.relevance != null ? (
                  <StatusPill tone="success">
                    匹配{" "}
                    {Math.round(Number(item.score ?? item.relevance) * 100)}%
                  </StatusPill>
                ) : (
                  <StatusPill tone="success">官方来源</StatusPill>
                )}
                <span>{item.source || item.department || "农业政策库"}</span>
              </div>
              <h2>
                {item.title || item.policy_name || `政策结果 ${index + 1}`}
              </h2>
              <p>
                {item.content ||
                  item.summary ||
                  item.text ||
                  "该政策条目未提供摘要，请打开原文查看详细内容。"}
              </p>
              {item.published_at && <small>发布时间：{item.published_at}</small>}
              {item.url && (
                <a href={item.url} target="_blank">
                  查看原文 <ArrowRight />
                </a>
              )}
            </Card>
          ))}
        </div>
      ) : searched ? (
        <Empty
          title="没有找到匹配政策"
          body="尝试减少地区限制或改用更简短的补贴名称。"
        />
      ) : (
        <div className="policy-guide">
          <Card>
            <span className="guide-number">01</span>
            <h2>描述你的情况</h2>
            <p>地区、作物和经营规模越具体，匹配结果越准确。</p>
          </Card>
          <Card>
            <span className="guide-number">02</span>
            <h2>核对申报条件</h2>
            <p>关注申报主体、材料清单和截止时间。</p>
          </Card>
          <Card>
            <span className="guide-number">03</span>
            <h2>联系主管部门</h2>
            <p>政策结果用于辅助检索，正式申报前请确认最新口径。</p>
          </Card>
        </div>
      )}
    </>
  );
}

export function EncyclopediaPage({
  initialCrop = "",
}: {
  initialCrop?: string;
}) {
  const [crops, setCrops] = useState<Record<string, JsonMap>>({});
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState("");
  const [compare, setCompare] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    get<Record<string, JsonMap>>("/api/encyclopedia")
      .then((data) => {
        setCrops(data);
        setSelected(
          initialCrop && data[initialCrop]
            ? initialCrop
            : Object.keys(data)[0] || "",
        );
        setCompare(Object.keys(data)[1] || "");
      })
      .finally(() => setLoading(false));
    // 首次载入时使用全局搜索传入的作物，避免打开错误词条。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (initialCrop && crops[initialCrop]) setSelected(initialCrop);
  }, [crops, initialCrop]);
  if (loading) return <Loading label="正在打开作物百科" />;
  const names = Object.keys(crops).filter((name) =>
    name.includes(query.trim()),
  );
  const crop = crops[selected] || {};
  const other = crops[compare] || {};
  return (
    <>
      <PageHeader
        eyebrow="CROP LIBRARY"
        title="作物百科"
        description="从播种到收获，集中查看作物生长阶段、肥水需求和病虫害知识。"
        actions={
          <div className="inline-search">
            <Search />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="搜索作物"
            />
          </div>
        }
      />
      <div className="ency-layout">
        <Card className="crop-directory" title={`作物目录 · ${names.length}`}>
          {names.length ? (
            names.map((name) => (
              <button
                key={name}
                className={selected === name ? "active" : ""}
                onClick={() => setSelected(name)}
                aria-pressed={selected === name}
              >
                <span>{name.slice(0, 1)}</span>
                <b>{name}</b>
                <ChevronRight />
              </button>
            ))
          ) : (
            <Empty title="没有匹配作物" body="换一个更短的名称试试。" />
          )}
        </Card>
        <div className="crop-detail">
          <Card className="crop-hero">
            <div className="crop-hero-main">
              <span className="crop-big-mark">{selected.slice(0, 1)}</span>
              <div>
                <small>CROP KNOWLEDGE</small>
                <h2>{selected || "请选择作物"}</h2>
                <p>
                  {crop.description ||
                    crop.basic_info?.description ||
                    `${selected}适宜在${formatList(crop.suitable_regions) || "适宜区域"}种植，别名${formatList(crop.aliases) || "暂无"}。`}
                </p>
              </div>
            </div>
            <div className="crop-tags">
              {[
                { label: "作物类型", value: cropCategory(selected) },
                { label: "推荐播种", value: primarySowingSeason(crop) },
                { label: "参考周期", value: growthCycle(crop) },
              ].map((item) => (
                <span key={item.label}>
                  <small>{item.label}</small>
                  <b>{item.value}</b>
                </span>
              ))}
            </div>
          </Card>
          <div className="crop-info-grid">
            <InfoCard title="生长阶段" data={growthStageSummary(crop)} />
            <InfoCard title="施肥与灌溉" data={careGuideSummary(crop)} />
            <InfoCard title="常见病虫害" data={pestSummary(crop)} />
            <InfoCard title="产量与市场" data={yieldMarketSummary(crop)} />
          </div>
          <Card
            title="作物对比"
            action={
              <select
                value={compare}
                onChange={(e) => setCompare(e.target.value)}
              >
                {Object.keys(crops)
                  .filter((name) => name !== selected)
                  .map((name) => (
                    <option key={name}>{name}</option>
                  ))}
              </select>
            }
          >
            <div className="compare-table">
              <div>
                <span>项目</span>
                <b>{selected}</b>
                <b>{compare}</b>
              </div>
              {[
                ["生长周期", growthCycle(crop), growthCycle(other)],
                ["适宜土壤", soilSummary(crop), soilSummary(other)],
                ["参考产量", yieldSummary(crop), yieldSummary(other)],
                [
                  "播种季节",
                  plantingSeasonSummary(crop),
                  plantingSeasonSummary(other),
                ],
              ].map((row, index) => (
                <div key={index}>
                  <span>{row[0]}</span>
                  <p>{row[1]}</p>
                  <p>{row[2]}</p>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}

function InfoCard({ title, data }: { title: string; data: string }) {
  return (
    <Card>
      <div className="info-card-head">
        <span>
          <Leaf />
        </span>
        <h2>{title}</h2>
      </div>
      <p>{data}</p>
    </Card>
  );
}

function formatList(value: unknown) {
  return Array.isArray(value) ? value.map(String).join("、") : "";
}

function cropCategory(name: string) {
  if (["番茄", "马铃薯", "甘薯"].includes(name)) return "蔬菜与薯类";
  if (["茶", "烟草", "棉花", "甘蔗"].includes(name)) return "经济作物";
  if (["大豆", "花生", "油菜"].includes(name)) return "油料作物";
  return "粮食作物";
}

function growthCycle(data: JsonMap) {
  const stages = Array.isArray(data.growth_stages) ? data.growth_stages : [];
  const days = stages.reduce(
    (sum: number, stage: JsonMap) => sum + Number(stage.duration_days || 0),
    0,
  );
  return days ? `${days} 天 · ${stages.length} 个阶段` : "资料待补充";
}

function primarySowingSeason(data: JsonMap) {
  const seasons = Object.values(data.planting_seasons || {}) as JsonMap[];
  const first = seasons[0];
  return first?.sowing_time || "资料待补充";
}

function plantingSeasonSummary(data: JsonMap) {
  const seasons = Object.values(data.planting_seasons || {}) as JsonMap[];
  if (!seasons.length) return "资料待补充";
  return seasons
    .map(
      (item) =>
        `${item.name || "当季"}：${item.sowing_time || "播期待补充"}播种，${item.harvest_time || "收获期待补充"}收获`,
    )
    .join("；");
}

function soilSummary(data: JsonMap) {
  const soil = data.soil_requirements || data.soil || {};
  if (typeof soil === "string") return soil;
  const parts = [
    formatList(soil.preferred_types),
    soil.ph_range ? `pH ${soil.ph_range}` : "",
    soil.fertility ? `肥力：${soil.fertility}` : "",
    soil.drainage ? `排水：${soil.drainage}` : "",
    soil.notes || "",
  ].filter(Boolean);
  return parts.length ? parts.join("；") : "资料待补充";
}

function yieldSummary(data: JsonMap) {
  const info = data.yield_info || data.yield || {};
  if (typeof info === "string" || typeof info === "number") return String(info);
  return info.medium_yield || info.high_yield || info.low_yield || "资料待补充";
}

function growthStageSummary(data: JsonMap) {
  const stages = Array.isArray(data.growth_stages) ? data.growth_stages : [];
  if (!stages.length) return "生长阶段资料待补充";
  return stages
    .map(
      (item: JsonMap) =>
        `${item.stage || "阶段"}${item.duration_days ? `（约 ${item.duration_days} 天）` : ""}`,
    )
    .join(" · ");
}

function careGuideSummary(data: JsonMap) {
  const fertilizer = Array.isArray(data.fertilization_guide)
    ? data.fertilization_guide
    : [];
  const irrigation = Array.isArray(data.irrigation_guide)
    ? data.irrigation_guide
    : [];
  const parts = [
    ...fertilizer
      .slice(0, 3)
      .map(
        (item: JsonMap) =>
          `${item.time || "施肥"}：${item.type || "按需追肥"} ${item.amount || ""}`,
      ),
    ...irrigation
      .slice(0, 2)
      .map(
        (item: JsonMap) =>
          `${item.stage || "灌溉"}：${item.purpose || "补水"} ${item.amount || ""}`,
      ),
  ];
  return parts.length ? parts.join("；") : "肥水管理资料待补充";
}

function pestSummary(data: JsonMap) {
  const diseases = Array.isArray(data.common_diseases)
    ? data.common_diseases
    : [];
  const pests = Array.isArray(data.common_pests) ? data.common_pests : [];
  const diseaseNames = diseases
    .map((item: JsonMap) => item.name)
    .filter(Boolean);
  const pestNames = pests.map((item: JsonMap) => item.name).filter(Boolean);
  const parts = [
    diseaseNames.length ? `病害：${diseaseNames.join("、")}` : "",
    pestNames.length ? `虫害：${pestNames.join("、")}` : "",
  ].filter(Boolean);
  return parts.length ? parts.join("；") : "病虫害资料待补充";
}

function yieldMarketSummary(data: JsonMap) {
  const yieldInfo = data.yield_info || {};
  const market = data.market_info || {};
  const parts = [
    yieldInfo.medium_yield ? `中等产量 ${yieldInfo.medium_yield}` : "",
    yieldInfo.high_yield ? `高产参考 ${yieldInfo.high_yield}` : "",
    market.peak_season ? `集中上市 ${market.peak_season}` : "",
    market.storage_tips ? `储存：${market.storage_tips}` : "",
  ].filter(Boolean);
  return parts.length ? parts.join("；") : "产量与市场资料待补充";
}

export function CalculatorPage() {
  return (
    <>
      <PageHeader
        eyebrow="AGRI CALCULATOR"
        title="农资计算"
        description="把播种量、肥料折算和农药兑水从经验估算变成清晰数字。"
      />
      <div className="calculator-stack">
        <SeedCalculator />
        <FertilizerCalculator />
        <PesticideCalculator />
      </div>
    </>
  );
}

function safeCalculationValue(
  value: number,
  {
    min,
    max = Number.POSITIVE_INFINITY,
    fallback,
  }: { min: number; max?: number; fallback: number },
) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function SeedCalculator() {
  const [crop, setCrop] = useState("小麦");
  const [area, setArea] = useState(1);
  const reference = seedData[crop];
  const [weight, setWeight] = useState(reference.weight);
  const [germ, setGerm] = useState(reference.germ);
  const [plants, setPlants] = useState(reference.plants);
  const [result, setResult] = useState<{
    total: number;
    perMu: number;
    area: number;
    adjusted: boolean;
  } | null>(null);
  function changeCrop(value: string) {
    const next = seedData[value];
    setCrop(value);
    setWeight(next.weight);
    setGerm(next.germ);
    setPlants(next.plants);
    setResult(null);
  }
  function calculate(event: FormEvent) {
    event.preventDefault();
    const safeArea = safeCalculationValue(area, {
      min: 0.01,
      fallback: 1,
    });
    const safeWeight = safeCalculationValue(weight, {
      min: 0.01,
      fallback: reference.weight,
    });
    const safeGerm = safeCalculationValue(germ, {
      min: 0.01,
      max: 1,
      fallback: reference.germ,
    });
    const safePlants = safeCalculationValue(plants, {
      min: 0.001,
      fallback: reference.plants,
    });
    const perMu = (safePlants * 10000 * safeWeight) / (safeGerm * 1000 * 1000);
    setResult({
      total: perMu * safeArea,
      perMu,
      area: safeArea,
      adjusted:
        safeArea !== area ||
        safeWeight !== weight ||
        safeGerm !== germ ||
        safePlants !== plants,
    });
  }
  return (
    <form className="calculator-layout" onSubmit={calculate} noValidate>
      <Card
        title="播种量计算"
        action={
          <span className="calculator-kind">
            <Sprout /> 种子
          </span>
        }
      >
        <div className="form-grid two">
          <label>
            作物
            <select value={crop} onChange={(e) => changeCrop(e.target.value)}>
              {Object.keys(seedData).map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            种植面积（亩）
            <input
              type="number"
              step="any"
              value={area}
              onChange={(e) => setArea(Number(e.target.value))}
            />
          </label>
          <label>
            千粒重（克）
            <input
              type="number"
              step="any"
              value={weight}
              onChange={(e) => setWeight(Number(e.target.value))}
            />
          </label>
          <label>
            发芽率
            <input
              type="number"
              step="any"
              value={germ}
              onChange={(e) => setGerm(Number(e.target.value))}
            />
          </label>
          <label className="full">
            目标亩播量（万株/亩）
            <input
              type="number"
              step="any"
              value={plants}
              onChange={(e) => setPlants(Number(e.target.value))}
            />
          </label>
        </div>
        <button className="primary-button calculator-action">计算播种量</button>
      </Card>
      <Card className="calculation-result">
        {result ? (
          <>
            <small>建议用种量</small>
            <strong>
              {result.total.toFixed(2)} <em>kg</em>
            </strong>
            <div className="calculation-detail-row">
              <span>每亩用种</span>
              <b>{result.perMu.toFixed(2)} kg/亩</b>
            </div>
            <div className="calculation-detail-row">
              <span>种植面积</span>
              <b>{result.area} 亩</b>
            </div>
            <p>
              依据千粒重、目标株数和发芽率估算，实际播种请结合整地质量调整。
            </p>
            {result.adjusted && (
              <p className="calculation-adjustment">
                部分输入超出可计算范围，结果已按安全边界计算；输入框中的原值未被修改。
              </p>
            )}
          </>
        ) : (
          <Empty title="等待计算" body="填写参数后点击“计算播种量”。" />
        )}
      </Card>
    </form>
  );
}

function FertilizerCalculator() {
  const [crop, setCrop] = useState("小麦");
  const [area, setArea] = useState(1);
  const base = fertilizerNeed[crop];
  const [nType, setN] = useState("尿素");
  const [pType, setP] = useState("磷酸二铵");
  const [kType, setK] = useState("氯化钾");
  const [result, setResult] = useState<
    {
      label: string;
      value: number;
      total: number;
      area: number;
      adjusted: boolean;
    }[]
  >([]);
  function calculate(event: FormEvent) {
    event.preventDefault();
    const safeArea = safeCalculationValue(area, {
      min: 0.01,
      fallback: 1,
    });
    const rows = [
      { label: nType, value: base.N / (fertilizerContent[nType].N / 100) },
      { label: pType, value: base.P / (fertilizerContent[pType].P / 100) },
      { label: kType, value: base.K / (fertilizerContent[kType].K / 100) },
    ];
    setResult(
      rows.map((item) => ({
        ...item,
        total: item.value * safeArea,
        area: safeArea,
        adjusted: safeArea !== area,
      })),
    );
  }
  return (
    <form className="calculator-layout" onSubmit={calculate} noValidate>
      <Card
        title="施肥量折算"
        action={
          <span className="calculator-kind">
            <FlaskConical /> 肥料
          </span>
        }
      >
        <div className="form-grid two">
          <label>
            作物
            <select value={crop} onChange={(e) => setCrop(e.target.value)}>
              {Object.keys(fertilizerNeed).map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            种植面积（亩）
            <input
              type="number"
              step="any"
              value={area}
              onChange={(e) => setArea(Number(e.target.value))}
            />
          </label>
          <label>
            氮肥
            <select value={nType} onChange={(e) => setN(e.target.value)}>
              {["尿素", "磷酸二铵", "复合肥(15-15-15)"].map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            磷肥
            <select value={pType} onChange={(e) => setP(e.target.value)}>
              {["磷酸二铵", "过磷酸钙", "复合肥(15-15-15)"].map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            钾肥
            <select value={kType} onChange={(e) => setK(e.target.value)}>
              {["氯化钾", "硫酸钾", "复合肥(15-15-15)"].map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
        </div>
        <p className="section-hint">
          {crop} 目标产量 {base.yield}kg/亩，参考纯养分 N {base.N} / P {base.P}{" "}
          / K {base.K} kg/亩。
        </p>
        <button className="primary-button calculator-action">
          计算肥料用量
        </button>
      </Card>
      <Card className="calculation-result fertilizer-result">
        {result.length ? (
          <>
            <small>{result[0].area} 亩总用量</small>
            {result.map((item, index) => (
              <div className="fert-result" key={index}>
                <span>{item.label}</span>
                <strong>{item.total.toFixed(1)} kg</strong>
                <small>{item.value.toFixed(1)} kg/亩</small>
              </div>
            ))}
            <p>不同肥料可能重复提供养分，配方应用前建议结合土壤检测修正。</p>
            {result.some((item) => item.adjusted) && (
              <p className="calculation-adjustment">
                面积输入已在计算时按安全边界处理，输入框中的原值未被修改。
              </p>
            )}
          </>
        ) : (
          <Empty title="等待计算" body="选择肥料后点击“计算肥料用量”。" />
        )}
      </Card>
    </form>
  );
}

function PesticideCalculator() {
  const [ratio, setRatio] = useState(1000);
  const [water, setWater] = useState(15);
  const [dose, setDose] = useState(50);
  const [result, setResult] = useState<{
    ratioAmount: number;
    doseConcentration: number;
    water: number;
    ratio: number;
    adjusted: boolean;
  } | null>(null);
  function calculate(event: FormEvent) {
    event.preventDefault();
    const safeRatio = safeCalculationValue(ratio, {
      min: 1,
      fallback: 1000,
    });
    const safeWater = safeCalculationValue(water, {
      min: 0.01,
      fallback: 15,
    });
    const safeDose = safeCalculationValue(dose, {
      min: 0,
      fallback: 50,
    });
    setResult({
      ratioAmount: (safeWater * 1000) / safeRatio,
      doseConcentration: safeDose / safeWater,
      water: safeWater,
      ratio: safeRatio,
      adjusted: safeRatio !== ratio || safeWater !== water || safeDose !== dose,
    });
  }
  return (
    <form className="calculator-layout" onSubmit={calculate} noValidate>
      <Card
        title="农药稀释计算"
        action={
          <span className="calculator-kind">
            <Calculator /> 农药
          </span>
        }
      >
        <div className="form-grid two">
          <label>
            稀释倍数
            <input
              type="number"
              step="any"
              value={ratio}
              onChange={(e) => setRatio(Number(e.target.value))}
            />
          </label>
          <label>
            亩用药量（ml/g）
            <input
              type="number"
              step="any"
              value={dose}
              onChange={(e) => setDose(Number(e.target.value))}
            />
          </label>
          <label>
            用水量（升/亩）
            <input
              type="number"
              step="any"
              value={water}
              onChange={(e) => setWater(Number(e.target.value))}
            />
          </label>
        </div>
        <button className="primary-button calculator-action">
          计算稀释结果
        </button>
      </Card>
      <Card className="calculation-result">
        {result ? (
          <>
            <small>两种口径同时计算</small>
            <div className="pesticide-result-row">
              <span>{result.ratio} 倍液每亩取药</span>
              <b>{result.ratioAmount.toFixed(1)} ml</b>
            </div>
            <div className="pesticide-result-row">
              <span>按亩用量折算每升水加药</span>
              <b>{result.doseConcentration.toFixed(2)} ml</b>
            </div>
            <div className="calculation-detail-row">
              <span>总用水量</span>
              <b>{result.water} 升/亩</b>
            </div>
            <p>请以药剂标签推荐浓度为准，佩戴防护用品并避免逆风施药。</p>
            {result.adjusted && (
              <p className="calculation-adjustment">
                部分输入已在计算时按安全边界处理，输入框中的原值未被修改。
              </p>
            )}
          </>
        ) : (
          <Empty title="等待计算" body="填写参数后点击“计算稀释结果”。" />
        )}
      </Card>
    </form>
  );
}

export function WizardPage({
  onNavigate,
}: {
  onNavigate: (page: string) => void;
}) {
  const [crops, setCrops] = useState<Record<string, JsonMap>>({});
  const [profile, setProfile] = useState<Profile | null>(null);
  const [crop, setCrop] = useState("");
  const [region, setRegion] = useState("");
  const [soil, setSoil] = useState("壤土");
  const [area, setArea] = useState(1);
  const [selectedGoals, setSelectedGoals] = useState<string[]>([]);
  const [result, setResult] = useState<JsonMap | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([
      get<Record<string, JsonMap>>("/api/encyclopedia"),
      get<Profile>("/api/profile"),
    ])
      .then(([cropData, user]) => {
        setCrops(cropData);
        setProfile(user);
        setCrop(Object.keys(cropData)[0] || "");
        setRegion(user.user_region || "");
        setSoil(user.user_soil_type || "壤土");
        setArea(user.user_farm_size || 1);
        setSelectedGoals(user.user_goals || []);
      })
      .catch((reason) => setError(reason.message));
  }, []);
  async function generate() {
    setLoading(true);
    setError("");
    try {
      setResult(
        await post("/api/plan", {
          crop,
          region,
          soil_type: soil,
          farm_size: area,
          goals: selectedGoals,
          experience: profile?.user_experience || "",
        }),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "生成失败");
    } finally {
      setLoading(false);
    }
  }
  if (!Object.keys(crops).length && !error)
    return <Loading label="正在准备种植向导" />;
  return (
    <>
      <PageHeader
        eyebrow="PLANTING WIZARD"
        title="种植向导"
        description="确认作物和农场条件，一次生成计划、进度、任务与提醒。"
      />
      {error && <ErrorState message={error} />}{" "}
      {!result ? (
        <div className="wizard-layout">
          <div className="wizard-steps">
            <div className="active">
              <span>1</span>
              <b>选择作物</b>
            </div>
            <i />
            <div className={crop ? "active" : ""}>
              <span>2</span>
              <b>确认条件</b>
            </div>
            <i />
            <div>
              <span>3</span>
              <b>生成方案</b>
            </div>
          </div>
          <div className="wizard-form-panel">
            <Card title="第一步 · 选择本季作物">
              <label className="wizard-crop-input">
                作物名称
                <input
                  list="wizard-crop-options"
                  value={crop}
                  onChange={(event) => setCrop(event.target.value)}
                  placeholder="可搜索知识库，也可直接输入其他作物"
                />
                <datalist id="wizard-crop-options">
                  {Object.keys(crops).map((name) => (
                    <option key={name} value={name} />
                  ))}
                </datalist>
              </label>
              <p className="section-hint">
                可从知识库快捷选择，也可以直接输入尚未收录的作物名称。
              </p>
              <div className="crop-picker">
                {Object.keys(crops).map((name) => (
                  <button
                    type="button"
                    className={crop === name ? "selected" : ""}
                    key={name}
                    onClick={() => setCrop(name)}
                    aria-pressed={crop === name}
                  >
                    <Sprout />
                    <b>{name}</b>
                  </button>
                ))}
              </div>
            </Card>
            <Card title="第二步 · 确认农场条件">
              <div className="form-grid three">
                <label>
                  所在地区
                  <input
                    value={region}
                    onChange={(e) => setRegion(e.target.value)}
                  />
                </label>
                <label>
                  土壤类型
                  <select
                    value={soil}
                    onChange={(e) => setSoil(e.target.value)}
                  >
                    {soils.map((item) => (
                      <option key={item}>{item}</option>
                    ))}
                  </select>
                </label>
                <label>
                  面积（亩）
                  <input
                    type="number"
                    min=".1"
                    step=".5"
                    value={area}
                    onChange={(e) => setArea(Number(e.target.value))}
                  />
                </label>
              </div>
              <div className="choice-group">
                <span>种植目标</span>
                <div>
                  {goals.map((goal) => (
                    <button
                      type="button"
                      className={selectedGoals.includes(goal) ? "selected" : ""}
                      onClick={() =>
                        setSelectedGoals(
                          selectedGoals.includes(goal)
                            ? selectedGoals.filter((item) => item !== goal)
                            : [...selectedGoals, goal],
                        )
                      }
                      key={goal}
                      aria-pressed={selectedGoals.includes(goal)}
                    >
                      {goal}
                    </button>
                  ))}
                </div>
              </div>
            </Card>
            <div className="wizard-submit">
              <div>
                <b>生成种植报告</b>
                <small>同时创建进度、任务和农事提醒</small>
              </div>
              <button
                className="primary-button"
                onClick={generate}
                disabled={!crop.trim() || loading}
              >
                <Sparkles />
                {loading ? "正在生成…" : `为 ${crop || "当前作物"} 生成报告`}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="plan-result">
          <Notice>方案生成完成，相关进度、任务和提醒已写入农场工作台。</Notice>
          <Card className="plan-document">
            <div className="plan-title">
              <span>
                <Sprout />
              </span>
              <div>
                <small>PLANTING PLAN</small>
                <h2>
                  {crop} · {region}种植方案
                </h2>
              </div>
            </div>
            <article>
              <MarkdownContent
                content={String(result.plan_text || "方案已生成")}
              />
            </article>
            <div className="plan-created">
              <span>
                <b>1</b>条进度
              </span>
              <span>
                <b>{result.task_count || 0}</b>项任务
              </span>
              <span>
                <b>{result.reminder_count || 0}</b>条提醒
              </span>
            </div>
          </Card>
          <div className="result-actions">
            <button
              className="secondary-button"
              onClick={() => setResult(null)}
            >
              重新生成
            </button>
            <button
              className="primary-button"
              onClick={() => onNavigate("calendar")}
            >
              查看农事日历 <ArrowRight />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
