import {
  BookOpen,
  Bot,
  CalendarDays,
  Calculator,
  CircleUserRound,
  ClipboardList,
  Coins,
  FileText,
  LayoutDashboard,
  Map,
  MessageCircle,
  ScrollText,
  Sprout,
} from "lucide-react";

export const navItems = [
  { id: "dashboard", label: "农场概览", short: "概览", icon: LayoutDashboard },
  { id: "chat", label: "智能对话", short: "对话", icon: MessageCircle },
  { id: "profile", label: "基本信息", short: "档案", icon: CircleUserRound },
  { id: "fields", label: "地块管理", short: "地块", icon: Map },
  { id: "finance", label: "财务管理", short: "财务", icon: Coins },
  { id: "calendar", label: "农事日历", short: "日历", icon: CalendarDays },
  { id: "policy", label: "政策补贴", short: "政策", icon: ScrollText },
  { id: "encyclopedia", label: "作物百科", short: "百科", icon: BookOpen },
  { id: "calculator", label: "农资计算", short: "计算", icon: Calculator },
  { id: "wizard", label: "种植向导", short: "向导", icon: Sprout },
  { id: "devices", label: "设备中心", short: "设备", icon: Bot },
  { id: "rules", label: "规则管理", short: "规则", icon: ClipboardList },
  { id: "docs", label: "文档中心", short: "文档", icon: FileText },
] as const;

export const goals = [
  "高产",
  "优质",
  "省工",
  "节水",
  "有机",
  "多样化种植",
  "经济效益",
  "自用为主",
];
export const soils = ["壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"];
export const experienceLevels = [
  "新手（1年以下）",
  "初级（1-3年）",
  "中级（3-5年）",
  "高级（5-10年）",
  "专家（10年以上）",
];

export const seedData: Record<
  string,
  { weight: number; germ: number; plants: number }
> = {
  小麦: { weight: 40, germ: 0.9, plants: 15 },
  玉米: { weight: 300, germ: 0.92, plants: 2.5 },
  水稻: { weight: 28, germ: 0.9, plants: 3 },
  大豆: { weight: 180, germ: 0.88, plants: 4 },
  棉花: { weight: 100, germ: 0.85, plants: 1.5 },
  花生: { weight: 500, germ: 0.9, plants: 10 },
  油菜: { weight: 3.5, germ: 0.85, plants: 0.3 },
  谷子: { weight: 3, germ: 0.85, plants: 0.5 },
  高粱: { weight: 28, germ: 0.88, plants: 1.5 },
  番茄: { weight: 3, germ: 0.85, plants: 0.02 },
};

export const fertilizerNeed: Record<
  string,
  { N: number; P: number; K: number; yield: number }
> = {
  小麦: { N: 15, P: 6, K: 8, yield: 500 },
  玉米: { N: 18, P: 7, K: 10, yield: 600 },
  水稻: { N: 14, P: 5, K: 8, yield: 500 },
  大豆: { N: 5, P: 5, K: 6, yield: 200 },
  棉花: { N: 18, P: 7, K: 12, yield: 300 },
  花生: { N: 10, P: 6, K: 10, yield: 350 },
  油菜: { N: 12, P: 5, K: 8, yield: 200 },
};

export const fertilizerContent: Record<
  string,
  { N: number; P: number; K: number }
> = {
  尿素: { N: 46, P: 0, K: 0 },
  磷酸二铵: { N: 18, P: 46, K: 0 },
  氯化钾: { N: 0, P: 0, K: 60 },
  硫酸钾: { N: 0, P: 0, K: 50 },
  "复合肥(15-15-15)": { N: 15, P: 15, K: 15 },
  过磷酸钙: { N: 0, P: 16, K: 0 },
};
