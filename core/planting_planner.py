"""
种植规划算法模块
功能：
- 根据用户输入生成个性化种植计划
- 多因素决策（地区、土壤、气候、目标）
- 风险评估与规避建议
- 阶段性任务规划
- 资源需求估算
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import dotenv

# 加载环境变量
dotenv.load_dotenv()
DEFAULT_KNOWLEDGE_DIR = os.getenv("AGRICULTURE_KNOWLEDGE_DIR", "agriculture_knowledge/crops")


@dataclass
class CropInfo:
    """作物信息数据类"""
    name: str
    aliases: List[str]
    suitable_regions: List[str]
    planting_seasons: Dict[str, Any]
    soil_requirements: Dict[str, Any]
    climate_requirements: Dict[str, Any]
    growth_stages: List[Dict[str, Any]]
    fertilization_guide: List[Dict[str, Any]]
    irrigation_guide: List[Dict[str, Any]]
    common_diseases: List[Dict[str, Any]]
    common_pests: List[Dict[str, Any]]
    yield_info: Dict[str, Any]


@dataclass
class PlantingPlan:
    """种植计划数据类"""
    crop: str
    region: str
    soil_type: str
    farm_size: float
    goals: List[str]
    schedule: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    resources: Dict[str, Any]
    expected_yield: str
    created_at: str


class CropDatabase:
    """作物数据库"""

    def __init__(self, knowledge_dir: str = None):
        self.knowledge_dir = knowledge_dir or DEFAULT_KNOWLEDGE_DIR
        self.crops = {}
        self._load_crops()

    def _load_crops(self):
        """加载所有作物知识"""
        if not os.path.exists(self.knowledge_dir):
            print(f"警告：作物知识库目录不存在: {self.knowledge_dir}")
            return

        for filename in os.listdir(self.knowledge_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.knowledge_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        crop_info = CropInfo(**data)
                        self.crops[data['crop_name']] = crop_info
                        # 同时添加别名映射
                        for alias in data.get('aliases', []):
                            self.crops[alias] = crop_info
                except Exception as e:
                    print(f"加载作物知识失败 {filename}: {e}")

    def get_crop(self, name: str) -> Optional[CropInfo]:
        """获取作物信息"""
        return self.crops.get(name)

    def get_all_crops(self) -> List[str]:
        """获取所有作物名称（去重）"""
        seen = set()
        result = []
        for crop in self.crops.values():
            if crop.name not in seen:
                result.append(crop.name)
                seen.add(crop.name)
        return result


class RegionDatabase:
    """地区气候数据库"""

    # 简化版地区气候数据
    REGION_CLIMATE = {
        "华北": {
            "climate_type": "温带季风气候",
            "avg_temp": "10-12℃",
            "frost_free_days": 180,
            "annual_rainfall": "500-800mm",
            "suitable_crops": ["小麦", "玉米", "棉花", "花生"],
            "planting_windows": {
                "冬小麦": "9月下旬-10月上旬",
                "春玉米": "4月中旬-5月上旬",
                "夏玉米": "6月上旬",
                "棉花": "4月中下旬"
            }
        },
        "东北": {
            "climate_type": "温带大陆性季风气候",
            "avg_temp": "4-8℃",
            "frost_free_days": 120,
            "annual_rainfall": "400-700mm",
            "suitable_crops": ["玉米", "大豆", "水稻", "春小麦"],
            "planting_windows": {
                "春玉米": "4月下旬-5月上旬",
                "大豆": "4月下旬-5月上旬",
                "水稻": "4月中旬-5月上旬",
                "春小麦": "3月下旬-4月上旬"
            }
        },
        "黄淮海": {
            "climate_type": "暖温带季风气候",
            "avg_temp": "13-15℃",
            "frost_free_days": 200,
            "annual_rainfall": "600-900mm",
            "suitable_crops": ["小麦", "玉米", "棉花", "花生"],
            "planting_windows": {
                "冬小麦": "10月上旬-10月中旬",
                "夏玉米": "6月上旬-6月中旬",
                "棉花": "4月中旬-4月下旬"
            }
        },
        "西北": {
            "climate_type": "温带大陆性气候",
            "avg_temp": "7-12℃",
            "frost_free_days": 150,
            "annual_rainfall": "200-400mm",
            "suitable_crops": ["小麦", "玉米", "棉花", "瓜果"],
            "planting_windows": {
                "春小麦": "3月中旬-4月上旬",
                "玉米": "4月中旬-5月上旬",
                "棉花": "4月中旬"
            }
        },
        "华东": {
            "climate_type": "亚热带季风气候",
            "avg_temp": "15-18℃",
            "frost_free_days": 230,
            "annual_rainfall": "1000-1500mm",
            "suitable_crops": ["水稻", "小麦", "油菜", "茶叶"],
            "planting_windows": {
                "早稻": "3月下旬-4月上旬",
                "晚稻": "6月下旬-7月上旬",
                "冬小麦": "10月下旬-11月上旬"
            }
        },
        "华南": {
            "climate_type": "热带-亚热带季风气候",
            "avg_temp": "20-24℃",
            "frost_free_days": 300,
            "annual_rainfall": "1500-2000mm",
            "suitable_crops": ["水稻", "甘蔗", "香蕉", "荔枝"],
            "planting_windows": {
                "早稻": "2月下旬-3月上旬",
                "晚稻": "7月下旬-8月上旬",
                "甘蔗": "2月-3月"
            }
        }
    }

    def get_region_info(self, region: str) -> Optional[Dict[str, Any]]:
        """获取地区气候信息"""
        # 精确匹配
        if region in self.REGION_CLIMATE:
            return self.REGION_CLIMATE[region]

        # 模糊匹配
        for key, value in self.REGION_CLIMATE.items():
            if key in region or region in key:
                return value

        return None


class PlantingPlanner:
    """种植规划器"""

    def __init__(self):
        self.crop_db = CropDatabase()
        self.region_db = RegionDatabase()

    def generate_plan(self, user_input: Dict[str, Any]) -> PlantingPlan:
        """
        生成种植计划

        Args:
            user_input: 包含以下字段的字典
                - region: 地区
                - soil_type: 土壤类型（可选）
                - farm_size: 农场面积（可选）
                - goals: 种植目标列表（可选）
                - experience: 种植经验（可选）
                - crop: 指定作物（可选，未指定则推荐）

        Returns:
            PlantingPlan: 种植计划对象
        """
        region = user_input.get("region", "")
        soil_type = user_input.get("soil_type", "")
        farm_size = user_input.get("farm_size", 1.0)
        goals = user_input.get("goals", [])
        specified_crop = user_input.get("crop", "")

        # 获取地区信息
        region_info = self.region_db.get_region_info(region)

        # 确定作物
        if specified_crop:
            crop = specified_crop
        else:
            # 根据地区和目标推荐作物
            crop = self._recommend_crop(region, goals, region_info)

        # 获取作物信息
        crop_info = self.crop_db.get_crop(crop)
        if not crop_info:
            crop_info = self._get_default_crop_info(crop)

        # 生成时间表
        schedule = self._generate_schedule(crop_info, region_info)

        # 生成阶段性任务
        tasks = self._generate_stage_tasks(crop_info, schedule.get("sowing_date"))

        # 风险评估
        risks = self._assess_risks(crop, region, crop_info, region_info)

        # 计算资源需求
        resources = self._calculate_resources(crop_info, farm_size)

        # 估算产量
        expected_yield = self._estimate_yield(crop_info, farm_size, region_info)

        return PlantingPlan(
            crop=crop,
            region=region,
            soil_type=soil_type,
            farm_size=farm_size,
            goals=goals,
            schedule=schedule,
            tasks=tasks,
            risks=risks,
            resources=resources,
            expected_yield=expected_yield,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _recommend_crop(self, region: str, goals: List[str],
                        region_info: Optional[Dict]) -> str:
        """根据地区和目标推荐作物"""
        if region_info:
            suitable_crops = region_info.get("suitable_crops", [])
            if suitable_crops:
                # 根据目标排序
                if "高产" in goals:
                    return suitable_crops[0]  # 通常第一个是高产作物
                elif "省工" in goals:
                    return "玉米" if "玉米" in suitable_crops else suitable_crops[0]
                else:
                    return suitable_crops[0]

        # 默认推荐
        return "小麦"

    def _generate_schedule(self, crop_info: CropInfo,
                           region_info: Optional[Dict]) -> Dict[str, Any]:
        """生成种植时间表"""
        schedule = {}

        # 确定播种时间
        planting_seasons = crop_info.planting_seasons
        if planting_seasons:
            # 选择第一个季节（简化处理）
            season_key = list(planting_seasons.keys())[0]
            season_info = planting_seasons[season_key]
            schedule["season"] = season_info.get("name", season_key)
            schedule["sowing_time"] = season_info.get("sowing_time", "")
            schedule["harvest_time"] = season_info.get("harvest_time", "")
            schedule["suitable_climate"] = season_info.get("suitable_climate", "")
            schedule["notes"] = season_info.get("notes", "")

        # 计算各阶段日期（简化版）
        if crop_info.growth_stages:
            stage_schedule = []
            current_date = datetime.now()

            for stage in crop_info.growth_stages:
                duration = stage.get("duration_days", 30)
                stage_schedule.append({
                    "stage": stage.get("stage", ""),
                    "start_date": current_date.strftime("%Y-%m-%d"),
                    "end_date": (current_date + timedelta(days=duration)).strftime("%Y-%m-%d"),
                    "duration_days": duration,
                    "key_tasks": stage.get("key_tasks", []),
                    "notes": stage.get("notes", "")
                })
                current_date += timedelta(days=duration)

            schedule["stages"] = stage_schedule
            schedule["total_duration"] = sum(s.get("duration_days", 0)
                                             for s in crop_info.growth_stages)

        return schedule

    def _generate_stage_tasks(self, crop_info: CropInfo,
                              sowing_date: Optional[str]) -> List[Dict[str, Any]]:
        """生成各阶段详细任务"""
        tasks = []

        if not crop_info.growth_stages:
            return tasks

        base_date = datetime.now()
        if sowing_date:
            try:
                base_date = datetime.strptime(sowing_date, "%Y-%m-%d")
            except:
                pass

        for stage in crop_info.growth_stages:
            duration = stage.get("duration_days", 30)

            # 获取该阶段的农事操作
            stage_tasks = []
            key_tasks = stage.get("key_tasks", [])

            for i, task in enumerate(key_tasks):
                task_date = base_date + timedelta(days=i * 3)  # 任务间隔3天
                stage_tasks.append({
                    "task": task,
                    "date": task_date.strftime("%Y-%m-%d"),
                    "priority": "高" if task in ["施肥", "浇水", "病虫害防治"] else "中"
                })

            tasks.append({
                "stage": stage.get("stage", ""),
                "start_date": base_date.strftime("%Y-%m-%d"),
                "end_date": (base_date + timedelta(days=duration)).strftime("%Y-%m-%d"),
                "tasks": stage_tasks,
                "notes": stage.get("notes", "")
            })

            base_date += timedelta(days=duration)

        return tasks

    def _assess_risks(self, crop: str, region: str,
                      crop_info: CropInfo,
                      region_info: Optional[Dict]) -> List[Dict[str, Any]]:
        """风险评估"""
        risks = []

        # 气候风险
        if region_info:
            rainfall = region_info.get("annual_rainfall", "")
            if "200-400" in rainfall and crop in ["水稻", "甘蔗"]:
                risks.append({
                    "type": "气候风险",
                    "description": f"{region}年降雨量较少，不适合种植{crop}",
                    "severity": "高",
                    "mitigation": "建议改种耐旱作物如玉米、小麦，或采用灌溉设施"
                })

        # 病虫害风险
        if crop_info.common_diseases:
            high_risk_diseases = [d for d in crop_info.common_diseases[:2]]
            risks.append({
                "type": "病虫害风险",
                "description": f"{crop}常见病害：" + "、".join([d["name"] for d in high_risk_diseases]),
                "severity": "中",
                "mitigation": "建议选用抗病品种，定期巡查，发病初期及时防治"
            })

        # 市场风险（简化）
        market_info = crop_info.yield_info
        if market_info:
            risks.append({
                "type": "市场风险",
                "description": "农产品价格存在波动",
                "severity": "低",
                "mitigation": "关注市场行情，适时销售；可考虑与收购商签订订单"
            })

        return risks

    def _calculate_resources(self, crop_info: CropInfo,
                            farm_size: float) -> Dict[str, Any]:
        """计算资源需求"""
        resources = {
            "seeds": {},
            "fertilizers": [],
            "irrigation": {},
            "labor": {}
        }

        # 种子需求（简化估算）
        if crop_info.name == "小麦":
            resources["seeds"] = {
                "amount": f"{15 * farm_size:.0f}-{20 * farm_size:.0f}斤",
                "variety": "优质高产品种"
            }
        elif crop_info.name == "玉米":
            resources["seeds"] = {
                "amount": f"{3 * farm_size:.0f}-{5 * farm_size:.0f}斤",
                "variety": "杂交种"
            }
        elif crop_info.name == "番茄":
            resources["seeds"] = {
                "amount": f"{10 * farm_size:.0f}克（育苗用）",
                "variety": "抗病高产品种"
            }

        # 肥料需求
        if crop_info.fertilization_guide:
            for fert in crop_info.fertilization_guide[:3]:  # 取前3个关键施肥期
                resources["fertilizers"].append({
                    "time": fert.get("time", ""),
                    "type": fert.get("type", ""),
                    "amount_per_mu": fert.get("amount", ""),
                    "total_amount": self._scale_by_area(fert.get("amount", ""), farm_size)
                })

        # 灌溉需求
        if crop_info.irrigation_guide:
            critical_irrigation = [i for i in crop_info.irrigation_guide
                                   if "关键" in i.get("purpose", "") or "高峰" in i.get("purpose", "")]
            resources["irrigation"] = {
                "critical_periods": [i.get("stage", "") for i in critical_irrigation[:2]],
                "total_water_estimate": f"{200 * farm_size:.0f}-{300 * farm_size:.0f}立方米"
            }

        # 人工需求（简化估算）
        resources["labor"] = {
            "total_days": f"{10 * farm_size:.0f}-{15 * farm_size:.0f}天/季",
            "peak_periods": ["播种期", "收获期"],
            "notes": "机械化程度高的可减少人工投入"
        }

        return resources

    def _scale_by_area(self, amount_str: str, farm_size: float) -> str:
        """根据面积缩放用量"""
        # 简化处理，实际需要更复杂的单位解析
        if "/亩" in amount_str or "亩" in amount_str:
            # 提取数字部分
            import re
            numbers = re.findall(r'\d+', amount_str)
            if numbers:
                num = int(numbers[0])
                scaled = num * farm_size
                return amount_str.replace(str(num), str(int(scaled)))
        return amount_str

    def _estimate_yield(self, crop_info: CropInfo, farm_size: float,
                        region_info: Optional[Dict]) -> str:
        """估算产量"""
        yield_info = crop_info.yield_info
        if not yield_info:
            return "未知"

        medium_yield = yield_info.get("medium_yield", "")
        if medium_yield:
            # 解析产量范围
            import re
            match = re.search(r'(\d+)-(\d+)kg/亩', medium_yield)
            if match:
                low, high = int(match.group(1)), int(match.group(2))
                total_low = int(low * farm_size)
                total_high = int(high * farm_size)
                return f"{total_low}-{total_high}kg（约{medium_yield}）"

        return yield_info.get("medium_yield", "中等产量")

    def _get_default_crop_info(self, crop_name: str) -> CropInfo:
        """获取默认作物信息（当知识库中没有时）"""
        return CropInfo(
            name=crop_name,
            aliases=[],
            suitable_regions=["全国"],
            planting_seasons={},
            soil_requirements={},
            climate_requirements={},
            growth_stages=[],
            fertilization_guide=[],
            irrigation_guide=[],
            common_diseases=[],
            common_pests=[],
            yield_info={"medium_yield": "请咨询当地农技部门"}
        )

    def format_plan_as_text(self, plan: PlantingPlan) -> str:
        """将种植计划格式化为文本"""
        text = f"🌾 **{plan.crop}种植规划方案**\n\n"

        # 基本信息
        text += f"📍 **种植地区**: {plan.region}\n"
        if plan.soil_type:
            text += f"🌍 **土壤类型**: {plan.soil_type}\n"
        text += f"📐 **种植面积**: {plan.farm_size}亩\n"
        if plan.goals:
            text += f"🎯 **种植目标**: {'、'.join(plan.goals)}\n"
        text += "\n"

        # 时间安排
        if plan.schedule:
            text += "📅 **种植时间表**:\n"
            if "sowing_time" in plan.schedule:
                text += f"  • 播种时间: {plan.schedule['sowing_time']}\n"
            if "harvest_time" in plan.schedule:
                text += f"  • 收获时间: {plan.schedule['harvest_time']}\n"
            if "total_duration" in plan.schedule:
                text += f"  • 全生育期: 约{plan.schedule['total_duration']}天\n"
            text += "\n"

        # 关键农事任务
        if plan.tasks:
            text += "📝 **关键农事安排**:\n"
            for i, stage in enumerate(plan.tasks[:4], 1):  # 显示前4个阶段
                text += f"\n  {i}. **{stage['stage']}** ({stage['start_date']} ~ {stage['end_date']})\n"
                for task in stage.get("tasks", [])[:3]:  # 每个阶段显示前3个任务
                    text += f"     - {task['task']}\n"
            text += "\n"

        # 资源需求
        if plan.resources:
            text += "💰 **资源需求估算**:\n"
            if "seeds" in plan.resources:
                seeds = plan.resources["seeds"]
                text += f"  • 种子: {seeds.get('amount', '')}\n"
            if "fertilizers" in plan.resources:
                text += "  • 肥料:\n"
                for fert in plan.resources["fertilizers"][:2]:
                    text += f"    - {fert.get('time', '')}: {fert.get('type', '')}\n"
            if "irrigation" in plan.resources:
                irrigation = plan.resources["irrigation"]
                text += f"  • 灌溉: 关键期{', '.join(irrigation.get('critical_periods', []))}\n"
            text += "\n"

        # 产量预期
        if plan.expected_yield:
            text += f"🌾 **预期产量**: {plan.expected_yield}\n\n"

        # 风险提示
        if plan.risks:
            text += "⚠️ **风险提醒**:\n"
            for risk in plan.risks:
                text += f"  • 【{risk['type']}】{risk['description']}\n"
                text += f"    缓解措施: {risk['mitigation']}\n"
            text += "\n"

        text += "💡 **温馨提示**: 以上规划仅供参考，请根据当地实际情况和农技部门指导进行调整。"

        return text


# 便捷函数
def generate_planting_plan(user_input: Dict[str, Any]) -> str:
    """便捷函数：直接生成种植计划文本"""
    planner = PlantingPlanner()
    plan = planner.generate_plan(user_input)
    return planner.format_plan_as_text(plan)


if __name__ == "__main__":
    # 测试
    test_input = {
        "region": "华北",
        "soil_type": "壤土",
        "farm_size": 5.0,
        "goals": ["高产"],
        "crop": "小麦"
    }

    result = generate_planting_plan(test_input)
    print(result)
