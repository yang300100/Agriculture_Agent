"""
简化版农业知识检索（无需Embeddings）
使用关键词匹配 + 简单相似度计算
支持动态作物发现和模糊匹配
"""

import os
import json
import re
from glob import glob
from typing import List, Dict, Any
from difflib import SequenceMatcher

import dotenv

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
dotenv.load_dotenv()
DEFAULT_KNOWLEDGE_DIR = os.getenv("AGRICULTURE_KNOWLEDGE_DIR", os.path.join(PROJECT_ROOT, "agriculture_knowledge/crops"))
DEFAULT_RAG_KNOWLEDGE_DIR = os.path.dirname(DEFAULT_KNOWLEDGE_DIR) if DEFAULT_KNOWLEDGE_DIR.endswith("/crops") else PROJECT_ROOT
FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")


class SimpleAgricultureRAG:
    """简化版农业知识RAG（无需Embeddings），支持动态作物发现"""

    def __init__(self, knowledge_dir: str = None):
        self.knowledge_dir = knowledge_dir or DEFAULT_RAG_KNOWLEDGE_DIR
        self.knowledge_base = []
        self.crop_keywords = {}  # 动态构建
        self._load_all_knowledge()
        self._build_crop_keywords()

    def _load_all_knowledge(self):
        """加载所有作物知识文件"""
        crops_dir = os.path.join(self.knowledge_dir, "crops")
        if not os.path.exists(crops_dir):
            print(f"警告: 知识库目录不存在 {crops_dir}")
            return

        for json_file in glob(os.path.join(crops_dir, "*.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_base.append({
                        "crop": data.get("crop_name", ""),
                        "aliases": data.get("aliases", []),
                        "data": data,
                        "file": os.path.basename(json_file)
                    })
            except Exception as e:
                print(f"加载失败 {json_file}: {e}")

        print(f"已加载 {len(self.knowledge_base)} 个作物知识")

    def _build_crop_keywords(self):
        """从知识库动态构建作物关键词映射（含常见变体）"""
        # 常见作物名称变体映射（用于模糊匹配）
        common_variants = {
            "小麦": ["麦子", "冬小麦", "春小麦", "冬麦", "春麦"],
            "玉米": ["苞米", "包谷", "棒子", "玉茭"],
            "番茄": ["西红柿", "洋柿子", "蕃茄"],
            "水稻": ["大米", "稻谷", "稻子", "粳稻", "籼稻", "杂交稻"],
            "大豆": ["黄豆", "毛豆", "青豆", "黑豆"],
            "棉花": ["棉", "棉花作物"],
            "土豆": ["马铃薯", "洋芋", "山药蛋", "地蛋"],
            "花生": ["落花生", "地豆"],
            "高粱": ["蜀黍", "茭子"],
            "谷子": ["小米", "粟"],
            "油菜": ["菜籽", "油菜籽"],
            "甘薯": ["红薯", "地瓜", "番薯", "白薯"],
            "甘蔗": ["糖蔗", "果蔗"],
            "烟草": ["烟叶", "烤烟"],
            "茶叶": ["茶", "茶树", "绿茶", "红茶"],
            "蔬菜": ["青菜", "白菜", "菠菜", "芹菜", "韭菜", "萝卜", "胡萝卜"],
        }

        self.crop_keywords = {}
        for item in self.knowledge_base:
            crop_name = item["crop"]
            aliases = item.get("aliases", [])
            keywords = [crop_name] + aliases
            # 合并通用变体
            if crop_name in common_variants:
                for v in common_variants[crop_name]:
                    if v not in keywords:
                        keywords.append(v)
            self.crop_keywords[crop_name] = keywords

        # 补充知识库中没有但常见变体表中有的作物
        for crop_name, variants in common_variants.items():
            if crop_name not in self.crop_keywords:
                self.crop_keywords[crop_name] = [crop_name] + variants

    def _extract_crop_from_query(self, query: str) -> List[str]:
        """从查询中提取作物名称，返回匹配的作物列表（按匹配长度降序）"""
        matches = []
        for crop, keywords in self.crop_keywords.items():
            for keyword in sorted(keywords, key=len, reverse=True):  # 长关键词优先
                if keyword in query:
                    matches.append((crop, len(keyword)))
                    break  # 已匹配该作物，跳到下一个作物

        # 按关键词长度降序排列（匹配更具体的作物优先）
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        return SequenceMatcher(None, text1, text2).ratio()

    def _extract_topic_from_query(self, query: str) -> str:
        """提取查询主题（扩展版）"""
        topics = {
            "播种": ["播种", "种植", "什么时候种", "几月份种", "时间", "季节", "下种", "育苗"],
            "收获": ["收获", "收割", "采摘", "成熟", "什么时候收", "采收期", "采收"],
            "施肥": ["施肥", "肥料", "追肥", "怎么施肥", "用什么肥", "多少肥", "营养"],
            "浇水": ["浇水", "灌溉", "水", "怎么浇水", "灌水", "排水"],
            "病虫害": ["病", "虫", "害", "防治", "农药", "打药", "杀虫", "杀菌", "虫害", "病害"],
            "土壤": ["土壤", "土", "地", "ph", "酸碱", "改良", "墒情"],
            "产量": ["产量", "亩产", "收成", "能产多少", "多少斤", "效益"],
            "气候": ["气候", "温度", "气温", "日照", "光照", "积温", "降水"],
            "储存": ["储存", "贮藏", "仓储", "保鲜", "保存", "入库"],
            "市场": ["价格", "行情", "卖钱", "销路", "市场", "售卖", "销售渠道"],
        }

        all_matches = []
        for topic, keywords in topics.items():
            for keyword in keywords:
                if keyword in query:
                    all_matches.append((topic, len(keyword)))

        if all_matches:
            all_matches.sort(key=lambda x: x[1], reverse=True)
            return all_matches[0][0]

        return "general"

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索农业知识（增强版：动态作物识别）

        Args:
            query: 查询问题
            k: 返回结果数量

        Returns:
            相关知识列表
        """
        results = []

        # 1. 识别作物（支持多作物匹配）
        crop_names = self._extract_crop_from_query(query)
        topic = self._extract_topic_from_query(query)

        print(f"查询: {query}")
        print(f"   识别作物: {crop_names or '未识别'}")
        print(f"   识别主题: {topic}")

        # 2. 查找对应作物知识
        if crop_names:
            primary_crop = crop_names[0]
            target_crop = next(
                (item["data"] for item in self.knowledge_base
                 if item["crop"] == primary_crop or primary_crop in item.get("aliases", [])),
                None
            )

            if target_crop:
                self._add_crop_results(results, target_crop, topic, query)
            else:
                # 作物识别到了但知识库中没有 → 返回所有作物基本信息作为参考
                for item in self.knowledge_base[:k]:
                    results.append(self._format_basic_info(item))
        else:
            # 没有识别到作物 → 返回所有作物的基本信息
            for item in self.knowledge_base[:k]:
                results.append(self._format_basic_info(item))

        return results[:k]

    def _add_crop_results(self, results: List[Dict], target_crop: Dict, topic: str, query: str):
        """根据主题添加作物知识结果"""
        crop_name = target_crop["crop_name"]

        # 播种/时间相关
        if topic in ("播种", "气候") or any(w in query for w in ["什么时候", "几月", "季节"]):
            seasons = target_crop.get("planting_seasons", {})
            if seasons:
                for season_key, season_info in seasons.items():
                    results.append({
                        "content": f"{crop_name} - {season_info.get('name', season_key)}:\n"
                                   f"播种时间: {season_info.get('sowing_time', '')}\n"
                                   f"收获时间: {season_info.get('harvest_time', '')}\n"
                                   f"适宜气候: {season_info.get('suitable_climate', '')}\n"
                                   f"备注: {season_info.get('notes', '')}",
                        "metadata": {"crop": crop_name, "type": "planting_time"},
                        "score": 0.95
                    })
            climate = target_crop.get("climate_requirements", {})
            if climate.get("temperature"):
                t = climate["temperature"]
                results.append({
                    "content": f"{crop_name}温度要求:\n发芽: {t.get('germination', '')}\n"
                               f"生长: {t.get('growth', '')}\n"
                               f"耐寒: {t.get('cold_resistance', '')}",
                    "metadata": {"crop": crop_name, "type": "climate"},
                    "score": 0.85
                })

        # 土壤相关
        elif topic == "土壤":
            soil = target_crop.get("soil_requirements", {})
            results.append({
                "content": f"{crop_name}的土壤要求:\n"
                           f"适宜土壤: {', '.join(soil.get('preferred_types', []))}\n"
                           f"pH范围: {soil.get('ph_range', '')}\n"
                           f"肥力要求: {soil.get('fertility', '')}\n"
                           f"排水要求: {soil.get('drainage', '')}",
                "metadata": {"crop": crop_name, "type": "soil"},
                "score": 0.9
            })

        # 施肥
        elif topic == "施肥":
            fertilization = target_crop.get("fertilization_guide", [])
            for fert in fertilization[:3]:
                results.append({
                    "content": f"{crop_name}施肥 - {fert.get('time', '')}:\n"
                               f"肥料类型: {fert.get('type', '')}\n"
                               f"用量: {fert.get('amount', '')}\n"
                               f"方法: {fert.get('method', '')}",
                    "metadata": {"crop": crop_name, "type": "fertilization"},
                    "score": 0.9
                })

        # 病虫害
        elif topic == "病虫害":
            diseases = target_crop.get("common_diseases", [])
            pests = target_crop.get("common_pests", [])
            for disease in diseases[:2]:
                results.append({
                    "content": f"{crop_name}病害 - {disease.get('name', '')}:\n"
                               f"症状: {disease.get('symptoms', '')}\n"
                               f"防治: {disease.get('prevention', '')}\n"
                               f"发生期: {disease.get('occurrence_stage', '')}",
                    "metadata": {"crop": crop_name, "type": "disease"},
                    "score": 0.9
                })
            for pest in pests[:2]:
                results.append({
                    "content": f"{crop_name}虫害 - {pest.get('name', '')}:\n"
                               f"危害: {pest.get('symptoms', '')}\n"
                               f"防治: {pest.get('control', '')}",
                    "metadata": {"crop": crop_name, "type": "pest"},
                    "score": 0.85
                })

        # 产量/收获
        elif topic in ("收获", "产量"):
            yield_info = target_crop.get("yield_info", {})
            results.append({
                "content": f"{crop_name}产量信息:\n"
                           f"低产: {yield_info.get('low_yield', '')}\n"
                           f"中产: {yield_info.get('medium_yield', '')}\n"
                           f"高产: {yield_info.get('high_yield', '')}\n"
                           f"影响因素: {', '.join(yield_info.get('factors', []))}",
                "metadata": {"crop": crop_name, "type": "yield"},
                "score": 0.9
            })
            # 也加上收获相关的市场信息
            market = target_crop.get("market_info", {})
            if market:
                results.append({
                    "content": f"{crop_name}市场信息:\n"
                               f"上市旺季: {market.get('peak_season', '')}\n"
                               f"价格因素: {market.get('price_factors', '')}\n"
                               f"储存提示: {market.get('storage_tips', '')}",
                    "metadata": {"crop": crop_name, "type": "market"},
                    "score": 0.8
                })

        # 储存/市场
        elif topic in ("储存", "市场"):
            market = target_crop.get("market_info", {})
            if market:
                results.append({
                    "content": f"{crop_name}市场与储存:\n"
                               f"上市旺季: {market.get('peak_season', '')}\n"
                               f"价格因素: {', '.join(market.get('price_factors', []))}\n"
                               f"储存注意: {market.get('storage_tips', '')}",
                    "metadata": {"crop": crop_name, "type": "market"},
                    "score": 0.9
                })

        # 浇水/灌溉
        elif topic == "浇水":
            irrigation = target_crop.get("irrigation_guide", [])
            for irr in irrigation[:4]:
                results.append({
                    "content": f"{crop_name}灌溉 - {irr.get('stage', '')}:\n"
                               f"目的: {irr.get('purpose', '')}\n"
                               f"水量: {irr.get('amount', '')}",
                    "metadata": {"crop": crop_name, "type": "irrigation"},
                    "score": 0.85
                })

        # 默认返回基本信息和生长阶段
        else:
            results.append(self._format_basic_info({"crop": crop_name, "data": target_crop}))
            stages = target_crop.get("growth_stages", [])
            if stages:
                stage_info = "生长阶段:\n"
                for stage in stages[:5]:
                    stage_info += f"  {stage.get('stage', '')}: 约{stage.get('duration_days', '')}天"
                    tasks = stage.get("key_tasks", [])
                    if tasks:
                        stage_info += f" — {', '.join(tasks[:2])}"
                    stage_info += "\n"
                results.append({
                    "content": stage_info,
                    "metadata": {"crop": crop_name, "type": "growth_stages"},
                    "score": 0.75
                })

    def _format_basic_info(self, item: Dict) -> Dict[str, Any]:
        """格式化作物基本信息"""
        crop_name = item.get("crop", item.get("data", {}).get("crop_name", ""))
        data = item.get("data", {})
        return {
            "content": f"作物: {crop_name}\n"
                       f"别名: {', '.join(data.get('aliases', []))}\n"
                       f"适宜地区: {', '.join(data.get('suitable_regions', []))}\n"
                       f"土壤要求: pH {data.get('soil_requirements', {}).get('ph_range', '')}",
            "metadata": {"crop": crop_name, "type": "general"},
            "score": 0.7
        }


# 便捷函数
def search_agriculture_knowledge(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """便捷函数：搜索农业知识"""
    rag = SimpleAgricultureRAG()
    return rag.search(query, k)


if __name__ == "__main__":
    # 测试
    rag = SimpleAgricultureRAG()

    test_queries = [
        "小麦什么时候播种",
        "玉米怎么施肥",
        "番茄病虫害怎么防治",
        "土豆适合什么土壤",
        "水稻产量一般多少",
        "大豆市场行情怎么样",
        "华北地区适合种什么",
        "棉花怎么管理",
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        results = rag.search(query)
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i} ({result['metadata'].get('type', '?')}):")
            print(result['content'][:200])
