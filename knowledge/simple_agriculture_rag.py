"""
简化版农业知识检索（无需Embeddings）
使用关键词匹配 + 简单相似度计算
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
# 使用上级目录作为知识库根目录
DEFAULT_RAG_KNOWLEDGE_DIR = os.path.dirname(DEFAULT_KNOWLEDGE_DIR) if DEFAULT_KNOWLEDGE_DIR.endswith("/crops") else PROJECT_ROOT


class SimpleAgricultureRAG:
    """简化版农业知识RAG"""

    def __init__(self, knowledge_dir: str = None):
        self.knowledge_dir = knowledge_dir or DEFAULT_RAG_KNOWLEDGE_DIR
        self.knowledge_base = []
        self.crop_keywords = {
            "小麦": ["小麦", "麦子", "冬小麦", "春小麦"],
            "玉米": ["玉米", "苞米", "包谷", "棒子"],
            "番茄": ["番茄", "西红柿", "洋柿子"],
            "水稻": ["水稻", "大米", "稻谷"],
            "大豆": ["大豆", "黄豆", "毛豆"],
            "棉花": ["棉花", "棉"],
            "土豆": ["土豆", "马铃薯", "洋芋"]
        }
        self._load_all_knowledge()

    def _load_all_knowledge(self):
        """加载所有知识文件"""
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

    def _extract_crop_from_query(self, query: str) -> str:
        """从查询中提取作物名称"""
        query_lower = query.lower()

        # 检查所有作物关键词
        for crop, keywords in self.crop_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    return crop

        # 检查知识库中的别名
        for item in self.knowledge_base:
            for alias in item.get("aliases", []):
                if alias in query:
                    return item["crop"]

        return ""

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        return SequenceMatcher(None, text1, text2).ratio()

    def _extract_topic_from_query(self, query: str) -> str:
        """提取查询主题"""
        topics = {
            "播种": ["播种", "种植", "什么时候种", "几月份种", "时间"],
            "收获": ["收获", "收割", "采摘", "成熟", "什么时候收"],
            "施肥": ["施肥", "肥料", "追肥", "怎么施肥"],
            "浇水": ["浇水", "灌溉", "水", "怎么浇水"],
            "病虫害": ["病", "虫", "害", "防治", "农药", "打药"],
            "土壤": ["土壤", "土", "地", "ph"],
            "产量": ["产量", "亩产", "收成", "能产多少"]
        }

        for topic, keywords in topics.items():
            for keyword in keywords:
                if keyword in query:
                    return topic

        return "general"

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索农业知识

        Args:
            query: 查询问题
            k: 返回结果数量

        Returns:
            相关知识列表
        """
        results = []

        # 1. 识别作物
        crop_name = self._extract_crop_from_query(query)
        topic = self._extract_topic_from_query(query)

        print(f"查询: {query}")
        print(f"   识别作物: {crop_name or '未识别'}")
        print(f"   识别主题: {topic}")

        # 2. 查找对应作物
        target_crop = None
        for item in self.knowledge_base:
            if item["crop"] == crop_name or crop_name in item["aliases"]:
                target_crop = item["data"]
                break

        if not target_crop:
            # 如果没有识别到作物，返回所有作物的基本信息
            for item in self.knowledge_base[:k]:
                results.append({
                    "content": f"作物: {item['crop']}\n"
                               f"适宜地区: {', '.join(item['data'].get('suitable_regions', [])[:3])}\n"
                               f"简介: {item['data'].get('aliases', [])}",
                    "metadata": {"crop": item["crop"], "type": "basic_info"},
                    "score": 0.5
                })
            return results

        # 3. 根据主题提取相关知识
        if topic == "播种" or "时间" in query or "什么时候" in query:
            # 返回种植季节信息
            seasons = target_crop.get("planting_seasons", {})
            for season_key, season_info in seasons.items():
                results.append({
                    "content": f"{target_crop['crop_name']} - {season_info.get('name', season_key)}:\n"
                               f"播种时间: {season_info.get('sowing_time', '')}\n"
                               f"收获时间: {season_info.get('harvest_time', '')}\n"
                               f"适宜气候: {season_info.get('suitable_climate', '')}\n"
                               f"备注: {season_info.get('notes', '')}",
                    "metadata": {"crop": target_crop["crop_name"], "type": "planting_time"},
                    "score": 0.95
                })

        elif topic == "土壤":
            soil = target_crop.get("soil_requirements", {})
            results.append({
                "content": f"{target_crop['crop_name']}的土壤要求:\n"
                           f"适宜土壤: {', '.join(soil.get('preferred_types', []))}\n"
                           f"pH范围: {soil.get('ph_range', '')}\n"
                           f"肥力要求: {soil.get('fertility', '')}",
                "metadata": {"crop": target_crop["crop_name"], "type": "soil"},
                "score": 0.9
            })

        elif topic == "施肥":
            fertilization = target_crop.get("fertilization_guide", [])
            for i, fert in enumerate(fertilization[:3]):
                results.append({
                    "content": f"{target_crop['crop_name']}施肥 - {fert.get('time', '')}:\n"
                               f"肥料类型: {fert.get('type', '')}\n"
                               f"用量: {fert.get('amount', '')}\n"
                               f"方法: {fert.get('method', '')}",
                    "metadata": {"crop": target_crop["crop_name"], "type": "fertilization"},
                    "score": 0.9
                })

        elif topic == "病虫害":
            diseases = target_crop.get("common_diseases", [])
            pests = target_crop.get("common_pests", [])

            for disease in diseases[:2]:
                results.append({
                    "content": f"{target_crop['crop_name']}病害 - {disease.get('name', '')}:\n"
                               f"症状: {disease.get('symptoms', '')}\n"
                               f"防治: {disease.get('prevention', '')}\n"
                               f"发生期: {disease.get('occurrence_stage', '')}",
                    "metadata": {"crop": target_crop["crop_name"], "type": "disease"},
                    "score": 0.9
                })

            for pest in pests[:2]:
                results.append({
                    "content": f"{target_crop['crop_name']}虫害 - {pest.get('name', '')}:\n"
                               f"危害: {pest.get('symptoms', '')}\n"
                               f"防治: {pest.get('control', '')}",
                    "metadata": {"crop": target_crop["crop_name"], "type": "pest"},
                    "score": 0.85
                })

        elif topic == "产量":
            yield_info = target_crop.get("yield_info", {})
            results.append({
                "content": f"{target_crop['crop_name']}产量信息:\n"
                           f"低产: {yield_info.get('low_yield', '')}\n"
                           f"中产: {yield_info.get('medium_yield', '')}\n"
                           f"高产: {yield_info.get('high_yield', '')}\n"
                           f"影响因素: {', '.join(yield_info.get('factors', []))}",
                "metadata": {"crop": target_crop["crop_name"], "type": "yield"},
                "score": 0.9
            })

        else:
            # 默认返回基本信息
            results.append({
                "content": f"作物: {target_crop['crop_name']}\n"
                           f"别名: {', '.join(target_crop.get('aliases', []))}\n"
                           f"适宜地区: {', '.join(target_crop.get('suitable_regions', []))}\n"
                           f"土壤要求: pH {target_crop.get('soil_requirements', {}).get('ph_range', '')}",
                "metadata": {"crop": target_crop["crop_name"], "type": "general"},
                "score": 0.8
            })

            # 添加生长阶段信息
            stages = target_crop.get("growth_stages", [])
            if stages:
                stage_info = "生长阶段:\n"
                for stage in stages[:4]:
                    stage_info += f"  - {stage.get('stage', '')}: 约{stage.get('duration_days', '')}天\n"
                results.append({
                    "content": stage_info,
                    "metadata": {"crop": target_crop["crop_name"], "type": "growth_stages"},
                    "score": 0.75
                })

        return results[:k]


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
        "番茄病虫害防治",
        "华北地区适合种什么"
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        results = rag.search(query)
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(result['content'])
