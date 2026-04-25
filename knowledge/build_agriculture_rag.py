"""
农业知识库构建脚本
功能：
- 读取 agriculture_knowledge/ 下的 JSON 文件
- 将农业知识向量化
- 创建 FAISS 向量库
- 支持增量更新
"""

import os
import sys

# 获取项目根目录（脚本的上级目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
import json
from glob import glob
from typing import List, Dict, Any
import dotenv

from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

# 环境变量
dotenv.load_dotenv()

# Embedding模型配置（独立配置，兼容旧配置）
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if not EMBEDDING_API_KEY:
    raise EnvironmentError("未检测到 EMBEDDING_API_KEY 环境变量！")

os.environ["OPENAI_API_KEY"] = EMBEDDING_API_KEY
if EMBEDDING_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = EMBEDDING_BASE_URL

# 配置
KNOWLEDGE_DIR = "agriculture_knowledge"
FAISS_INDEX_DIR = "agriculture_faiss_index"



def load_crop_knowledge(crop_file: str) -> List[Dict[str, str]]:
    """
    加载单个作物知识文件，并切分为文本块

    Returns:
        文本块列表，每个块包含page_content和metadata
    """
    with open(crop_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []
    crop_name = data.get('crop_name', '')
    base_metadata = {
        "crop": crop_name,
        "source": os.path.basename(crop_file),
        "type": "crop_knowledge"
    }

    # 1. 基本信息块
    basic_info = f"作物名称：{crop_name}\n"
    basic_info += f"别名：{', '.join(data.get('aliases', []))}\n"
    basic_info += f"适宜地区：{', '.join(data.get('suitable_regions', []))}\n"
    chunks.append({
        "page_content": basic_info,
        "metadata": {**base_metadata, "section": "basic_info"}
    })

    # 2. 种植季节块
    if 'planting_seasons' in data:
        for season_key, season_info in data['planting_seasons'].items():
            content = f"{crop_name} - {season_info.get('name', season_key)}\n"
            content += f"播种时间：{season_info.get('sowing_time', '')}\n"
            content += f"收获时间：{season_info.get('harvest_time', '')}\n"
            content += f"适宜气候：{season_info.get('suitable_climate', '')}\n"
            content += f"备注：{season_info.get('notes', '')}\n"
            chunks.append({
                "page_content": content,
                "metadata": {**base_metadata, "section": "planting_season", "season": season_key}
            })

    # 3. 土壤要求块
    if 'soil_requirements' in data:
        soil = data['soil_requirements']
        content = f"{crop_name}的土壤要求：\n"
        content += f"适宜土壤类型：{', '.join(soil.get('preferred_types', []))}\n"
        content += f"pH范围：{soil.get('ph_range', '')}\n"
        content += f"肥力要求：{soil.get('fertility', '')}\n"
        if 'notes' in soil:
            content += f"注意事项：{soil['notes']}\n"
        chunks.append({
            "page_content": content,
            "metadata": {**base_metadata, "section": "soil_requirements"}
        })

    # 4. 气候要求块
    if 'climate_requirements' in data:
        climate = data['climate_requirements']
        content = f"{crop_name}的气候要求：\n"
        if 'temperature' in climate:
            temp = climate['temperature']
            content += f"发芽温度：{temp.get('germination', '')}\n"
            content += f"生长温度：{temp.get('growth', '')}\n"
        if 'water' in climate:
            water = climate['water']
            content += f"年降雨量：{water.get('annual_rainfall', '')}\n"
            content += f"需水关键期：{water.get('critical_period', '')}\n"
        if 'light' in climate:
            content += f"光照要求：{climate['light']}\n"
        chunks.append({
            "page_content": content,
            "metadata": {**base_metadata, "section": "climate_requirements"}
        })

    # 5. 生长阶段块
    if 'growth_stages' in data:
        for stage in data['growth_stages']:
            content = f"{crop_name} - {stage.get('stage', '')}\n"
            content += f"持续天数：约{stage.get('duration_days', '')}天\n"
            content += f"关键农事：{', '.join(stage.get('key_tasks', []))}\n"
            content += f"注意事项：{stage.get('notes', '')}\n"
            chunks.append({
                "page_content": content,
                "metadata": {**base_metadata, "section": "growth_stage", "stage_name": stage.get('stage', '')}
            })

    # 6. 施肥指导块
    if 'fertilization_guide' in data:
        for fert in data['fertilization_guide']:
            content = f"{crop_name}施肥 - {fert.get('time', '')}\n"
            content += f"肥料类型：{fert.get('type', '')}\n"
            content += f"用量：{fert.get('amount', '')}\n"
            content += f"施用方法：{fert.get('method', '')}\n"
            chunks.append({
                "page_content": content,
                "metadata": {**base_metadata, "section": "fertilization", "time": fert.get('time', '')}
            })

    # 7. 灌溉指导块
    if 'irrigation_guide' in data:
        for irr in data['irrigation_guide']:
            content = f"{crop_name}灌溉 - {irr.get('stage', '')}\n"
            content += f"灌溉目的：{irr.get('purpose', '')}\n"
            content += f"用水量：{irr.get('amount', '')}\n"
            chunks.append({
                "page_content": content,
                "metadata": {**base_metadata, "section": "irrigation", "stage": irr.get('stage', '')}
            })

    # 8. 病虫害块
    if 'common_diseases' in data:
        for disease in data['common_diseases'][:3]:  # 取前3种病害
            content = f"{crop_name}病害 - {disease.get('name', '')}\n"
            content += f"症状：{disease.get('symptoms', '')}\n"
            content += f"防治方法：{disease.get('prevention', '')}\n"
            content += f"发生时期：{disease.get('occurrence_stage', '')}\n"
            chunks.append({
                "page_content": content,
                "metadata": {**base_metadata, "section": "disease", "disease_name": disease.get('name', '')}
            })

    if 'common_pests' in data:
        for pest in data['common_pests'][:2]:  # 取前2种虫害
            content = f"{crop_name}虫害 - {pest.get('name', '')}\n"
            content += f"危害症状：{pest.get('symptoms', '')}\n"
            content += f"防治方法：{pest.get('control', '')}\n"
            chunks.append({
                "page_content": content,
                "metadata": {**base_metadata, "section": "pest", "pest_name": pest.get('name', '')}
            })

    # 9. 产量信息块
    if 'yield_info' in data:
        yield_info = data['yield_info']
        content = f"{crop_name}产量信息：\n"
        content += f"低产：{yield_info.get('low_yield', '')}\n"
        content += f"中产：{yield_info.get('medium_yield', '')}\n"
        content += f"高产：{yield_info.get('high_yield', '')}\n"
        content += f"影响因素：{', '.join(yield_info.get('factors', []))}\n"
        chunks.append({
            "page_content": content,
            "metadata": {**base_metadata, "section": "yield_info"}
        })

    return chunks


def build_agriculture_knowledge_base():
    """构建农业知识向量库"""
    print("=" * 60)
    print("农业知识库构建工具")
    print("=" * 60)

    # 初始化embeddings
    # 使用 DeepSeek Embeddings API
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # DeepSeek 支持的模型
    )

    # 收集所有文本块
    all_chunks = []

    # 加载作物知识
    crops_dir = os.path.join(KNOWLEDGE_DIR, "crops")
    if os.path.exists(crops_dir):
        json_files = glob(os.path.join(crops_dir, "*.json"))
        print(f"\n找到 {len(json_files)} 个作物知识文件")

        for file_path in json_files:
            print(f"处理: {os.path.basename(file_path)}")
            try:
                chunks = load_crop_knowledge(file_path)
                all_chunks.extend(chunks)
                print(f"  -> 切分为 {len(chunks)} 个文本块")
            except Exception as e:
                print(f"  -> 处理失败: {e}")

    if not all_chunks:
        print("\n错误: 没有找到任何知识文件")
        return

    print(f"\n总计: {len(all_chunks)} 个文本块")

    # 准备向量化数据
    texts = [chunk["page_content"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    # 创建FAISS向量库
    print("\n正在创建FAISS向量库...")
    vectorstore = FAISS.from_texts(
        texts,
        embeddings,
        metadatas=metadatas
    )

    # 保存向量库
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)

    print(f"\n✅ 向量库已保存到: {FAISS_INDEX_DIR}")
    print(f"   包含 {len(texts)} 条知识条目")

    # 导出文本数据供查看
    export_file = os.path.join(FAISS_INDEX_DIR, "knowledge_export.json")
    with open(export_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"   知识文本已导出到: {export_file}")


def search_knowledge(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    搜索农业知识

    Args:
        query: 查询问题
        k: 返回结果数量

    Returns:
        相关知识列表
    """
    if not os.path.exists(FAISS_INDEX_DIR):
        print("错误: 向量库不存在，请先运行构建脚本")
        return []

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in results
    ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "search":
        # 搜索模式
        query = sys.argv[2] if len(sys.argv) > 2 else "小麦什么时候播种"
        print(f"\n搜索: {query}\n")
        results = search_knowledge(query)
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['metadata'].get('section', '')}]")
            print(result['content'])
            print()
    else:
        # 构建模式
        build_agriculture_knowledge_base()
