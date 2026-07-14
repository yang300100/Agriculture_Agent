"""政策补贴查询节点"""
import logging
from langchain_core.messages import AIMessage
from ..state import AgentState

logger = logging.getLogger(__name__)


def policy_query_node(state: AgentState) -> AgentState:
    """查询农业政策和补贴信息"""
    query = state.user_question or ""
    crop = state.short_term_facts.get("crop", "")
    region = state.short_term_facts.get("region", "")

    # 收集相关补贴信息
    subsidies = _collect_subsidies(query, crop, region)
    answer = _format_policy_answer(query, crop, region, subsidies)
    state.final_answer = answer
    state.messages.append(AIMessage(content=answer))
    return state


def _collect_subsidies(query: str, crop: str, region: str) -> list:
    """从政策索引和作物知识中搜索补贴信息"""
    results = []
    seen_contents = set()  # 去重用：基于内容相似度

    def _is_duplicate(content: str) -> bool:
        """检查内容是否与已有结果高度相似"""
        normalized = content.strip()[:200]  # 比较前200字
        if normalized in seen_contents:
            return True
        seen_contents.add(normalized)
        return False

    # 1. 搜索 FAISS 政策索引
    try:
        from knowledge.faiss_agriculture_rag import FAISSAgricultureRAG
        faiss = FAISSAgricultureRAG()
        if faiss.is_available:
            search_query = f"{crop} 补贴 政策 {region}" if crop else f"农业补贴 政策 {region}"
            faiss_results = faiss.search(search_query, k=3)
            for r in faiss_results:
                content = r.get("content", "")[:500]
                if not _is_duplicate(content):
                    results.append({
                        "content": content,
                        "source": "政策文档",
                    })
    except Exception as e:
        logger.warning("FAISS 政策检索失败: %s", e)

    # 2. 从作物知识库获取市场/补贴信息
    try:
        from knowledge.simple_agriculture_rag import SimpleAgricultureRAG
        rag = SimpleAgricultureRAG()
        rag_results = rag.search(f"{crop} 补贴 价格 政策", k=2)
        for r in rag_results:
            if any(kw in r.get("content", "") for kw in ["补贴", "价格", "政策"]):
                content = r.get("content", "")[:500]
                if not _is_duplicate(content):
                    results.append({
                        "content": content,
                        "source": r.get("metadata", {}).get("source", "作物知识库"),
                    })
    except Exception as e:
        logger.warning("简单检索失败: %s", e)

    # 3. 内置常见补贴参考
    if not results:
        results.append({
            "content": _builtin_subsidy_reference(),
            "source": "补贴政策参考",
        })

    return results


def _builtin_subsidy_reference() -> str:
    """内置常见农业补贴政策参考"""
    text = (
        "常见农业补贴类型：\n"
        "1. 耕地地力保护补贴：一般每亩50-100元，具体标准由各省制定\n"
        "2. 农机购置补贴：购买指定农机可享受30%左右补贴\n"
        "3. 农业保险保费补贴：中央和地方财政补贴保费50%-80%\n"
        "4. 最低收购价政策：小麦、水稻有最低收购价保护\n"
        "5. 大豆玉米带状复合种植补贴：每亩150-200元"
    )
    return text


CROP_SUBSIDY_MAP = {
    "小麦": "小麦最低收购价政策：国家在小麦主产区实行最低收购价，保障农民收益。",
    "水稻": "水稻最低收购价政策：国家在稻谷主产区实行最低收购价，早籼稻、中晚籼稻、粳稻分别制定价格。",
    "玉米": "玉米生产者补贴：东北三省和内蒙古对玉米种植给予生产者补贴，每亩约50-150元。",
    "大豆": "大豆生产者补贴：东北地区大豆种植补贴高于玉米，每亩约200-300元。",
    "棉花": "棉花目标价格补贴：新疆实行目标价格改革试点。",
    "油菜": "油菜籽临时收储政策：部分地区对油菜籽实行托市收购和种植补贴。",
    "花生": "花生良种补贴：部分主产区对花生种植给予良种补贴和生产扶持。",
    "甘薯": "甘薯种植补贴：部分地区将甘薯纳入粮食作物补贴范围。",
    "甘蔗": "甘蔗种植补贴：广西、云南等主产区对甘蔗种植给予良种和机械化补贴。",
    "茶叶": "茶叶产业扶持：各产茶省对茶园建设、品牌推广给予补贴支持。",
    "烟草": "烟叶生产补贴：烟草公司对烟农提供物资补贴和技术指导。",
}


def _format_policy_answer(query: str, crop: str, region: str,
                          results: list) -> str:
    """格式化政策回答"""
    lines = []
    if crop:
        lines.append(f"## {crop}相关补贴政策\n")
    else:
        lines.append("## 农业补贴政策\n")

    if results:
        for r in results:
            lines.append(f"**来源：{r.get('source', '')}**\n{r.get('content', '')}\n")
    else:
        lines.append("暂未找到相关补贴政策信息。建议：")
        lines.append("- 咨询当地农业农村局了解最新补贴政策")
        lines.append("- 访问政府网站查询相关政策文件")

    if crop and crop in CROP_SUBSIDY_MAP:
        lines.append(f"\n### {crop}特别说明\n{CROP_SUBSIDY_MAP[crop]}")

    lines.append("\n---\n*政策信息仅供参考，具体以当地政府文件为准。*")
    return "\n".join(lines)
