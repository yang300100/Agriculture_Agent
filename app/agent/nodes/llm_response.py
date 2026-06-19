"""LLM 回答节点：统一 LLM 回复、专家回答、追问引导"""

import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState
from ..config import *
from ..utils import extract_facts_from_conversation, aggregate_sentences

logger = logging.getLogger(__name__)


def llm_response_node(state: AgentState) -> AgentState:
    """
    统一 LLM 回答节点 - 所有回复都通过 LLM 生成
    并自动提取和记忆关键信息
    """
    intent = state.intent_type
    user_question = state.user_question or ""
    long_memory = state.long_term_profile.get("summary", "")
    user_profile = state.user_profile

    # 从当前对话提取关键事实并更新 short_term_facts（累积，不覆盖已有信息）
    new_facts = extract_facts_from_conversation(state)
    for key, value in new_facts.items():
        # 只在值为空时才更新，避免覆盖已有信息
        if not state.short_term_facts.get(key):
            state.short_term_facts[key] = value
            # 同时更新用户档案
            if key in ["region", "soil_type", "farm_size", "experience", "goals"]:
                user_profile[key] = value

    # 构建已收集的信息摘要
    collected_info = []
    if state.short_term_facts.get("region") or user_profile.get("region"):
        collected_info.append(f"地区：{state.short_term_facts.get('region', user_profile.get('region', ''))}")
    if state.short_term_facts.get("crop"):
        collected_info.append(f"作物：{state.short_term_facts.get('crop')}")
    if state.short_term_facts.get("farm_size") or user_profile.get("farm_size"):
        collected_info.append(f"面积：{state.short_term_facts.get('farm_size', user_profile.get('farm_size', ''))}亩")
    if state.short_term_facts.get("soil_type") or user_profile.get("soil_type"):
        collected_info.append(f"土壤：{state.short_term_facts.get('soil_type', user_profile.get('soil_type', ''))}")

    # 构建完整的对话历史（包括最近的几轮对话）
    recent_history = []
    for msg in state.messages[-10:]:  # 取最近10条消息
        if isinstance(msg, HumanMessage):
            recent_history.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            recent_history.append(f"助手：{msg.content[:100]}...")

    history_text = "\n".join(recent_history[-6:]) if recent_history else "暂无对话历史"

    # 构建系统提示词
    system_prompt = f"""你是一位专业的智能种植规划助手。请根据用户的意图和问题，提供自然、友好、专业的回答。

【当前用户意图类型】：{intent}

【用户档案】（来自用户填写的基础信息）：
- 地区：{user_profile.get('region', '未填写')}
- 土壤类型：{user_profile.get('soil_type', '未填写')}
- 种植面积：{user_profile.get('farm_size', '未填写')} 亩
- 种植经验：{user_profile.get('experience', '未填写')}
- 种植目标：{', '.join(user_profile.get('goals', [])) if user_profile.get('goals') else '未填写'}

【本次对话已收集的信息】：
{chr(10).join(collected_info) if collected_info else '暂无新信息'}

【对话历史】（最近几轮）：
{history_text}

【历史摘要】：
{long_memory if long_memory else '暂无历史摘要'}

【请遵循以下规则】：
1. 【自然对话】像真人一样自然交流，避免机械化的回复
2. 【记住信息】**这是最重要的规则**：
   - 如果【本次对话已收集的信息】中已经包含地区、作物等信息，绝对不要再次询问！
   - 直接基于已知信息给出专业建议
   - 只有在信息缺失时，才询问缺失的部分
3. 【上下文连贯性】**关键规则**：
   - 仔细看【对话历史】，理解之前讨论的主题
   - 用户输入简短内容（如只输入作物名）通常是对之前话题的确认或补充
   - 保持话题连贯，不要突然切换到新话题
   - 如果之前正在讨论病虫害问题，用户输入作物名，应继续讨论该作物的病虫害
4. 【专业准确】提供准确的农业种植知识和建议
5. 【个性化】根据用户的地区、作物等信息提供定制化建议

【意图类型说明】：
- greeting: 用户问候，请友好回应并询问有什么可以帮助
- thanks: 用户感谢，请礼貌回应并询问是否还有其他问题
- farewell: 用户告别，请友好道别并祝种植顺利
- identity: 用户询问身份，请介绍自己作为智能种植规划助手
- function: 用户询问功能，请介绍你能提供的种植相关服务
- crop_selection: 作物选择建议，请根据地区、土壤等推荐作物
- planting_schedule: 种植时间规划，请提供播种、收获等时间建议
- planting_method: 种植方法指导，请提供详细的栽培技术指导
- reminder_setup: 提醒设置管理，请帮助设置浇水、施肥等提醒
- progress_tracking: 进度跟踪查询，请查看并更新种植进度
- disease_prevention: 病虫害防治，请提供诊断和防治建议。如果用户只输入作物名且之前正在讨论病害，继续分析该作物的病害问题
- harvest_planning: 收获规划建议，请提供最佳收获时间和方法
- image_analysis: 图片分析，请分析上传的农作物图片
- unclear: 意图不明，请礼貌询问用户具体需求

现在请回复用户的问题："""

    # 如果有检索到的知识，添加到系统提示词中
    if state.retrieved_docs:
        knowledge_text = "\n".join([f"- {doc['page_content'][:200]}" for doc in state.retrieved_docs[:3]])
        system_prompt += f"\n\n【相关知识】：\n{knowledge_text}"

    # 构建消息 - 系统提示 + 历史对话 + 当前问题
    messages = [SystemMessage(content=system_prompt)]

    # 添加历史对话（最近6轮，不包括系统消息）
    history_messages = [msg for msg in state.messages if isinstance(msg, (HumanMessage, AIMessage))]

    # 找到当前问题的位置
    current_msg_index = -1
    for i, msg in enumerate(history_messages):
        if isinstance(msg, HumanMessage) and msg.content == user_question:
            current_msg_index = i
            break

    # 添加历史消息（最多6轮 = 12条消息，不包括当前这条）
    if current_msg_index > 0:
        history_start = max(0, current_msg_index - 12)
        for i in range(history_start, current_msg_index):
            messages.append(history_messages[i])
    elif len(history_messages) > 1:
        # 如果没找到当前问题，添加最近的历史（最后一条是当前的）
        for msg in history_messages[-13:-1]:  # 最多12条历史
            messages.append(msg)

    # 确保最后一条是当前用户问题
    if not (isinstance(messages[-1], HumanMessage) and messages[-1].content == user_question):
        messages.append(HumanMessage(content=user_question))

    # 调用 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    try:
        response = llm.invoke(messages)
        state.final_answer = response.content
    except Exception as e:
        logger.debug(f"LLM 调用失败: {e}")
        state.final_answer = "抱歉，我暂时无法回答，请稍后再试。"

    state.messages.append(AIMessage(content=state.final_answer))
    return state


def general_response_node(state: AgentState) -> AgentState:
    """通用回复节点 - 现在通过 LLM 生成所有回复"""
    return llm_response_node(state)


def clarification_node(state: AgentState) -> AgentState:
    """
    使用 LLM 生成动态追问引导
    """
    user_question = state.user_question or ""
    intent = state.intent_type
    long_memory = state.long_term_profile.get("summary", "")
    user_profile = state.user_profile
    short_facts = state.short_term_facts

    # 检查已收集的信息
    has_region = bool(short_facts.get('region') or user_profile.get('region'))
    has_crop = bool(short_facts.get('crop'))
    has_soil = bool(short_facts.get('soil_type') or user_profile.get('soil_type'))
    has_area = bool(short_facts.get('farm_size') or user_profile.get('farm_size'))

    # 构建系统提示词让 LLM 生成追问
    clarify_prompt = f"""你是一位专业的种植规划顾问。用户的意图不够明确，需要你礼貌地询问更多信息。

【用户当前意图】：{intent}

【用户已提供的信息】：
- 地区：{'已提供：' + (short_facts.get('region') or user_profile.get('region', '')) if has_region else '未提供'}
- 作物：{'已提供：' + short_facts.get('crop', '') if has_crop else '未提供'}
- 土壤类型：{'已提供：' + (short_facts.get('soil_type') or user_profile.get('soil_type', '')) if has_soil else '未提供'}
- 种植面积：{'已提供：' + str(short_facts.get('farm_size') or user_profile.get('farm_size', '')) + '亩' if has_area else '未提供'}
- 种植经验：{user_profile.get('experience', '未提供')}
- 用户问题：{user_question}

【对话历史】：{long_memory}

【重要要求】：
1. **绝不要重复询问用户已经提供的信息**
2. 只询问缺失的关键信息（用"未提供"标记的）
3. 如果所有关键信息都已收集，直接说"请详细描述您的需求"
4. 语气友好自然，像真人顾问一样

【意图所需的最低信息】：
- crop_selection: 需要地区和作物（地区已从用户档案获取，如已提供则不再问）
- planting_schedule: 需要作物和地区
- planting_method: 需要作物
- reminder_setup: 需要作物和提醒类型
- progress_tracking: 需要作物
- disease_prevention: 需要作物和症状
- harvest_planning: 需要作物
- unclear: 询问具体想咨询哪方面的问题

请生成追问话术（只问缺失的信息）："""

    # 调用 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.7,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    try:
        response = llm.invoke([HumanMessage(content=clarify_prompt)])
        clarify_msg = response.content.strip()
    except Exception as e:
        logger.debug(f"LLM 追问生成失败: {e}")
        # 降级到简单追问
        clarify_msg = "为了更好地帮助您，能否告诉我更多信息？比如您所在的地区和想种植的作物？"

    state.final_answer = clarify_msg
    state.messages.append(AIMessage(content=clarify_msg))
    return state


def llm_expert_answer(state: AgentState) -> AgentState:
    """种植专家回答节点"""
    long_memory = state.long_term_profile.get("summary", "")
    memory_context = f"\n【对话历史总结】：{long_memory}\n" if long_memory else ""

    # 从short_term_facts获取用户上下文
    user_context = ""
    if state.short_term_facts:
        user_context = "\n【用户已知信息】\n"
        for key, value in state.short_term_facts.items():
            user_context += f"  - {key}: {value}\n"

    system_prompt = f"""
你是一位经验丰富的农业种植专家，请遵循以下规则：

1. 【基于知识回答】优先使用提供的农业知识回答问题
2. 【科学实用】建议要科学、实用、可操作，适合农户实际执行
3. 【因地制宜】考虑用户的地区、土壤、气候等条件给出建议
4. 【风险提示】对可能的风险（病虫害、天气等）给出预警和防范建议
5. 【通俗易懂】使用朴实易懂的语言，避免过于学术化

【回答格式】
- 对于种植时间：明确说明播种期和收获期
- 对于种植方法：分步骤说明关键操作
- 对于病虫害：描述症状 + 防治方法
- 对于不确定的问题：诚实说明，建议咨询当地农技站

【对话上下文】
{memory_context}
{user_context}
"""

    # 构造知识证据
    evidence_blocks = ""
    if state.retrieved_docs:
        aggregated = aggregate_sentences(state.retrieved_docs)
        evidence_blocks = "\n【检索到的农业知识】\n"
        for i, item in enumerate(aggregated, 1):
            evidence_blocks += f"\n【知识{i}｜{item['source']}】\n{item['content']}\n"

    # 调用LLM - 正确传递对话历史
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    # 构建消息列表
    messages = [SystemMessage(content=system_prompt)]

    # 添加历史对话（最近6轮）
    user_question = state.user_question or ""
    current_msg_index = -1
    for i, msg in enumerate(state.messages):
        if isinstance(msg, HumanMessage) and msg.content == user_question:
            current_msg_index = i
            break

    history_start = max(0, current_msg_index - 12) if current_msg_index >= 0 else max(0, len(state.messages) - 12)
    for i in range(history_start, current_msg_index if current_msg_index >= 0 else len(state.messages)):
        msg = state.messages[i]
        if isinstance(msg, (HumanMessage, AIMessage)):
            messages.append(msg)

    # 添加知识证据
    if evidence_blocks:
        messages.append(HumanMessage(content=evidence_blocks))

    # 添加当前问题
    if not (isinstance(messages[-1], HumanMessage) and messages[-1].content == user_question):
        messages.append(HumanMessage(content=user_question))

    response = llm.invoke(messages)

    state.final_answer = response.content
    state.messages.append(AIMessage(content=response.content))
    return state
