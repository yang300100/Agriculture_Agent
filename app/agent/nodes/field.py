"""地块管理节点 - 处理地块查询和管理请求"""

import logging

from langchain_core.messages import AIMessage

from ..state import AgentState

from core.map_manager import MapManager

logger = logging.getLogger(__name__)


def field_management_node(state: AgentState) -> AgentState:
    """
    地块管理节点 - 处理地块查询和管理请求
    """
    if state.intent_type == "field_management":
        user_question = state.user_question or ""

        try:
            # 初始化地图管理器
            map_manager = MapManager()
            fields = map_manager.get_all_fields()

            # 检查是否是查询请求
            is_query = any(word in user_question for word in ["多少", "几个", "哪里", "在哪", "查询", "查看", "显示"])
            is_rotation = any(word in user_question for word in ["轮作", "换茬", "连作", "下季", "接着种", "种什么好"])

            if is_rotation and fields:
                # 轮作建议
                from core.crop_rotation import CropRotationAdvisor
                advisor = CropRotationAdvisor()
                answer_parts = ["## 轮作建议\n"]
                for field in fields:
                    crop = field.current_crop
                    if crop:
                        report = advisor.format_rotation_report(crop, field.name)
                        answer_parts.append(report)
                        answer_parts.append("\n---\n")
                if answer_parts:
                    state.final_answer = "\n".join(answer_parts)

            elif is_query or not fields:
                # 生成地块信息报告
                if fields:
                    answer_parts = ["📍 **我的地块信息**\n"]
                    total_area = 0
                    for i, field in enumerate(fields, 1):
                        answer_parts.append(f"\n**地块{i}：{field.name}**")
                        answer_parts.append(f"- 面积：{field.area_mu:.2f}亩")
                        answer_parts.append(f"- 位置：{field.center_lat:.4f}°N, {field.center_lon:.4f}°E")
                        if field.soil_type:
                            answer_parts.append(f"- 土壤：{field.soil_type}")
                        if field.current_crop:
                            answer_parts.append(f"- 当前作物：{field.current_crop}")
                        total_area += field.area_mu

                    answer_parts.append(f"\n---")
                    answer_parts.append(f"**总计**：{len(fields)}个地块，共{total_area:.2f}亩")
                    answer_parts.append(f"\n💡 您可以在侧边栏「我的地块」中管理和添加新地块")

                    state.final_answer = "\n".join(answer_parts)
                else:
                    state.final_answer = """📍 **地块管理**

您还没有添加任何地块。

请在侧边栏「我的地块」中：
1. 点击「添加新地块」
2. 在地图上绘制地块边界
3. 系统自动计算面积
4. 填写地块信息并保存

地块信息将用于：
- 精准天气预测
- 分区种植规划
- 面积和成本核算"""
            else:
                # 一般性地块管理介绍
                state.final_answer = """📍 **地块管理功能**

您可以通过以下方式管理您的农田地块：

**1. 添加地块**
- 在侧边栏点击「我的地块」
- 点击「添加新地块」按钮
- 在地图上绘制多边形边界
- 系统自动计算面积

**2. 地块信息**
- 记录土壤类型
- 标注当前作物
- 查看总面积统计

**3. 应用场景**
- 基于位置获取精准天气
- 分地块管理种植计划
- 按地块记录成本和收入

请问您想查看已有地块信息还是添加新地块？"""

            state.messages.append(AIMessage(content=state.final_answer))

        except Exception as e:
            state.final_answer = f"地块管理功能出现错误：{str(e)}。请稍后重试。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state
