"""财务查询节点 - 处理成本和收入查询、生成财务报表"""

import logging

from langchain_core.messages import AIMessage

from ..state import AgentState

from core.finance_manager import FinanceManager
from core.market_service import MarketService

logger = logging.getLogger(__name__)


def finance_query_node(state: AgentState) -> AgentState:
    """
    财务查询节点 - 处理成本和收入查询、生成财务报表、市场价格查询
    """
    if state.intent_type == "finance_query":
        user_question = state.user_question or ""

        # 获取当前作物
        crop = state.short_term_facts.get("crop") or state.user_profile.get("crop", "")

        # 解析查询意图
        is_record_request = any(word in user_question for word in ["记账", "记一笔", "添加", "录入", "花了", "收入"])
        is_report_request = any(word in user_question for word in ["报表", "报告", "汇总", "统计"])
        is_price_request = any(word in user_question for word in ["价格", "行情", "卖多少钱", "市场价", "值多少钱"])

        try:
            finance_manager = FinanceManager()
            market_service = MarketService()

            if is_record_request:
                # 处理记账请求 - 这里简化处理，实际应该解析具体金额
                state.final_answer = """💰 **记账功能**

请在侧边栏"财务管理"中记录详细的成本和收入信息。

支持记录：
• 种子、肥料、农药等成本支出
• 作物销售收入
• 查看亩均成本和收益分析

您也可以导入CSV文件批量导入历史财务数据。"""

            elif is_report_request:
                # 生成年度报表
                report = finance_manager.get_annual_report()
                state.final_answer = finance_manager.format_annual_report(report)

            elif is_price_request and crop:
                # 市场价格查询 + 收益估算
                price_report = market_service.format_market_report(crop)
                area = state.short_term_facts.get("farm_size") or state.user_profile.get("farm_size", 1.0)
                revenue = market_service.estimate_revenue(crop, area)
                if "error" not in revenue:
                    price_report += (
                        f"\n\n💰 **收益估算** (面积: {area}亩)\n"
                        f"- 预估总收入: ¥{revenue['revenue_low']:,} ~ ¥{revenue['revenue_high']:,}\n"
                        f"- 均价: {revenue['avg_price']} {revenue['price_unit']}\n"
                        f"- 参考产量: {revenue['yield_low']} ~ {revenue['yield_high']}"
                    )
                state.final_answer = price_report

            elif is_price_request:
                state.final_answer = market_service.format_market_report()

            elif crop:
                # 查询特定作物的财务情况
                summary = finance_manager.get_crop_financial_summary(crop)
                if summary:
                    state.final_answer = finance_manager.format_summary_report(summary)
                else:
                    state.final_answer = f"📊 **{crop}财务记录**\n\n暂无{crop}的财务记录。\n\n请在侧边栏「财务管理」中添加成本或收入记录。"

            else:
                # 显示总体财务概况
                report = finance_manager.get_annual_report()
                if report['crop_reports']:
                    state.final_answer = finance_manager.format_annual_report(report)
                else:
                    state.final_answer = """📊 **财务概览**

暂无财务记录。

请在侧边栏"财务管理"中：
1. 记录各项成本支出（种子、肥料、人工等）
2. 记录作物销售收入
3. 查看收益分析报告

您也可以导入CSV文件批量导入历史数据。"""

            state.messages.append(AIMessage(content=state.final_answer))

        except Exception as e:
            logger.error("财务查询异常: %s", e, exc_info=True)
            state.final_answer = "查询财务信息时出现错误，请稍后重试。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state
