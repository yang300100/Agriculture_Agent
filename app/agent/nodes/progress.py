"""种植进度跟踪节点 — 读取用户真实进度和任务"""

import logging
from langchain_core.messages import AIMessage
from ..state import AgentState

logger = logging.getLogger(__name__)


def progress_tracking_node(state: AgentState) -> AgentState:
    """查询用户的实际种植进度和待办任务"""
    try:
        import os as _os
        from core.planting_tracker import PlantingTracker
        username = getattr(state, 'username', 'default')
        tracker = PlantingTracker(_os.path.join("data", username))
        progresses = tracker.get_progress()
        tasks = tracker.get_tasks()
    except Exception as e:
        logger.warning("读取种植进度失败: %s", e)
        state.messages.append(AIMessage(content="暂时无法读取种植进度，请稍后再试。"))
        state.final_answer = "进度读取失败"
        return state

    crop = state.short_term_facts.get("crop", "")

    # 如果用户指定了作物，筛选相关记录
    if crop:
        progresses = [p for p in progresses if crop in p.crop]
        tasks = [t for t in tasks if crop in t.crop]

    answer = _format_progress_answer(progresses, tasks, crop)
    state.messages.append(AIMessage(content=answer))
    state.final_answer = answer
    return state


def _format_progress_answer(progresses, tasks, crop: str) -> str:
    """格式化进度回答"""
    lines = []

    if crop:
        lines.append(f"## 🌾 {crop} 种植进度\n")
    else:
        lines.append("## 🌾 您的种植进度总览\n")

    # 进度
    if progresses:
        lines.append("**当前种植进度：**\n")
        for p in progresses[:5]:
            status_icon = {"进行中": "🌱", "已完成": "✅", "待开始": "⚪"}.get(p.status, "📋")
            lines.append(
                f"- {status_icon} **{p.crop}** — {p.stage} "
                f"({p.stage_number}/{p.total_stages} 阶段, {p.progress_percent}%)"
            )
    else:
        if crop:
            lines.append(f"暂无 {crop} 的种植进度记录。\n")
        else:
            lines.append("暂无种植进度记录。在侧边栏「种植进度」中添加。\n")

    # 任务
    if tasks:
        lines.append("\n**待办农事任务：**\n")
        active_tasks = [t for t in tasks if t.status in ("待办", "进行中")]
        overdue_tasks = [t for t in tasks if t.status == "已逾期"]
        for t in overdue_tasks[:3]:
            lines.append(f"- ⚠️ **{t.title}**（{t.crop}）— 已逾期")
        for t in active_tasks[:5]:
            priority = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.priority, "")
            lines.append(f"- {priority} **{t.title}**（{t.crop}）— {t.status}")
        if not active_tasks and not overdue_tasks:
            lines.append("当前没有待办任务。")
    else:
        lines.append("\n暂无农事任务。在侧边栏「农事任务」中添加。")

    if not progresses and not tasks:
        lines.append("\n💡 您可以在侧边栏添加种植进度和农事任务，我会帮您跟踪。")

    return "\n".join(lines)
