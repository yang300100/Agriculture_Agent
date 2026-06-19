"""种植规划节点 - 生成个性化种植计划，同时创建进度卡片和任务"""

import logging
from datetime import datetime

from langchain_core.messages import AIMessage

from ..state import AgentState

from core.planting_planner import PlantingPlanner
from core.planting_tracker import PlantingTracker

logger = logging.getLogger(__name__)


def planting_plan_node(state: AgentState) -> AgentState:
    """种植规划节点 - 生成个性化种植计划，同时创建进度卡片和任务"""
    if state.intent_type in ["crop_selection", "planting_schedule"]:
        # 提取用户信息
        user_info = {
            "region": state.short_term_facts.get("region") or state.user_profile.get("region", ""),
            "soil_type": state.short_term_facts.get("soil_type") or state.user_profile.get("soil_type", ""),
            "farm_size": state.short_term_facts.get("farm_size") or state.user_profile.get("farm_size", 1.0),
            "goals": state.short_term_facts.get("goals") or state.user_profile.get("goals", []),
            "experience": state.user_profile.get("experience", ""),
            "crop": state.short_term_facts.get("crop", "")
        }

        try:
            # 生成种植计划
            planner = PlantingPlanner()
            plan = planner.generate_plan(user_info)

            # 更新AgentState
            state.planting_plan = {
                "crops": [plan.crop],
                "schedule": plan.schedule,
                "methods": {},
                "progress": {},
                "created_at": plan.created_at
            }

            # 更新用户档案
            if plan.region:
                state.user_profile["region"] = plan.region
            if plan.soil_type:
                state.user_profile["soil_type"] = plan.soil_type

            # 创建进度卡片和任务（使用用户专属目录）
            try:
                import os as _os
                username = getattr(state, 'username', 'default')
                tracker = PlantingTracker(_os.path.join("data", username))

                # 1. 创建整体种植进度记录
                current_stage = "准备期"
                stage_number = 0
                total_stages = len(plan.schedule.get("stages", [])) if plan.schedule else 1

                if plan.schedule and plan.schedule.get("stages"):
                    first_stage = plan.schedule["stages"][0]
                    current_stage = first_stage.get("stage", "准备期")
                    stage_number = 1

                # 创建进度记录
                progress = tracker.create_progress({
                    "crop": plan.crop,
                    "stage": current_stage,
                    "stage_number": stage_number,
                    "total_stages": total_stages,
                    "start_date": datetime.now().strftime("%Y-%m-%d"),
                    "expected_end_date": plan.schedule.get("harvest_time", ""),
                    "progress_percent": 0,
                    "status": "进行中",
                    "tasks": [],
                    "notes": f"种植面积: {plan.farm_size}亩, 地区: {plan.region}"
                })

                # 2. 为每个阶段的关键任务创建任务卡片
                if plan.tasks:
                    for stage_info in plan.tasks:
                        stage_name = stage_info.get("stage", "")
                        for task_info in stage_info.get("tasks", [])[:2]:  # 每个阶段最多2个任务
                            task_date = task_info.get("date", "")
                            task_name = task_info.get("task", "")
                            priority = task_info.get("priority", "中")

                            tracker.create_task({
                                "crop": plan.crop,
                                "task_type": task_name[:4] if len(task_name) >= 4 else task_name,
                                "title": f"{stage_name} - {task_name}",
                                "description": f"{plan.crop}的{stage_name}阶段任务",
                                "status": "待办",
                                "priority": "high" if priority == "高" else "medium",
                                "end_date": task_date,
                                "progress_percent": 0
                            })

                # 3. 添加资源准备任务（种子、肥料等）
                if plan.resources:
                    if plan.resources.get("seeds"):
                        tracker.create_task({
                            "crop": plan.crop,
                            "task_type": "播种",
                            "title": f"准备{plan.crop}种子",
                            "description": f"需准备: {plan.resources['seeds'].get('amount', '适量')}",
                            "status": "待办",
                            "priority": "high",
                            "end_date": plan.schedule.get("sowing_time", datetime.now().strftime("%Y-%m-%d")),
                            "progress_percent": 0
                        })

            except Exception as e:
                logger.warning(f"创建进度卡片失败: {e}")

            # 格式化回答
            answer = planner.format_plan_as_text(plan)

            # 作物选择意图：追加多方案对比 + 轮作建议
            if state.intent_type == "crop_selection":
                try:
                    from core.crop_comparison import generate_multi_crop_plan, format_comparison_table
                    options = generate_multi_crop_plan(user_info, num_options=3)
                    if options:
                        comparison = format_comparison_table(options)
                        answer += f"\n\n---\n{comparison}"
                except Exception:
                    pass

                # 轮作建议
                try:
                    from core.map_manager import MapManager
                    from core.crop_rotation import CropRotationAdvisor
                    fields = MapManager().get_all_fields()
                    previous_crop = state.short_term_facts.get("previous_crop", "")
                    rotation_lines = []

                    advisor = CropRotationAdvisor()
                    for f in fields:
                        if f.current_crop and f.current_crop != plan.crop:
                            risk = advisor.check_continuous_cropping_risk(f.current_crop)
                            rotation_lines.append(
                                f"- 地块「{f.name}」当前种 {f.current_crop}："
                                f"{risk['message']}"
                            )
                    if previous_crop and previous_crop != plan.crop:
                        risk = advisor.check_continuous_cropping_risk(previous_crop)
                        rotation_lines.append(
                            f"- 上季种 {previous_crop} 后种 {plan.crop}：{risk['message']}"
                        )
                    if rotation_lines:
                        answer += "\n\n**🔄 轮作提示：**\n" + "\n".join(rotation_lines)
                except Exception:
                    pass

            state.final_answer = answer
            state.messages.append(AIMessage(content=answer))

        except Exception as e:
            state.final_answer = f"生成种植计划时出现错误：{str(e)}。请稍后再试或联系技术支持。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state
