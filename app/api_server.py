"""FastAPI 后端服务 — 常驻进程 + 定时调度"""

import os, sys, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="智能种植助手 API", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 注册全部业务路由
from app.api_routes import register_routes
register_routes(app)

# 定时任务
from app.scheduler_jobs import check_reminders_job, check_weather_job, check_disease_job, check_device_rules_job, check_task_execution_job, check_camera_capture_job, check_autonomous_cycle_job

scheduler = BackgroundScheduler()
scheduler.add_job(check_reminders_job, "interval", minutes=5, id="reminders")
scheduler.add_job(check_weather_job, "interval", minutes=30, id="weather")
scheduler.add_job(check_disease_job, "interval", hours=6, id="disease")
scheduler.add_job(check_device_rules_job, "interval", minutes=5, id="device_rules")
scheduler.add_job(check_task_execution_job, "interval", minutes=3, id="task_execution")
scheduler.add_job(
    check_camera_capture_job, "interval",
    minutes=int(os.getenv("CAMERA_CHECK_INTERVAL", "30")),
    id="camera_capture",
)
scheduler.add_job(
    check_autonomous_cycle_job, "interval",
    minutes=int(os.getenv("AUTO_DECISION_INTERVAL", "30")),
    id="autonomous_cycle",
)
scheduler.start()
logger.info("APScheduler 已启动: 提醒/5min 天气/30min 病害/6h 设备规则/5min 任务执行/3min 摄像头巡检/30min 自主决策/30min")


if __name__ == "__main__":
    import uvicorn
    logger.info("FastAPI 后端启动: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
