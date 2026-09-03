"""FastAPI 后端服务 — 常驻进程 + 定时调度"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 设备驱动层日志设为 WARNING，减少无硬件时的连接超时日志
logging.getLogger("devices").setLevel(logging.WARNING)
logging.getLogger("core.device_registry_factory").setLevel(logging.WARNING)
logging.getLogger("core.device_executor").setLevel(logging.WARNING)
# geopy 的 Nominatim 连接超时警告（已用本机IP定位替代，无实际影响）
logging.getLogger("geopy").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app):
    """服务启动时在主线程内一次性初始化数据库。

    数据库（core.database 包）此前是在请求处理器里懒加载的：当 uvicorn
    线程池中的多个请求首次并发访问 PlantingTracker/get_session 时，会各自
    在不同线程里同时导入 core.database 包及其 repository 子模块，触发跨线程
    模块锁死锁（_ModuleLock('core.database.repository') DeadlockError）而返回
    500。这里在启动阶段（仍处主线程、尚未处理任何请求）先 init_db() 一次，
    让所有数据库模块提前进入 sys.modules 缓存，后续请求即不会再并发重导入。
    """
    from core.database.engine import init_db
    init_db()
    yield


app = FastAPI(title="智能种植助手 API", version="1.0", lifespan=lifespan)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册全部业务路由
from app.api_routes import register_routes
register_routes(app)

# Web 进程不再隐式启动调度器。调度任务由 app/scheduler_runner.py
# 独立运行，避免 Uvicorn 多 Worker 重复执行硬件动作。
logger.info("Web 进程已启用无副作用模式；定时任务由独立调度进程负责")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "18001"))
    logger.info("FastAPI 后端启动: http://localhost:%s", port)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=300,
    )
