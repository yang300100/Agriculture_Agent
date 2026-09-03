"""
智能种植规划助手启动脚本。

支持分别启动后端、CLI，以及在 Streamlit 与新前端之间进行选择。
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv


# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

FRONTEND_STREAMLIT = "streamlit"
FRONTEND_NEXT = "next"
FRONTEND_ALIASES = {
    "1": FRONTEND_STREAMLIT,
    "streamlit": FRONTEND_STREAMLIT,
    "legacy": FRONTEND_STREAMLIT,
    "old": FRONTEND_STREAMLIT,
    "2": FRONTEND_NEXT,
    "next": FRONTEND_NEXT,
    "nextjs": FRONTEND_NEXT,
    "react": FRONTEND_NEXT,
    "new": FRONTEND_NEXT,
}


def _configure_console_encoding():
    """让中文和状态图标在 Windows 终端及重定向输出中稳定显示。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (LookupError, OSError):
                pass


_configure_console_encoding()


def _ensure_database():
    """确保数据库已初始化，并尝试从 JSON 迁移数据。"""
    db_path = os.path.join(PROJECT_ROOT, "data", "agriculture.db")
    if not os.path.exists(db_path):
        print("初始化数据库...")
        try:
            from core.database.engine import init_db

            init_db()
            print("  数据库已创建: data/agriculture.db")
            json_data = os.path.join(PROJECT_ROOT, "data", "users.json")
            if os.path.exists(json_data):
                print("  检测到 JSON 数据，开始迁移...")
                from scripts.migrate_json_to_sqlite import migrate_all

                migrate_all()
        except Exception as exc:
            print(f"  数据库初始化失败: {exc}")


def check_env():
    """检查环境配置。"""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        print(f"⚠️  未找到 .env 文件（应在: {env_path}）")
        print("请创建 .env 文件并配置 LLM_API_KEY 或 OPENAI_API_KEY")
        return False
    os.chdir(PROJECT_ROOT)
    print(f"✅ 已加载环境配置: {env_path}")
    return True


def build_knowledge_base():
    """构建农业知识库。"""
    print("📚 检查农业知识库...")
    index_path = os.path.join(PROJECT_ROOT, "agriculture_faiss_index")
    if not os.path.exists(index_path):
        print("  知识库不存在，开始构建...")
        try:
            build_script = os.path.join(
                PROJECT_ROOT, "knowledge", "build_agriculture_rag.py"
            )
            subprocess.run([sys.executable, build_script], check=True)
            print("  ✅ 知识库构建完成")
        except subprocess.CalledProcessError as exc:
            print(f"  ❌ 构建失败: {exc}")
            return False
    else:
        print("  ✅ 知识库已存在")
    return True


def normalize_frontend(value):
    """把编号或别名转换为统一的前端名称。"""
    if value is None:
        return None
    return FRONTEND_ALIASES.get(str(value).strip().lower())


def choose_frontend():
    """在终端中选择要使用的前端。"""
    print()
    print("请选择要启动的前端：")
    print("  1. Streamlit 前端（原版，http://localhost:8501）")
    print("  2. React / Vinext 前端（新版）")
    print("  0. 取消启动")

    while True:
        try:
            choice = input("请输入选项 [1/2，默认 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n未读取到选择，将使用 Streamlit 前端。")
            return FRONTEND_STREAMLIT

        if not choice:
            return FRONTEND_STREAMLIT
        if choice == "0":
            return None
        frontend = normalize_frontend(choice)
        if frontend:
            return frontend
        print("⚠️  选项无效，请输入 1、2 或 0。")


def _find_npm():
    """查找当前系统可用的 npm 命令。"""
    return shutil.which("npm.cmd") or shutil.which("npm")


def _backend_base_url():
    """返回前后端共用的后端地址，优先采用显式 API_BASE。"""
    configured = os.getenv("API_BASE", "").strip()
    if configured:
        return configured.rstrip("/")
    port = os.getenv("PORT", "18001").strip() or "18001"
    return f"http://localhost:{port}"


def start_streamlit_web():
    """启动原有 Streamlit 前端。"""
    print("🌐 启动 Streamlit 前端...")
    print("  请在浏览器中访问: http://localhost:8501")
    print("  按 Ctrl+C 停止服务\n")
    main_path = os.path.join(PROJECT_ROOT, "app", "main.py")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", main_path],
            cwd=PROJECT_ROOT,
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n👋 Streamlit 前端已停止")
        return True


def start_next_web():
    """启动 frontend_next 中的 React / Vinext 前端。"""
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend_next")
    package_json = os.path.join(frontend_dir, "package.json")
    if not os.path.exists(package_json):
        print(f"❌ 未找到新前端: {package_json}")
        return False

    npm = _find_npm()
    if not npm:
        print("❌ 未找到 npm，请先安装 Node.js 22.13.0 或更高版本。")
        return False
    if not os.path.isdir(os.path.join(frontend_dir, "node_modules")):
        print("❌ 新前端依赖尚未安装。")
        print(f'请先运行: cd "{frontend_dir}" && npm install')
        return False

    backend_base = _backend_base_url()
    print("🌐 启动 React / Vinext 前端...")
    print("  浏览器地址请以启动日志中的 Local 地址为准")
    print(f"  前端将连接 FastAPI: {backend_base}")
    print("  按 Ctrl+C 停止服务\n")
    try:
        frontend_env = os.environ.copy()
        frontend_env.setdefault("NEXT_PUBLIC_API_BASE", backend_base)
        result = subprocess.run(
            [npm, "run", "dev"], cwd=frontend_dir, env=frontend_env
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n👋 React / Vinext 前端已停止")
        return True


def start_web(frontend=None):
    """选择并启动一个 Web 前端。"""
    selected = normalize_frontend(frontend) if frontend is not None else choose_frontend()
    if frontend is not None and not selected:
        print(f"❌ 不支持的前端: {frontend}")
        print("可选值: streamlit、next")
        return False
    if selected is None:
        print("已取消启动前端。")
        return False
    if selected == FRONTEND_NEXT:
        return start_next_web()
    return start_streamlit_web()


def start_cli():
    """启动 CLI 模式。"""
    print("💻 启动 CLI 命令行模式...\n")
    main_path = os.path.join(PROJECT_ROOT, "app", "main.py")
    subprocess.run([sys.executable, main_path], cwd=PROJECT_ROOT)


def start_backend():
    """启动 FastAPI 后端；需要时同时托管一个独立调度进程。"""
    _ensure_database()
    scheduler_process = _start_scheduler_process()
    try:
        print("启动 FastAPI 后端服务...")
        api_path = os.path.join(PROJECT_ROOT, "app", "api_server.py")
        subprocess.run([sys.executable, api_path], cwd=PROJECT_ROOT)
    finally:
        _terminate_process(scheduler_process, "调度器")


def _scheduler_enabled():
    return os.getenv("ENABLE_SCHEDULER", "false").lower() in (
        "1", "true", "yes", "on"
    )


def _start_scheduler_process():
    """按配置启动独立调度进程。"""
    if not _scheduler_enabled():
        return None
    print("启动独立 APScheduler 调度进程...")
    runner = os.path.join(PROJECT_ROOT, "app", "scheduler_runner.py")
    return subprocess.Popen([sys.executable, runner], cwd=PROJECT_ROOT)


def _terminate_process(process, label):
    if process is None or process.poll() is not None:
        return
    print(f"\n正在停止本次启动的{label}...")
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _backend_is_ready(base_url=None):
    """通过公开健康端点确认农业后端身份，兼容认证和自定义端口。"""
    health_url = f"{(base_url or _backend_base_url()).rstrip('/')}/api/health"
    try:
        with urllib.request.urlopen(health_url, timeout=3) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return (
                payload.get("service") == "agriculture-agent"
                and payload.get("status") == "ok"
            )
    except (json.JSONDecodeError, urllib.error.URLError, OSError, ValueError):
        return False


def _wait_for_backend(process, timeout=30, base_url=None):
    """等待后端就绪，并在进程提前退出时停止等待。"""
    print("⏳ 等待后端就绪...", end="", flush=True)
    for _ in range(timeout):
        if _backend_is_ready(base_url):
            print(" ✅")
            return True
        if process is not None and process.poll() is not None:
            print(" ❌")
            print(f"后端进程已退出，退出码: {process.returncode}")
            return False
        print(".", end="", flush=True)
        time.sleep(1)
    print(" ⚠️")
    print("后端可能尚未就绪，仍将继续启动前端。")
    return False


def start_all(frontend=None):
    """启动 FastAPI 后端，并选择一个前端。"""
    selected = normalize_frontend(frontend) if frontend is not None else choose_frontend()
    if frontend is not None and not selected:
        print(f"❌ 不支持的前端: {frontend}")
        print("可选值: streamlit、next")
        return False
    if selected is None:
        print("已取消启动。")
        return False

    _ensure_database()
    backend_base = _backend_base_url()
    backend_process = None
    scheduler_process = None
    try:
        if _backend_is_ready(backend_base):
            print(f"✅ 检测到 FastAPI 后端已在 {backend_base} 运行")
        else:
            print("启动 FastAPI 后端服务...")
            api_path = os.path.join(PROJECT_ROOT, "app", "api_server.py")
            backend_process = subprocess.Popen(
                [sys.executable, api_path], cwd=PROJECT_ROOT
            )
            backend_ready = _wait_for_backend(
                backend_process, base_url=backend_base
            )
            if not backend_ready and backend_process.poll() is not None:
                return False
        scheduler_process = _start_scheduler_process()
        return start_web(selected)
    finally:
        # 只停止本次脚本创建的后端，不影响用户原本运行的后端。
        if backend_process is not None and backend_process.poll() is None:
            _terminate_process(backend_process, "FastAPI 后端")
        _terminate_process(scheduler_process, "调度器")


def show_help():
    """显示帮助信息。"""
    print("=" * 64)
    print("       智能种植规划助手启动工具")
    print("=" * 64)
    print()
    print("用法: python app/start.py [命令] [前端]")
    print()
    print("命令:")
    print("  all [前端]  启动后端 + 所选前端（未指定时交互选择，默认）")
    print("  web [前端]  仅启动所选前端（未指定时交互选择）")
    print("  backend     仅启动 FastAPI 后端")
    print("  scheduler   仅启动独立定时调度器")
    print("  cli         启动命令行交互模式")
    print("  build       仅构建知识库")
    print("  check       检查环境配置")
    print("  help        显示此帮助信息")
    print()
    print("可选前端:")
    print("  streamlit   原有 Streamlit 前端")
    print("  next        frontend_next 中的 React / Vinext 前端")
    print()
    print("示例:")
    print("  python app/start.py                 # 交互选择前端并启动全部")
    print("  python app/start.py all next        # 后端 + 新前端")
    print("  python app/start.py all streamlit   # 后端 + 原前端")
    print("  python app/start.py web next        # 仅启动新前端")


def main():
    """解析命令并启动对应服务。"""
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    frontend = sys.argv[2] if len(sys.argv) > 2 else None

    if command in ("help", "-h", "--help"):
        show_help()
        return 0
    if command == "check":
        return 0 if check_env() else 1
    if command == "build":
        return 0 if build_knowledge_base() else 1
    if command == "backend":
        start_backend()
        return 0
    if command == "scheduler":
        from core.scheduler_service import run_scheduler_forever
        return run_scheduler_forever()
    if command in (FRONTEND_STREAMLIT, FRONTEND_NEXT):
        frontend = command
        command = "web"
    if command not in ("all", "web", "cli"):
        print(f"❌ 未知命令: {command}\n")
        show_help()
        return 2
    if not check_env():
        return 1
    if command == "all":
        return 0 if start_all(frontend) else 1
    if command == "cli":
        if not build_knowledge_base():
            print("\n⚠️  知识库构建失败，但程序仍可运行\n")
        start_cli()
        return 0

    if not build_knowledge_base():
        print("\n⚠️  知识库构建失败，但程序仍可运行\n")
    return 0 if start_web(frontend) else 1


if __name__ == "__main__":
    raise SystemExit(main())
