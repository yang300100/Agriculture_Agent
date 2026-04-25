"""
智能种植规划助手启动脚本
提供便捷的启动和初始化功能
"""

import os
import sys
import subprocess

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def check_env():
    """检查环境配置"""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        print(f"⚠️  未找到 .env 文件 (应在: {env_path})")
        print("请创建 .env 文件并配置 OPENAI_API_KEY")
        print("示例：")
        print("  OPENAI_API_KEY=your_key_here")
        return False
    # 切换到项目根目录运行
    os.chdir(PROJECT_ROOT)
    print(f"✅ 已加载环境配置: {env_path}")
    return True


def build_knowledge_base():
    """构建农业知识库"""
    print("📚 检查农业知识库...")

    index_path = os.path.join(PROJECT_ROOT, "agriculture_faiss_index")
    if not os.path.exists(index_path):
        print("  知识库不存在，开始构建...")
        try:
            build_script = os.path.join(PROJECT_ROOT, "knowledge", "build_agriculture_rag.py")
            subprocess.run([sys.executable, build_script], check=True)
            print("  ✅ 知识库构建完成")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 构建失败: {e}")
            return False
    else:
        print("  ✅ 知识库已存在")

    return True


def start_web():
    """启动Web界面"""
    print("🌐 启动 Streamlit Web 界面...")
    print("  请在浏览器中访问: http://localhost:8501")
    print("  按 Ctrl+C 停止服务\n")

    test1_path = os.path.join(PROJECT_ROOT, "app", "test1.py")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", test1_path])
    except KeyboardInterrupt:
        print("\n👋 服务已停止")


def start_cli():
    """启动CLI模式"""
    print("💻 启动 CLI 命令行模式...\n")
    test1_path = os.path.join(PROJECT_ROOT, "app", "test1.py")
    subprocess.run([sys.executable, test1_path])


def show_help():
    """显示帮助信息"""
    print("=" * 60)
    print("     智能种植规划助手启动工具")
    print("=" * 60)
    print()
    print("用法: python app/start.py [命令]")
    print()
    print("命令:")
    print("  web       启动 Streamlit Web 界面 (默认)")
    print("  cli       启动命令行交互模式")
    print("  build     仅构建知识库")
    print("  check     检查环境配置")
    print("  help      显示此帮助信息")
    print()
    print("示例:")
    print("  python app/start.py          # 启动Web界面")
    print("  python app/start.py web      # 启动Web界面")
    print("  python app/start.py cli      # 启动CLI模式")
    print("  python app/start.py build    # 构建知识库")
    print()


def main():
    # 获取命令行参数
    command = sys.argv[1] if len(sys.argv) > 1 else "web"

    if command == "help" or command == "-h" or command == "--help":
        show_help()
        return

    if command == "check":
        check_env()
        return

    if command == "build":
        build_knowledge_base()
        return

    # 检查环境
    if not check_env():
        return

    # 构建知识库（如果需要）
    if not build_knowledge_base():
        print("\n⚠️  知识库构建失败，但程序仍可运行")
        print("  部分功能可能受限\n")

    # 启动对应模式
    if command == "cli":
        start_cli()
    else:  # 默认启动web
        start_web()


if __name__ == "__main__":
    main()
