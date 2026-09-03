"""统一解析项目运行时数据目录。"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def resolve_data_dir() -> str:
    """返回绝对数据目录，空配置回退到项目根目录下的 data。"""
    configured = os.getenv("DATA_STORAGE_DIR", "").strip()
    path = Path(configured) if configured else PROJECT_ROOT / "data"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


DEFAULT_DATA_DIR = resolve_data_dir()
