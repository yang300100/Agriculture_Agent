"""独立调度进程入口。"""

import logging

from dotenv import load_dotenv

from core.scheduler_service import run_scheduler_forever


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    raise SystemExit(run_scheduler_forever())
