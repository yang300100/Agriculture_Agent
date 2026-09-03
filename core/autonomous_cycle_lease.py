"""自主巡检跨进程数据库租约。"""

from datetime import datetime, timedelta

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from core.database.engine import get_session, init_db
from core.database.models import AutonomousCycleLease


def claim_cycle_lease(
    username: str,
    region: str,
    min_interval_minutes: int,
    now: datetime | None = None,
) -> bool:
    """原子领取用户和区域的巡检时间窗。"""
    current = now or datetime.now()
    lease_until = current + timedelta(minutes=max(0, min_interval_minutes))
    init_db()
    session = get_session()
    try:
        updated = session.execute(
            update(AutonomousCycleLease)
            .where(
                AutonomousCycleLease.username == username,
                AutonomousCycleLease.region == region,
                AutonomousCycleLease.lease_until <= current,
            )
            .values(claimed_at=current, lease_until=lease_until)
        )
        if updated.rowcount == 1:
            session.commit()
            return True

        exists = session.query(AutonomousCycleLease.id).filter_by(
            username=username, region=region
        ).first()
        if exists is None:
            session.add(AutonomousCycleLease(
                username=username,
                region=region,
                claimed_at=current,
                lease_until=lease_until,
            ))
        else:
            session.rollback()
            return False
        session.commit()
        return True
    except IntegrityError:
        session.rollback()
        return False
    finally:
        session.close()
