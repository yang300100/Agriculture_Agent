"""哥哥，把巡检后续工作放进数据库，不依赖网页或内存计时器。"""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError

from core.database.engine import get_session
from core.database.models import AutonomyJob, AutonomyAlert


def plain(value):
    if is_dataclass(value):
        return asdict(value)
    return value if isinstance(value, dict) else {}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)


def timestamp(value=None):
    if value is None:
        return datetime.now()
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("执行时间必须是有效日期时间")
    return value.astimezone().replace(tzinfo=None) if value.tzinfo else value


def key_for(*values):
    return hashlib.sha256(dumps(values).encode("utf-8")).hexdigest()


def serialize(row):
    result = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        if column.name in {"payload", "result"} and value:
            value = json.loads(value)
        elif isinstance(value, datetime):
            value = value.isoformat()
        result[column.name] = value
    return result


def enqueue(username, region, kind, payload, due_at=None, dedup_key=None):
    due = timestamp(due_at) if due_at is not None else None
    key = dedup_key or key_for(region, kind, payload, due.isoformat() if due else None)
    with get_session() as session:
        row = session.query(AutonomyJob).filter_by(username=username, dedup_key=key).first()
        if row:
            return serialize(row)
        row = AutonomyJob(username=username, region=str(region), kind=kind, dedup_key=key,
                          payload=dumps(payload), due_at=due,
                          status="pending" if due else "awaiting_schedule", attempts=0,
                          last_error="", updated_at=datetime.now())
        session.add(row)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            row = session.query(AutonomyJob).filter_by(username=username, dedup_key=key).one()
        return serialize(row)


def list_rows(model, username, limit=100):
    with get_session() as session:
        return [serialize(row) for row in session.query(model).filter_by(username=username)
                .order_by(model.id.desc()).limit(max(1, min(int(limit), 200))).all()]


def claim_jobs(now, limit, username=None):
    """原子领取；进程中断后租约到期才允许重新处理。"""
    with get_session() as session:
        eligible = and_(AutonomyJob.due_at <= now, or_(
            AutonomyJob.status == "pending",
            and_(AutonomyJob.status == "executing", AutonomyJob.lease_until <= now),
        ))
        query = session.query(AutonomyJob).filter(eligible)
        if username:
            query = query.filter(AutonomyJob.username == username)
        ids = [row.id for row in query.order_by(AutonomyJob.due_at, AutonomyJob.id).limit(limit).all()]
        claimed = []
        for job_id in ids:
            row = session.get(AutonomyJob, job_id)
            if row.kind == "planned_action" and row.status == "executing":
                # 哥哥，崩溃时无法知道指令是否送达；重启后只能核对，不能重发。
                message = "作业下发期间进程中断，执行结果待核验，未重新下发"
                count = session.query(AutonomyJob).filter(AutonomyJob.id == job_id, eligible).update({
                    "status": "unverified", "lease_until": None, "last_error": message,
                    "result": dumps({"success": False, "terminal": True, "status": "unverified", "error": message}),
                    "updated_at": now,
                }, synchronize_session=False)
                if count:
                    session.commit()
                    session.expire_all()
                    claimed.append(serialize(session.get(AutonomyJob, job_id)))
                continue
            count = session.query(AutonomyJob).filter(AutonomyJob.id == job_id, eligible).update({
                "status": "executing", "lease_until": now + timedelta(minutes=2),
                "attempts": AutonomyJob.attempts + 1, "updated_at": now,
            }, synchronize_session=False)
            if count:
                session.commit()
                session.expire_all()
                claimed.append(serialize(session.get(AutonomyJob, job_id)))
        return claimed


def finish_job(job, result, now):
    result = plain(result)
    success = result.get("success") is True
    terminal = result.get("terminal") is True or job["attempts"] >= 3
    error = str(result.get("error") or result.get("message") or "复查未通过")[:1000]
    status = "completed" if success else "failed" if terminal else "pending"
    if result.get("status") == "unverified":
        status = "unverified"
    deferred = result.get("defer_until")
    if deferred:
        status = "pending"
    due = timestamp(deferred) if deferred else now + timedelta(seconds=min(300, 30 * 2 ** min(job["attempts"], 10)))
    with get_session() as session:
        session.query(AutonomyJob).filter_by(id=job["id"], status="executing").filter(
            AutonomyJob.lease_until == timestamp(job["lease_until"])
        ).update({"status": status, "result": dumps(result), "lease_until": None,
                  "last_error": "" if success else error,
                  "due_at": due, "attempts": max(0, job["attempts"] - 1) if deferred else job["attempts"],
                  "updated_at": now}, synchronize_session=False)
        session.commit()
        return serialize(session.get(AutonomyJob, job["id"]))


def save_alert(username, region, message, dedup_key=None, severity="warning"):
    now = datetime.now()
    key = key_for(region, dedup_key) if dedup_key else key_for(region, message, now.date().isoformat())
    with get_session() as session:
        row = session.query(AutonomyAlert).filter_by(username=username, dedup_key=key).first()
        if row:
            return serialize(row)
        row = AutonomyAlert(username=username, region=str(region), message=str(message), severity=severity,
                            dedup_key=key, status="pending", attempts=0, next_attempt_at=now,
                            last_error="", updated_at=now)
        session.add(row)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            row = session.query(AutonomyAlert).filter_by(username=username, dedup_key=key).one()
        return serialize(row)


def claim_alerts(now, limit=20, username=None):
    with get_session() as session:
        query = session.query(AutonomyAlert).filter(
            AutonomyAlert.status.in_(["pending", "sending"]),
            AutonomyAlert.next_attempt_at <= now,
            or_(AutonomyAlert.lease_until.is_(None), AutonomyAlert.lease_until <= now),
        )
        if username:
            query = query.filter_by(username=username)
        rows = query.order_by(AutonomyAlert.id).limit(limit).all()
        claimed = []
        for row in rows:
            count = session.query(AutonomyAlert).filter_by(id=row.id).filter(
                AutonomyAlert.status.in_(["pending", "sending"]),
                AutonomyAlert.next_attempt_at <= now,
                or_(AutonomyAlert.lease_until.is_(None), AutonomyAlert.lease_until <= now),
            ).update({"status": "sending", "lease_until": now + timedelta(minutes=1),
                      "attempts": AutonomyAlert.attempts + 1, "updated_at": now}, synchronize_session=False)
            if count:
                session.commit()
                session.expire_all()
                claimed.append(serialize(session.get(AutonomyAlert, row.id)))
        return claimed


def finish_alert(alert, result, now):
    success = result.get("success") is True
    with get_session() as session:
        session.query(AutonomyAlert).filter_by(id=alert["id"], status="sending").filter(
            AutonomyAlert.lease_until == timestamp(alert["lease_until"])
        ).update({"status": "delivered" if success else "failed" if alert["attempts"] >= 5 else "pending",
                  "delivered_at": now if success else None, "lease_until": None,
                  "last_error": "" if success else str(result.get("error") or "通知未送达")[:1000],
                  "next_attempt_at": now + timedelta(seconds=min(3600, 30 * 2 ** alert["attempts"])),
                  "updated_at": now}, synchronize_session=False)
        session.commit()
        return serialize(session.get(AutonomyAlert, alert["id"]))
