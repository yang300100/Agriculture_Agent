"""哥哥，巡检任务、复查与告警在重启后也要有据可查。"""

from datetime import datetime, timedelta
from uuid import uuid4

from core.autonomy_runtime import (
    acknowledge_alert, dispatch_alerts, list_alerts, list_jobs, process_due_jobs,
    raise_alert, record_heartbeat, runtime_status, schedule_action,
)


def test_缺少执行时间的计划不立即启动且重复计划只有一份():
    username = 'runtime_' + uuid4().hex
    action = {'action': 'irrigate', 'device_id': 'pump', 'params': {'duration': 5}}
    first = schedule_action(username, '1', action)
    assert schedule_action(username, '1', action)['id'] == first['id']
    assert list_jobs(username)[0]['status'] == 'awaiting_schedule'
    assert process_due_jobs(username=username, now=datetime.now(), handlers={'planned_action': lambda job: {'success': True}}) == []


def test_领取重试与完成状态持久化且不会重复执行():
    username = 'runtime_' + uuid4().hex
    now = datetime.now()
    row = schedule_action(username, '1', {'action': 'light', 'device_id': 'lamp'}, due_at=now)
    failed = process_due_jobs(username=username, now=now, handlers={'planned_action': lambda job: {'success': False, 'error': '暂时不可用'}})
    assert any(j['id'] == row['id'] and j['status'] == 'pending' for j in failed)
    saved = list_jobs(username)[0]
    assert saved['attempts'] == 1 and saved['last_error'] == '暂时不可用'
    assert process_due_jobs(username=username, now=now, handlers={'planned_action': lambda job: {'success': True}}) == []
    process_due_jobs(username=username, now=now + timedelta(minutes=5), handlers={'planned_action': lambda job: {'success': True}})
    assert list_jobs(username)[0]['status'] == 'completed'
    assert process_due_jobs(username=username, now=now + timedelta(minutes=10), handlers={'planned_action': lambda job: {'success': True}}) == []


def test_通知未配置不能记为已送达且人工确认按用户隔离(monkeypatch):
    username = 'runtime_' + uuid4().hex
    monkeypatch.delenv('AUTONOMY_ALERT_WEBHOOK', raising=False)
    row = raise_alert(username, '1', '灌溉效果需要复核', dedup_key='one')
    assert raise_alert(username, '1', '灌溉效果需要复核', dedup_key='one')['id'] == row['id']
    dispatch_alerts(username=username, sender=lambda alert: {'success': False, 'error': '未配置通知渠道'})
    saved = list_alerts(username)[0]
    assert saved['status'] == 'pending' and saved['delivered_at'] is None
    assert saved['last_error'] == '未配置通知渠道'
    assert acknowledge_alert('other_user', row['id']) is False
    assert acknowledge_alert(username, row['id']) is True
    assert list_alerts(username)[0]['status'] == 'acknowledged'


def test_送达和人工确认是两个独立状态():
    username = 'runtime_' + uuid4().hex
    now = datetime.now()
    raise_alert(username, '1', '设备失联', dedup_key='offline')
    dispatch_alerts(username=username, now=now + timedelta(seconds=1), sender=lambda alert: {'success': True})
    saved = list_alerts(username)[0]
    assert saved['status'] == 'delivered' and saved['acknowledged_at'] is None


def test_心跳过期与配置开关分别表达():
    now = datetime.now()
    record_heartbeat('running', now=now)
    assert runtime_status(now=now)['running'] is True
    assert runtime_status(now=now + timedelta(minutes=2))['running'] is False
    record_heartbeat('stopped', now=now)
    assert runtime_status(now=now)['running'] is False
