"""哥哥，对话续存必须保留身份，写入中断要保住原消息。"""

import pytest

from core.chat_history import ChatHistoryStore


def test_续存对话保持原会话ID():
    store = ChatHistoryStore(username="history_stable_id")
    original = [{"role": "user", "content": "查看设备"}]
    sid = store.save_session("", original)
    store.save_session("", [{"role": "user", "content": "另一个会话"}])
    updated = original + [{"role": "assistant", "content": "设备在线"}]
    assert store.save_session(sid, updated) == sid
    assert store.load_session(sid) == updated


def test_消息写入失败回滚并保留原对话():
    store = ChatHistoryStore(username="history_rollback")
    original = [{"role": "user", "content": "原始消息"}]
    sid = store.save_session("", original)
    with pytest.raises(Exception):
        store.save_session(sid, [
            {"role": "user", "content": "新的消息"},
            {"role": "assistant", "content": {"invalid": "object"}},
        ])
    assert store.load_session(sid) == original


def test_他人会话ID不能覆盖原数据():
    alice = ChatHistoryStore(username="history_alice")
    bob = ChatHistoryStore(username="history_bob")
    sid = alice.save_session("", [{"role": "user", "content": "alice"}])
    other_sid = bob.save_session(sid, [{"role": "user", "content": "bob"}])
    assert sid != other_sid
    assert alice.load_session(sid) == [{"role": "user", "content": "alice"}]
