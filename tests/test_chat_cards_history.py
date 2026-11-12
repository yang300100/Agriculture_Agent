"""业务卡片引用要随会话保存，不能跨用户恢复设备或任务。"""

from uuid import uuid4

from sqlalchemy import create_engine, inspect, text

from core.chat_history import ChatHistoryStore
from core.database.migrations import apply_migrations
from core.device_registry_factory import save_custom_devices


def test_会话恢复卡片引用但不保存旧状态和任意控制参数():
    username = 'cards_history_' + uuid4().hex
    store = ChatHistoryStore(username=username)
    save_custom_devices(username, [{'device_id': 'owned-pump', 'name': '麦田灌溉泵',
        'driver': 'simulator', 'capabilities': ['irrigate']}])
    messages = [{'role': 'assistant', 'content': '**麦田灌溉泵**', 'cards': [
        {'type': 'device', 'device_id': 'owned-pump', 'state': {'status': 'running'},
         'params': {'duration': 999999}, 'unexpected': 'discard'}]}]
    sid = store.save_session('', messages)
    reloaded = ChatHistoryStore(username=username).load_session(sid)
    assert reloaded == [{'role': 'assistant', 'content': '**麦田灌溉泵**',
        'cards': [{'type': 'device', 'device_id': 'owned-pump'}]}]


def test_伪造他人设备卡与用户消息附带的操作卡不会恢复():
    other = 'cards_other_' + uuid4().hex
    ChatHistoryStore(username=other)
    save_custom_devices(other, [{'device_id': 'private-pump', 'name': '他人设备',
        'driver': 'simulator', 'capabilities': ['irrigate']}])
    store = ChatHistoryStore(username='cards_history_' + uuid4().hex)
    sid = store.save_session('', [
        {'role': 'assistant', 'content': '查询', 'cards': [{'type': 'device', 'device_id': 'private-pump'}]},
        {'role': 'user', 'content': '用户引用不是操作授权', 'cards': [{'type': 'device', 'device_id': 'private-pump'}]},
    ])
    assert all(not message.get('cards') for message in store.load_session(sid))


def test_旧消息仍按原格式返回且设备被移除时保留正文():
    username = 'cards_history_' + uuid4().hex
    store = ChatHistoryStore(username=username)
    plain = [{'role': 'user', 'content': '你好'}]
    assert store.load_session(store.save_session('', plain)) == plain
    save_custom_devices(username, [{'device_id': 'removed', 'name': '临时泵', 'driver': 'simulator', 'capabilities': ['irrigate']}])
    sid = store.save_session('', [{'role': 'assistant', 'content': '原始说明', 'cards': [{'type': 'device', 'device_id': 'removed'}]}])
    save_custom_devices(username, [])
    restored = store.load_session(sid)
    assert restored[0]['content'] == '原始说明' and not restored[0].get('cards')


def test_旧聊天表迁移保留正文且可重复执行(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old-chat.db'}")
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, content TEXT)'))
        connection.execute(text("INSERT INTO chat_messages VALUES (1, '原始对话')"))
    apply_migrations(engine)
    apply_migrations(engine)
    assert 'cards_json' in {column['name'] for column in inspect(engine).get_columns('chat_messages')}
    with engine.connect() as connection:
        assert connection.execute(text('SELECT content, cards_json FROM chat_messages')).one() == ('原始对话', None)
    engine.dispose()
