"""哥哥，种植批次只能关联自己的地块，旧计划不猜测归属。"""

from uuid import uuid4

import pytest

from core.planting_tracker import PlantingTracker
from core.database.repository.fields import FieldRepository


def test_进度地块绑定可重载且不能绑定其他用户地块(tmp_path):
    tracker = PlantingTracker(str(tmp_path / ('plot_' + uuid4().hex)))
    field = FieldRepository().create(user_id=tracker._uid, name='麦田', coordinates='[]')
    progress = tracker.create_progress({'crop': '小麦', 'plot_id': field.id})
    card = tracker.get_progress_cards()[0]
    assert card['plot_id'] == str(field.id)
    assert tracker.get_progress()[0].plot_id == str(field.id)
    other = PlantingTracker(str(tmp_path / ('plot_' + uuid4().hex)))
    other_field = FieldRepository().create(user_id=other._uid, name='别人的地块', coordinates='[]')
    with pytest.raises(ValueError, match='地块'):
        tracker.update_progress(progress.id, {'plot_id': other_field.id})
    assert tracker.get_progress_cards()[0]['plot_id'] == str(field.id)
    assert tracker.update_progress(progress.id, {'plot_id': None}) is True
    assert tracker.get_progress_cards()[0]['plot_id'] is None


def test_旧记录未绑定时保持空值(tmp_path):
    tracker = PlantingTracker(str(tmp_path / ('plot_' + uuid4().hex)))
    tracker.create_progress({'crop': '小麦'})
    assert tracker.get_progress_cards()[0]['plot_id'] is None


@pytest.fixture
def plot_api():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api_routes import register_routes, _storage_dir
    owner = 'owner_' + uuid4().hex
    other = 'other_' + uuid4().hex
    owner_tracker = PlantingTracker(_storage_dir(owner))
    other_tracker = PlantingTracker(_storage_dir(other))
    field = FieldRepository().create(user_id=owner_tracker._uid, name='麦田', coordinates='[]')
    other_field = FieldRepository().create(user_id=other_tracker._uid, name='他人地块', coordinates='[]')
    app = FastAPI()
    register_routes(app)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, owner, other, field.id, other_field.id


def test_接口创建绑定查询修改解绑并隔离账号(plot_api):
    client, owner, other, field, other_field = plot_api
    path = f'/api/progress?username={owner}'
    assert client.post(path, json={'crop':'小麦', 'plot_id':field}).status_code == 200
    row = client.get(path).json()[0]
    assert row['plot_id'] == str(field)
    pid = row['id']
    assert client.get(f'/api/progress?username={other}').json() == []
    assert client.put(f'/api/progress/{pid}/plot?username={other}', json={'plot_id':other_field}).status_code == 404
    assert client.put(f'/api/progress/{pid}/plot?username={owner}', json={'plot_id':other_field}).status_code == 422
    assert client.get(path).json()[0]['plot_id'] == str(field)
    assert client.put(f'/api/progress/{pid}/plot?username={owner}', json={'plot_id':None}).json()['success'] is True
    assert client.get(path).json()[0]['plot_id'] is None
    assert client.put(f'/api/progress/{pid}/plot?username={owner}', json={'plot_id':field}).json()['success'] is True
    assert client.get(path).json()[0]['plot_id'] == str(field)


@pytest.mark.parametrize('value', ['invalid', 0, -1, True, 999999999])
def test_接口拒绝非法或不存在地块且不写入(plot_api, value):
    client, owner, _, field, _ = plot_api
    path = f'/api/progress?username={owner}'
    assert client.post(path, json={'crop':'小麦', 'plot_id':field}).status_code == 200
    pid = client.get(path).json()[0]['id']
    response = client.post(path, json={'crop':'玉米', 'plot_id':value})
    assert response.status_code == 422
    response = client.put(f'/api/progress/{pid}/plot?username={owner}', json={'plot_id':value})
    assert response.status_code == 422
    rows = client.get(path).json()
    assert len(rows) == 1 and rows[0]['plot_id'] == str(field)


def test_接口拒绝其他用户地块与无效请求体(plot_api):
    client, owner, _, field, other_field = plot_api
    path = f'/api/progress?username={owner}'
    assert client.post(path, json={'crop':'小麦', 'plot_id':other_field}).status_code == 422
    assert client.get(path).json() == []
    assert client.post(path, json={'crop':'小麦', 'plot_id':field}).status_code == 200
    pid = client.get(path).json()[0]['id']
    for body in ({}, {'plot_id':field, 'crop':'玉米'}):
        assert client.put(f'/api/progress/{pid}/plot?username={owner}', json=body).status_code == 422
    assert client.put(f'/api/progress/missing/plot?username={owner}', json={'plot_id':field}).status_code == 404
    assert client.get(path).json()[0]['plot_id'] == str(field)


def test_种植向导请求验证地块编号():
    from pydantic import ValidationError
    from app.api_routes import PlanRequest
    assert PlanRequest(crop='小麦', plot_id=5).plot_id == 5
    assert PlanRequest(crop='小麦', plot_id=None).plot_id is None
    for invalid in ('invalid', 0, -1):
        with pytest.raises(ValidationError):
            PlanRequest(crop='小麦', plot_id=invalid)


@pytest.mark.parametrize('model_name', ['ProgressData', 'PlanRequest'])
def test_请求模型拒绝布尔地块编号(model_name):
    from pydantic import ValidationError
    from app import api_routes
    model = getattr(api_routes, model_name)
    with pytest.raises(ValidationError):
        model(crop='小麦', plot_id=True)
