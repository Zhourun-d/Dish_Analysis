"""饮食记录接口测试：保存 / 列表 / 删除。"""

from datetime import datetime

import pytest


def _payload(**overrides):
    body = {
        "userId": "u1",
        "dishName": "宫保鸡丁",
        "probability": 85,
        "calories": 160,
        "protein": 15,
        "carbs": 8,
        "fat": 8,
        "allergens": "坚果(花生)",
        "tips": "酸甜微辣，花生酥脆",
        "imageUrl": "http://tmp/a.jpg",
        "recordedAt": "2026-06-19 12:30",
    }
    body.update(overrides)
    return body


def _save(client, **overrides):
    return client.post("/api/record/save", json=_payload(**overrides))


def _list(client, user="u1"):
    return client.get(f"/api/record/list?userId={user}").get_json()


def test_save_returns_new_record_id(client):
    body = _save(client).get_json()
    assert body["success"] is True
    assert body["recordId"] > 0


def test_save_without_data_returns_400(client):
    resp = client.post("/api/record/save", json={})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_saved_record_can_be_listed_back(client):
    _save(client)
    body = _list(client)
    assert body["success"] is True
    assert len(body["records"]) == 1

    record = body["records"][0]
    assert record["dish_name"] == "宫保鸡丁"
    assert record["calories"] == 160
    assert record["probability"] == 85
    assert record["allergens"] == "坚果(花生)"
    assert record["recorded_at"] == "2026-06-19 12:30"


def test_list_filters_by_user(client):
    _save(client, userId="u1")
    _save(client, userId="u2", dishName="红烧肉")

    assert [r["dish_name"] for r in _list(client, "u1")["records"]] == ["宫保鸡丁"]
    assert [r["dish_name"] for r in _list(client, "u2")["records"]] == ["红烧肉"]


def test_list_is_ordered_by_time_desc(client):
    _save(client, recordedAt="2026-06-18 12:00", dishName="早的那条")
    _save(client, recordedAt="2026-06-19 12:00", dishName="晚的那条")

    assert [r["dish_name"] for r in _list(client)["records"]] == ["晚的那条", "早的那条"]


@pytest.mark.parametrize("bad_time", ["", "2026/06/19 12:30", "昨天"])
def test_invalid_recorded_at_falls_back_to_now(client, bad_time):
    """格式不对或缺省时，应落到当前时间而不是报错。"""
    _save(client, recordedAt=bad_time, dishName="兜底时间")
    record = _list(client)["records"][0]
    assert record["recorded_at"].startswith(datetime.now().strftime("%Y-%m-%d"))


def test_delete_removes_the_record(client):
    record_id = _save(client).get_json()["recordId"]
    assert client.delete("/api/record/delete", json={"id": record_id}).get_json()["success"]
    assert _list(client)["records"] == []


def test_delete_without_id_returns_400(client):
    assert client.delete("/api/record/delete", json={}).status_code == 400


def test_delete_unknown_id_returns_404(client):
    assert client.delete("/api/record/delete", json={"id": 999999}).status_code == 404
