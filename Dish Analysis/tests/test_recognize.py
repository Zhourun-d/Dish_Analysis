"""菜品识别接口测试。

用假模型顶替 analyze.model / analyze.multi_model —— 这里验证的是接口契约与
后处理逻辑（校准、去重、阈值过滤），不是模型本身准不准。
"""

import io

import pytest

from nutrition_lib import NUTRITION_DATA


class _Tensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Box:
    def __init__(self, cls_id, conf):
        self.cls = _Tensor(cls_id)
        self.conf = _Tensor(conf)


def _image():
    return {"image": (io.BytesIO(b"fake-image-bytes"), "dish.jpg")}


def _classifier(cls_id, raw_conf):
    class _Probs:
        top1 = cls_id
        top1conf = _Tensor(raw_conf)

    class Model:
        names = {}

        def __call__(self, source, *args, **kwargs):
            return [type("R", (), {"probs": _Probs(), "boxes": None})()]

    return Model()


def _detector(names, boxes):
    class Model:
        def __call__(self, source, *args, **kwargs):
            return [type("R", (), {"probs": None, "boxes": list(boxes)})()]

    Model.names = names
    return Model()


@pytest.fixture()
def sample_dish(analyze_module):
    """从类别映射里挑一个在营养库中存在的菜品，保证营养链路也被走到。"""
    return next(
        (k, v) for k, v in analyze_module.id_to_name.items() if v in NUTRITION_DATA
    )


def test_recognize_returns_dishes_array(client, analyze_module, monkeypatch, sample_dish):
    cls_id, dish_name = sample_dish
    monkeypatch.setattr(analyze_module, "model", _classifier(cls_id, 0.9))

    body = client.post("/recognize", data=_image(), content_type="multipart/form-data").get_json()

    assert body["success"] is True
    assert len(body["dishes"]) == 1

    dish = body["dishes"][0]
    assert dish["name"] == dish_name
    # 原始 0.9 经校准后固定为 75
    assert dish["probability"] == 75
    assert dish["calories"] == NUTRITION_DATA[dish_name]["calories"]
    assert dish["allergens"] == NUTRITION_DATA[dish_name]["allergens"]


def test_recognize_without_image_returns_400(client):
    resp = client.post("/recognize", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_recognize_multi_returns_count(client, analyze_module, monkeypatch, sample_dish):
    cls_id, dish_name = sample_dish
    monkeypatch.setattr(
        analyze_module, "multi_model", _detector({0: dish_name}, [_Box(cls_id, 0.8)])
    )

    body = client.post(
        "/recognize_multi", data=_image(), content_type="multipart/form-data"
    ).get_json()

    assert body["success"] is True
    assert body["count"] == 1
    assert body["dishes"][0]["name"] == dish_name


def test_recognize_multi_dedupes_same_dish(client, analyze_module, monkeypatch, sample_dish):
    """同一道菜被检出两次，只应保留一条。"""
    cls_id, dish_name = sample_dish
    monkeypatch.setattr(
        analyze_module,
        "multi_model",
        _detector({0: dish_name, 1: dish_name}, [_Box(cls_id, 0.8), _Box(cls_id, 0.7)]),
    )

    body = client.post(
        "/recognize_multi", data=_image(), content_type="multipart/form-data"
    ).get_json()

    assert body["count"] == 1


def test_recognize_multi_filters_low_confidence(client, analyze_module, monkeypatch, sample_dish):
    """校准后低于 30% 的检出应被丢弃；全被丢弃时返回 400。"""
    cls_id, dish_name = sample_dish
    monkeypatch.setattr(
        analyze_module, "multi_model", _detector({0: dish_name}, [_Box(cls_id, 0.1)])
    )

    resp = client.post("/recognize_multi", data=_image(), content_type="multipart/form-data")

    assert resp.status_code == 400
    assert "置信度过低" in resp.get_json()["error"]


def test_recognize_multi_without_detection_returns_400(client, analyze_module, monkeypatch):
    monkeypatch.setattr(analyze_module, "multi_model", _detector({}, []))

    resp = client.post("/recognize_multi", data=_image(), content_type="multipart/form-data")

    assert resp.status_code == 400
    assert "未识别到任何菜品" in resp.get_json()["error"]
