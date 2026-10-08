"""对话接口测试：Function Calling 的两轮编排。

桩掉 requests.post，模拟"AI 先决定调用工具、再综合成最终回复"的完整链路，
不产生任何真实网络请求。
"""

import requests


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _tool_call_response(tool_name, arguments):
    return _FakeResponse({"choices": [{"message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": tool_name, "arguments": arguments},
        }],
    }}]})


def _text_response(text):
    return _FakeResponse({"choices": [{"message": {"role": "assistant", "content": text}}]})


def _user_profile(**overrides):
    profile = {"bmi": "22.5", "activity": "中", "goal": "减脂", "allergens": ["花生"]}
    profile.update(overrides)
    return profile


def test_chat_runs_two_rounds_and_dispatches_tool(client, monkeypatch):
    payloads = []

    def fake_post(url, json=None, **kwargs):
        payloads.append(json)
        if json and json.get("tools"):
            return _tool_call_response("check_allergy_risk", '{"dish_name": "宫保鸡丁"}')
        return _text_response("结论：请避开花生。")

    monkeypatch.setattr(requests, "post", fake_post)

    body = client.post("/api/chat", json={
        "message": "宫保鸡丁我能吃吗",
        "userId": "u1",
        "userProfile": _user_profile(),
        "history": [],
    }).get_json()

    assert body["success"] is True
    assert body["content"] == "结论：请避开花生。"
    assert body["blocks"] == [{"type": "text", "data": "结论：请避开花生。"}]

    # 第一轮带工具、第二轮不带
    assert len(payloads) == 2
    assert payloads[0]["tools"]
    assert payloads[1]["tools"] == []

    # 工具结果里应包含小安的本地判断
    tool_messages = [m for m in payloads[1]["messages"] if m.get("role") == "tool"]
    assert tool_messages, "第二轮应携带工具返回结果"
    assert "警告" in tool_messages[0]["content"]


def test_chat_injects_recent_records_into_prompt(client, monkeypatch):
    """用户近 7 天记录应被强制拼进上下文（本项目的"有记忆"设计）。"""
    payloads = []

    def fake_post(url, json=None, **kwargs):
        payloads.append(json)
        return _text_response("好的。")

    monkeypatch.setattr(requests, "post", fake_post)

    client.post("/api/record/save", json={
        "userId": "u1", "dishName": "红烧肉", "probability": 80,
        "calories": 380, "protein": 12, "carbs": 10, "fat": 32,
        "allergens": "无", "tips": "高脂少吃",
    })

    client.post("/api/chat", json={
        "message": "我最近吃得怎么样",
        "userId": "u1",
        "userProfile": _user_profile(),
        "history": [],
    })

    user_messages = [m for m in payloads[0]["messages"] if m.get("role") == "user"]
    assert any("红烧肉" in m["content"] for m in user_messages)


def test_chat_without_message_returns_400(client):
    resp = client.post("/api/chat", json={"userId": "u1"})
    assert resp.status_code == 400


def test_chat_falls_back_to_reply_when_model_calls_no_tool(client, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _text_response("直接回答"))

    body = client.post("/api/chat", json={
        "message": "你好", "userId": "u1", "userProfile": _user_profile(), "history": [],
    }).get_json()

    assert body["success"] is True
    assert body["content"] == "直接回答"
