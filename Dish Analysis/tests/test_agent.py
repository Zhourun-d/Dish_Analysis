"""智能体层测试：本地过敏判断、提示词构建、AI 调用失败时的降级。"""

import ast
import pathlib

import pytest

import agent


# ---------------------------------------------------------------- 过敏判断（纯本地逻辑）

def test_allergy_returns_clear_when_dish_has_no_allergen():
    assert "不含常见过敏原" in agent.call_xiaoan_for_allergy("炒豆芽", ["花生"])


def test_allergy_warns_on_matching_allergen():
    message = agent.call_xiaoan_for_allergy("宫保鸡丁", ["花生"])
    assert "警告" in message
    assert "花生" in message


def test_allergy_reports_no_conflict_for_other_allergen():
    assert "未检测到" in agent.call_xiaoan_for_allergy("宫保鸡丁", ["虾"])


def test_allergy_handles_unknown_dish():
    assert "暂未找到" in agent.call_xiaoan_for_allergy("这道菜根本不存在", ["花生"])


def test_allergy_handles_user_without_allergens():
    assert "暂未设置过敏原" in agent.call_xiaoan_for_allergy("宫保鸡丁", [])


# ---------------------------------------------------------------- 提示词构建

def test_xiaowu_prompt_carries_dish_and_profile():
    prompt = agent.build_xiaowu_prompt(
        {"name": "宫保鸡丁", "calories": 160, "protein": 15, "carbs": 8, "fat": 8,
         "allergens": "坚果(花生)"},
        {"bmi": "22.5", "bmiStatus": "正常", "activity": "中", "goal": "减脂",
         "allergens": ["花生"], "recent_records": []},
    )
    for expected in ("宫保鸡丁", "160", "22.5", "减脂", "花生"):
        assert expected in prompt


def test_xiaowu_prompt_includes_recent_records():
    prompt = agent.build_xiaowu_prompt(
        {"name": "宫保鸡丁", "calories": 160, "protein": 15, "carbs": 8, "fat": 8,
         "allergens": "无"},
        {"bmi": "22.5", "goal": "减脂", "allergens": [],
         "recent_records": [{"recorded_at": "2026-06-19 12:30",
                             "dish_name": "红烧肉", "calories": 380}]},
    )
    assert "红烧肉" in prompt
    assert "2026-06-19 12:30" in prompt


def test_xiaopan_prompt_carries_goal_and_allergens():
    prompt = agent.build_xiaopan_prompt("22.5", "正常", "中", "增肌", ["虾"], None)
    for expected in ("22.5", "增肌", "虾"):
        assert expected in prompt


# ---------------------------------------------------------------- 降级容错

def test_xiaowu_falls_back_to_local_data_when_api_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise Exception("模拟 DeepSeek 超时")

    monkeypatch.setattr(agent, "call_deepseek", boom)

    message = agent.call_xiaowu_for_nutrition(
        "宫保鸡丁", {"bmi": "22.5", "goal": "减脂", "allergens": []}
    )

    assert "宫保鸡丁" in message
    assert "160" in message


def test_xiaowu_reports_unknown_dish_without_calling_api(monkeypatch):
    def should_not_be_called(*args, **kwargs):
        raise AssertionError("未收录的菜品不应调用 API")

    monkeypatch.setattr(agent, "call_deepseek", should_not_be_called)

    assert "暂未收录" in agent.call_xiaowu_for_nutrition("这道菜根本不存在", {})


def test_xiaoji_handles_user_without_records(temp_db):
    assert "没有饮食记录" in agent.call_xiaoji_for_records("nobody", days=7)


# ---------------------------------------------------------------- 回归防护

def test_agent_does_not_import_analyze_at_module_level():
    """回归防护: agent.py 顶层若再导入 analyze，就会与 analyze.py 的 import agent
    形成循环，导致 gunicorn 以 analyze:app 加载时抛 ImportError、容器无法启动。"""
    tree = ast.parse(pathlib.Path(agent.__file__).read_text(encoding="utf-8"))

    for node in tree.body:  # 只看模块顶层
        if isinstance(node, ast.ImportFrom) and node.module == "analyze":
            pytest.fail("agent.py 顶层又导入了 analyze，会与 analyze.py 形成循环导入")
        if isinstance(node, ast.Import):
            assert all(alias.name != "analyze" for alias in node.names)
