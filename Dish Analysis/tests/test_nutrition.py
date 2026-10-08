"""营养数据库测试。"""

from nutrition_lib import (
    NUTRITION_DATA,
    get_all_dishes,
    get_nutrition,
    get_nutrition_multi,
)


def test_lookup_known_dish():
    data = get_nutrition("宫保鸡丁")
    assert data["calories"] == 160
    assert data["protein"] == 15
    assert data["allergens"] == "坚果(花生)"


def test_lookup_unknown_dish_returns_none():
    assert get_nutrition("这道菜根本不存在") is None


def test_result_is_a_copy_not_the_source():
    """返回副本，调用方改坏它不应污染全局营养库。"""
    data = get_nutrition("宫保鸡丁")
    data["calories"] = 9999
    assert NUTRITION_DATA["宫保鸡丁"]["calories"] == 160


def test_multi_library_lookup():
    assert get_nutrition_multi("炒饭")["calories"] == 180


def test_multi_library_unknown_returns_none():
    assert get_nutrition_multi("这道菜根本不存在") is None


def test_all_dishes_matches_library_size():
    dishes = get_all_dishes()
    assert len(dishes) == len(NUTRITION_DATA)
    assert "宫保鸡丁" in dishes
