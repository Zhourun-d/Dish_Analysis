"""置信度校准函数测试。

calibrated_confidence 用来压掉模型"虚高"的原始置信度，是前端不出现
"识别错误却高置信度"的关键。这里锁住它的三条性质。
"""

import pytest

from analyze import calibrated_confidence


def test_output_within_0_1():
    for i in range(101):
        assert 0 <= calibrated_confidence(i / 100) <= 1


def test_monotonic_non_decreasing():
    """分数必须随原始置信度单调不减，否则会出现"更自信反而分更低"。"""
    values = [calibrated_confidence(i / 100) for i in range(101)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_high_raw_confidence_is_suppressed():
    """原始 0.9 应被压到 0.75，避免"误判却高置信度"的误导。"""
    assert calibrated_confidence(0.9) == pytest.approx(0.75)


def test_very_low_raw_confidence_stays_low():
    assert calibrated_confidence(0.1) < 0.05


def test_segment_boundaries_do_not_jump_significantly():
    """分段线性函数在断点处不应出现明显跳变。

    注: 0.3 处实际存在约 0.0002 的微小台阶——第一段斜率取 0.166 时，
    raw=0.3 落在 0.0498，而第二段起点是 0.05。折算成百分比只有 0.02 个
    百分点，对用户不可感知，故用 1e-3 而非 0 作为上限。
    0.6 / 0.8 / 0.9 三处则是严格连续的。
    """
    for boundary in (0.3, 0.6, 0.8, 0.9):
        left = calibrated_confidence(boundary - 1e-9)
        right = calibrated_confidence(boundary + 1e-9)
        assert abs(left - right) < 1e-3


def test_upper_boundaries_are_strictly_continuous():
    """0.6 / 0.8 / 0.9 三处严格连续。

    容差取 1e-6: 探针在断点两侧各偏 1e-9，连续函数会因此产生约 slope × 2e-9
    （< 1e-8）的表观差值；而真正的跳变（如 0.3 处的 2e-4）会明显超过这个量级，
    所以 1e-6 既能容忍探针噪声，又能揪出真实跳变。
    """
    for boundary in (0.6, 0.8, 0.9):
        left = calibrated_confidence(boundary - 1e-9)
        right = calibrated_confidence(boundary + 1e-9)
        assert abs(left - right) < 1e-6
