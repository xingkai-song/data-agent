"""结果集比对器的单元测试。

比对器是评测体系的地基：如果它会误判，那么所有指标数字都是假的。
所以这里既测"该通过的通过"，也测"该拦住的拦住"。
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics.result_set import ResultSet, compare_result_sets


def rs(cols, rows):
    return ResultSet(columns=cols, rows=[tuple(r) for r in rows])


def test_identical_passes():
    a = rs(["gmv"], [[123.45]])
    assert compare_result_sets(a, a).matched


def test_decimal_and_float_are_equal():
    a = rs(["gmv"], [[Decimal("123.45")]])
    b = rs(["gmv"], [[123.45]])
    assert compare_result_sets(a, b).matched


def test_float_within_tolerance_passes():
    a = rs(["v"], [[100.00]])
    b = rs(["v"], [[100.005]])
    assert compare_result_sets(a, b, abs_tol=0.01).matched


def test_float_beyond_tolerance_fails():
    a = rs(["v"], [[100.00]])
    b = rs(["v"], [[100.50]])
    res = compare_result_sets(a, b, abs_tol=0.01)
    assert not res.matched
    assert "不一致" in res.reason


def test_row_order_does_not_matter_by_default():
    a = rs(["c", "v"], [["上海", 1], ["北京", 2]])
    b = rs(["c", "v"], [["北京", 2], ["上海", 1]])
    assert compare_result_sets(a, b).matched


def test_row_order_matters_when_required():
    a = rs(["c"], [["a"], ["b"]])
    b = rs(["c"], [["b"], ["a"]])
    assert not compare_result_sets(a, b, order_matters=True).matched


def test_row_count_mismatch_fails():
    a = rs(["c"], [["a"], ["b"]])
    b = rs(["c"], [["a"]])
    res = compare_result_sets(a, b)
    assert not res.matched
    assert "行数不一致" in res.reason


def test_column_count_mismatch_fails():
    a = rs(["a", "b"], [[1, 2]])
    b = rs(["a"], [[1]])
    res = compare_result_sets(a, b)
    assert not res.matched
    assert "列数不一致" in res.reason


def test_null_equals_null():
    a = rs(["c"], [[None]])
    b = rs(["c"], [[None]])
    assert compare_result_sets(a, b).matched


def test_null_versus_zero_fails():
    """这一条很重要：NULL 和 0 在业务上完全不同，不能被判为相等。"""
    a = rs(["c"], [[None]])
    b = rs(["c"], [[0]])
    assert not compare_result_sets(a, b).matched


def test_string_whitespace_is_ignored():
    a = rs(["c"], [[" 上海 "]])
    b = rs(["c"], [["上海"]])
    assert compare_result_sets(a, b).matched


def test_multi_cell_mismatch_reports_details():
    a = rs(["c", "v"], [["上海", 100], ["北京", 200]])
    b = rs(["c", "v"], [["上海", 999], ["北京", 888]])
    res = compare_result_sets(a, b)
    assert not res.matched
    assert len(res.mismatches) == 2
    assert "上海" in res.mismatches[0]


def test_large_value_small_relative_difference_is_caught():
    """回归测试：容差曾经设成 0.5%，导致 400 万金额差 2 万也能通过。

    对"精确比对"而言这是致命的——大额指标的实质性差异会被放过。
    所以相对容差必须收紧到接近 0，只保留一分钱的绝对容差。
    """
    a = rs(["gmv"], [[4_000_000.00]])
    b = rs(["gmv"], [[4_020_000.00]])
    assert not compare_result_sets(a, b).matched


def test_tolerance_can_be_relaxed_per_case():
    """确实允许误差的指标应该在任务级单独指定，而不是全局放松。"""
    a = rs(["rate"], [[0.8613]])
    b = rs(["rate"], [[0.8650]])
    assert not compare_result_sets(a, b).matched
    assert compare_result_sets(a, b, abs_tol=0.01).matched
