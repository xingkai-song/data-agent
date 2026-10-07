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

from eval.metrics.result_set import (
    ResultSet,
    compare_result_sets,
    numeric_columns_covered,
)


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


# ---------- 允许额外辅助列 ----------


def test_extra_columns_allowed_when_values_match():
    """回答成交额时顺便带上订单数，不该被判错。

    这是跑真实评测时发现的：Agent 多带一列辅助信息，
    严格的列数检查会把答对的题判成错的。
    """
    golden = rs(["gmv"], [[100.0]])
    candidate = rs(["gmv", "orders"], [[100.0, 5]])
    assert not compare_result_sets(golden, candidate).matched
    assert compare_result_sets(golden, candidate, allow_extra_columns=True).matched


def test_extra_columns_do_not_mask_wrong_main_metric():
    """宽松归宽松，主指标算错必须照样判错。

    否则这个口径就没有任何约束力了。
    """
    golden = rs(["gmv"], [[100.0]])
    candidate = rs(["gmv_by_create", "orders"], [[88.0, 5]])
    res = compare_result_sets(golden, candidate, allow_extra_columns=True)
    assert not res.matched
    assert "找不到" in res.reason


def test_extra_columns_on_multirow_result():
    golden = rs(["city", "gmv"], [["上海", 100.0], ["北京", 200.0]])
    candidate = rs(
        ["city", "gmv", "orders"], [["上海", 100.0, 3], ["北京", 200.0, 7]]
    )
    assert compare_result_sets(golden, candidate, allow_extra_columns=True).matched


def test_extra_columns_with_wrong_rows_still_fails():
    golden = rs(["city", "gmv"], [["上海", 100.0], ["北京", 200.0]])
    candidate = rs(
        ["city", "gmv", "orders"], [["上海", 100.0, 3], ["北京", 999.0, 7]]
    )
    assert not compare_result_sets(golden, candidate, allow_extra_columns=True).matched


# ---------- 形状无关的数值一致性 ----------


def test_numeric_covered_despite_label_representation():
    """布尔标签和字符串标签表示同一件事，不该判错。

    真实案例：标准用 is_vip 布尔值，Agent 用 user_type='VIP'/'普通用户'。
    """
    golden = rs(["is_vip", "aov"], [[True, 3633.59], [False, 3670.20]])
    candidate = rs(
        ["user_type", "aov_by_order"],
        [["VIP", 3633.59], ["普通用户", 3670.20]],
    )
    ok, missing = numeric_columns_covered(golden, candidate)
    assert ok, missing


def test_numeric_not_covered_when_numbers_differ():
    golden = rs(["is_vip", "aov"], [[True, 3633.59], [False, 3670.20]])
    candidate = rs(["user_type", "aov"], [["VIP", 1.0], ["普通用户", 2.0]])
    ok, missing = numeric_columns_covered(golden, candidate)
    assert not ok
    assert missing == ["aov"]


def test_numeric_ignores_non_numeric_golden_columns():
    """标签列不参与数值一致性检查。"""
    golden = rs(["reason", "amount"], [["七天无理由", 100.0]])
    candidate = rs(["why", "amt"], [["其他说法", 100.0]])
    ok, _ = numeric_columns_covered(golden, candidate)
    assert ok


def test_numeric_requires_all_numeric_columns_covered():
    golden = rs(["gmv_sep", "gmv_aug"], [[100.0, 80.0]])
    candidate = rs(["channel", "ym", "gmv"], [["a", "09", 100.0], ["a", "08", 80.0]])
    ok, missing = numeric_columns_covered(golden, candidate)
    assert not ok
    assert set(missing) == {"gmv_sep", "gmv_aug"}


def test_numeric_partial_coverage_fails():
    golden = rs(["a", "b"], [[1.0, 2.0]])
    candidate = rs(["x"], [[1.0]])
    ok, missing = numeric_columns_covered(golden, candidate)
    assert not ok
    assert missing == ["b"]
