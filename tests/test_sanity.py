"""结果体检（不变量检查）的测试。"""

from __future__ import annotations

from data_agent.agent.graph import _diagnose
from data_agent.agent.sanity import check_result


def _ds(columns, rows):
    return {"columns": columns, "rows": rows, "row_count": len(rows)}


def test_normal_result_has_no_warning():
    data = _ds(
        ["city", "gmv", "refund_amount", "net_gmv", "paid_order_count"],
        [["上海", 496682.66, 21983.71, 474698.95, 126]],
    )
    assert check_result(data) == []


def test_negative_amount_is_flagged():
    data = _ds(["city", "gmv"], [["上海", -100.0]])
    w = check_result(data)
    assert w and "负值" in w[0]


def test_growth_column_may_be_negative():
    # 「增长量」天然可为负，不该误报
    data = _ds(["city", "gmv_增长量"], [["上海", -5.0]])
    assert check_result(data) == []


def test_ratio_out_of_range_is_flagged():
    data = _ds(["city", "退款率"], [["上海", 1.35]])
    w = check_result(data)
    assert w and "比率" in w[0]


def test_ratio_in_range_is_ok():
    data = _ds(["city", "pay_conversion_rate"], [["上海", 0.8799]])
    assert check_result(data) == []


def test_net_greater_than_gross_is_flagged():
    data = _ds(["city", "gmv", "net_gmv"], [["上海", 100.0, 120.0]])
    w = check_result(data)
    assert w and "净额不应超过总额" in "".join(w)


def test_refund_greater_than_gmv_is_flagged():
    data = _ds(["city", "gmv", "refund_amount"], [["上海", 100.0, 300.0]])
    w = check_result(data)
    assert w and "退款不应超过成交额" in "".join(w)


def test_describe_table_shape_is_ignored():
    # describe_table 的 columns 是字段元数据（字典列表），不是结果集
    data = {
        "table": "orders",
        "columns": [{"name": "amount", "type": "DECIMAL(12,2)", "description": "金额"}],
        "notes": ["-1 不是数值列"],
    }
    assert check_result(data) == []


def test_non_result_payloads_are_ignored():
    assert check_result(None) == []
    assert check_result({}) == []
    assert check_result({"path": "/tmp/x.svg", "bytes_written": -1}) == []
    assert check_result({"columns": ["a"], "rows": []}) == []


def test_warnings_are_capped_and_deduped():
    data = _ds(
        ["gmv", "net_gmv", "退款率", "refund_amount"],
        [[-1.0, 999.0, 88.0, 888.0]],
    )
    w = check_result(data)
    assert len(w) <= 3
    assert len(set(w)) == len(w)


def test_diagnose_surfaces_sanity_warning():
    state = {
        "observations": [
            {"ok": True, "summary": {"row_count": 1}, "warnings": ["列 `gmv` 出现了负值"]}
        ]
    }
    hint = _diagnose(state)
    assert hint and "出现了负值" in hint


def test_diagnose_prefers_empty_over_warning():
    state = {
        "observations": [
            {"ok": True, "summary": {"row_count": 0}, "warnings": ["x"]},
        ]
    }
    hint = _diagnose(state)
    assert hint and "0 行" in hint
