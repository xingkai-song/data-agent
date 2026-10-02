"""语义层与上下文组装的测试。

这一层对应评测体系里的"组件级评测"：不等 Agent 写完，就能先验证
"检索到的语义信息对不对"。检索错了，后面生成的 SQL 一定是错的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.semantic import load_semantic_layer  # noqa: E402
from data_agent.semantic.context_builder import build_context, select  # noqa: E402
from data_agent.semantic.loader import SemanticLayerError  # noqa: E402


@pytest.fixture(scope="module")
def layer():
    return load_semantic_layer()


# ---------- 加载与校验 ----------


def test_layer_loads(layer):
    assert len(layer.tables) == 6
    assert len(layer.metrics) >= 9
    assert len(layer.rules) >= 8
    assert layer.metric("gmv") is not None
    assert layer.table("orders") is not None


def test_validation_rejects_unknown_base_table(tmp_path):
    (tmp_path / "tables.yaml").write_text(
        yaml.safe_dump([{"name": "t", "description": "d", "grain": "g",
                         "columns": [{"name": "c", "type": "INT", "description": "d"}]}]),
        encoding="utf-8",
    )
    (tmp_path / "metrics.yaml").write_text(
        yaml.safe_dump([{"name": "m", "display_name": "M", "description": "d",
                         "definition": "SUM(1)", "base_table": "not_exist",
                         "time_column": "c"}]),
        encoding="utf-8",
    )
    (tmp_path / "rules.yaml").write_text("[]", encoding="utf-8")

    with pytest.raises(SemanticLayerError, match="base_table"):
        load_semantic_layer(tmp_path)


def test_validation_rejects_unknown_time_column(tmp_path):
    (tmp_path / "tables.yaml").write_text(
        yaml.safe_dump([{"name": "t", "description": "d", "grain": "g",
                         "columns": [{"name": "c", "type": "INT", "description": "d"}]}]),
        encoding="utf-8",
    )
    (tmp_path / "metrics.yaml").write_text(
        yaml.safe_dump([{"name": "m", "display_name": "M", "description": "d",
                         "definition": "SUM(1)", "base_table": "t",
                         "time_column": "not_a_column"}]),
        encoding="utf-8",
    )
    (tmp_path / "rules.yaml").write_text("[]", encoding="utf-8")

    with pytest.raises(SemanticLayerError, match="time_column"):
        load_semantic_layer(tmp_path)


def test_metric_time_table_can_differ_from_base_table(layer):
    """商品销售额按明细表计算，但归期用的是订单的支付时间。"""
    m = layer.metric("item_amount")
    assert m is not None
    assert m.base_table == "order_items"
    assert m.time_table_name() == "orders"
    assert m.time_column == "pay_time"


# ---------- 检索：问题 -> 指标 ----------


@pytest.mark.parametrize(
    "question,expected_metric",
    [
        ("9 月的成交总额是多少", "gmv"),
        ("上个月销售额多少", "gmv"),
        ("扣掉退款之后的净成交额", "net_gmv"),
        ("9 月的净销售额", "net_gmv"),
        ("退款率有多高", "refund_rate"),
        ("订单的支付转化率", "pay_conversion_rate"),
        ("这个月新增了多少用户", "new_users"),
        ("客单价是多少", "aov"),
        ("卖得最好的商品", "item_amount"),
        ("各类目销售额", "item_amount"),
        ("9 月退款金额", "refund_amount"),
    ],
)
def test_metric_retrieval(layer, question, expected_metric):
    sel = select(layer, question)
    assert expected_metric in sel.metric_names(), (
        f"问题「{question}」未检索到指标 {expected_metric}，"
        f"实际命中 {sel.metric_names()}"
    )


def test_gmv_question_pulls_orders_table_and_key_rules(layer):
    sel = select(layer, "9 月的成交总额是多少")
    assert "orders" in sel.table_names()
    rule_ids = {r.id for r in sel.rules}
    # R001 归期字段、R002 金额口径、R003 未支付过滤，都是成交类必须带的
    assert {"R001", "R002", "R003"} <= rule_ids


def test_net_gmv_question_pulls_refunds_table(layer):
    sel = select(layer, "净成交额是多少")
    assert "refunds" in sel.table_names()
    assert "R004" in {r.id for r in sel.rules}


def test_seldom_used_metric_is_not_always_selected(layer):
    """检索应该收敛，不能每次都把全部指标塞进去。"""
    sel = select(layer, "9 月的成交总额是多少")
    assert len(sel.metrics) <= 4


@pytest.mark.parametrize(
    "question,expected",
    [
        ("9 月的成交总额是多少？", ["gmv"]),
        ("扣掉退款之后的净成交额是多少？", ["net_gmv"]),
        ("9 月各个城市的成交额排名是怎样的？", ["gmv"]),
        ("订单的支付转化率有多高？", ["pay_conversion_rate"]),
        ("9 月卖得最好的商品是哪些？", ["item_amount"]),
        ("9 月退款金额最高的原因是什么？", ["refund_amount"]),
    ],
)
def test_metric_selection_is_precise(layer, question, expected):
    """回归测试：检索不仅要"包含正确的指标"，还要"不夹杂无关指标"。

    这里曾经踩过两个坑：
      1. 评分按描述长度归一化，导致描述更短的 net_gmv 在问"成交额"时排第一；
      2. 别名子串匹配没有做长别名优先，问"净成交额"时 gmv 也一起被选中。
    """
    sel = select(layer, question)
    assert sel.metric_names() == expected, (
        f"问题「{question}」期望只命中 {expected}，实际 {sel.metric_names()}"
    )


def test_required_tables_always_accompany_their_metric(layer):
    """选中需要 join 的指标时，join 依赖的表必须一起被带上。"""
    sel = select(layer, "扣掉退款之后的净成交额是多少？")
    assert "net_gmv" in sel.metric_names()
    assert {"orders", "refunds"} <= set(sel.table_names())


# ---------- 上下文渲染 ----------


def test_context_contains_the_trap_warnings(layer):
    """上下文里必须带上"坑"，否则模型会踩。"""
    _, ctx = build_context(layer, "9 月的成交总额是多少")
    assert "pay_time" in ctx
    assert "不要用 create_time" in ctx
    assert "已扣除折扣" in ctx or "折扣" in ctx
    assert "R001" in ctx


def test_context_for_item_amount_mentions_discount(layer):
    _, ctx = build_context(layer, "9 月卖得最好的商品")
    assert "discount" in ctx
    assert "orders.pay_time" in ctx or "pay_time" in ctx


def test_empty_question_does_not_crash(layer):
    sel, ctx = select(layer, ""), None
    assert isinstance(sel.table_names(), list)
