"""指标编译器的测试。

重点锁住四件事：
  1. join 由关系图推导，指标定义里不再手写
  2. 派生指标按业务关系展开，归期由父指标统一
  3. 定义里引用到的表（VIP 指标的 users）会被自动 join
  4. 写错的语义层在加载时就被拦住
"""

from __future__ import annotations

import textwrap

import pytest

from data_agent.semantic.compiler import CompilerError, MetricCompiler
from data_agent.semantic.loader import SemanticLayerError, load_semantic_layer

WINDOW = {"start": "2026-09-01", "end": "2026-10-01"}


@pytest.fixture(scope="module")
def layer():
    return load_semantic_layer()


@pytest.fixture(scope="module")
def compiler(layer):
    return MetricCompiler(layer)


def _sql(compiler, name, **kwargs) -> str:
    return " ".join(compiler.compile(name, **WINDOW, **kwargs).sql.split())


# ----------------------------------------------------------------------
# 基础指标
# ----------------------------------------------------------------------


def test_simple_metric_compiles_to_single_table_query(compiler):
    sql = _sql(compiler, "gmv")
    assert sql.startswith("SELECT SUM(orders.amount) AS gmv")
    assert "FROM orders" in sql
    assert "orders.pay_time >= TIMESTAMP '2026-09-01'" in sql


def test_time_window_is_applied_on_the_anchor_table(compiler):
    """商品销售额的金额在明细表，归期必须落在 orders.pay_time 上。"""
    sql = _sql(compiler, "item_amount")
    assert "FROM order_items" in sql
    assert "JOIN orders ON order_items.order_id = orders.order_id" in sql
    assert "orders.pay_time >= TIMESTAMP '2026-09-01'" in sql
    assert "orders.pay_time < TIMESTAMP '2026-10-01'" in sql


def test_join_is_derived_from_relation_declaration(compiler):
    """item_amount 的 join 是编译器从 tables.yaml 的关系推导的，定义里没写。"""
    metric = compiler.layer.metric("item_amount")
    assert "JOIN" not in (metric.definition or "").upper()
    assert "JOIN orders" in _sql(compiler, "item_amount")


def test_definition_referenced_table_is_joined_automatically(compiler):
    """VIP 成交额的定义引用了 users.is_vip，编译器要自己把 users join 进来。"""
    sql = _sql(compiler, "vip_gmv")
    assert "users.is_vip" in sql
    assert "JOIN users ON orders.user_id = users.user_id" in sql


def test_template_mode_uses_placeholders(compiler):
    """不给时间窗口时输出模板，供提示词使用。"""
    sql = compiler.compile("gmv").sql
    assert ":start" in sql and ":end" in sql


# ----------------------------------------------------------------------
# 派生指标
# ----------------------------------------------------------------------


def test_same_table_derivation_collapses_to_one_scan(compiler):
    """客单价的分子分母都在 orders，应合并成一次扫描。"""
    compiled = compiler.compile("aov", **WINDOW)
    assert compiled.merged is True
    sql = " ".join(compiled.sql.split())
    assert sql.count("FROM orders") == 1
    assert "COUNT(DISTINCT orders.order_id)" in sql
    assert "SUM(orders.amount)" in sql


def test_cross_table_derivation_uses_cte_and_full_join(compiler):
    """净成交额跨 orders 与 refunds，按维度对齐后在外层相减。"""
    compiled = compiler.compile("net_gmv", **WINDOW)
    assert compiled.merged is False
    sql = " ".join(compiled.sql.split())
    assert sql.count("SUM(orders.amount)") == 1
    assert "SUM(refunds.refund_amount)" in sql
    assert "FULL OUTER JOIN" in sql


def test_derived_metric_overrides_child_time_anchor(compiler):
    """退款金额自己按 refund_time 归期；算净成交额时必须改成订单支付时间。"""
    alone = _sql(compiler, "refund_amount")
    assert "refunds.refund_time" in alone

    inside = _sql(compiler, "net_gmv")
    assert "refunds.refund_time" not in inside
    refund_branch = inside.split("net_gmv__d1")[1]
    assert "orders.pay_time" in refund_branch


def test_dimensions_are_aligned_across_branches(compiler):
    compiled = compiler.compile("net_gmv", dimensions=["orders.channel"], **WINDOW)
    sql = " ".join(compiled.sql.split())
    assert "orders.channel AS d0" in sql
    assert "GROUP BY 1" in sql
    assert "COALESCE(net_gmv__d0.d0, net_gmv__d1.d0) AS d0" in sql


def test_nested_derivation_expands_recursively(compiler):
    """VIP 占比 = VIP 成交额 / 成交总额，两层派生都要展开到底。"""
    sql = _sql(compiler, "vip_gmv_share")
    assert "users.is_vip" in sql
    assert "SUM(orders.amount)" in sql
    assert "{vip_gmv}" not in sql and "{gmv}" not in sql


def test_fan_out_risk_is_reported(compiler):
    """从订单走向明细再取商品维度会让订单行数膨胀，编译器要警告。"""
    compiled = compiler.compile("gmv", dimensions=["products.category"], **WINDOW)
    assert any("膨胀" in n for n in compiled.notes)


def test_compiling_unknown_metric_raises(compiler):
    with pytest.raises(CompilerError, match="不存在"):
        compiler.compile("no_such_metric")


# ----------------------------------------------------------------------
# 写错的语义层要被拦住
# ----------------------------------------------------------------------

TABLE_YAML = """
- name: orders
  description: 订单
  grain: 一行一单
  columns:
    - name: order_id
      type: VARCHAR
      description: id
"""


def _write_layer(tmp_path, metrics: str, tables: str = TABLE_YAML):
    d = tmp_path / "definitions"
    d.mkdir()
    (d / "tables.yaml").write_text(textwrap.dedent(tables), encoding="utf-8")
    (d / "metrics.yaml").write_text(textwrap.dedent(metrics), encoding="utf-8")
    (d / "rules.yaml").write_text("[]\n", encoding="utf-8")
    return d


def test_metric_must_declare_exactly_one_of_definition_and_expression(tmp_path):
    d = _write_layer(
        tmp_path,
        """
        - name: bad
          display_name: 坏的
          description: 两个都写了
          definition: COUNT(*)
          expression: "{gmv}"
          base_table: orders
          time_column: order_id
        """,
    )
    with pytest.raises(SemanticLayerError, match="只能声明"):
        load_semantic_layer(d)


def test_unknown_metric_reference_is_rejected(tmp_path):
    d = _write_layer(
        tmp_path,
        """
        - name: bad
          display_name: 坏的
          description: 引用了不存在的指标
          expression: "{nope} / 2"
          base_table: orders
          time_column: order_id
        """,
    )
    with pytest.raises(SemanticLayerError, match="不存在的指标"):
        load_semantic_layer(d)


def test_cycle_in_derived_metrics_is_rejected(tmp_path):
    d = _write_layer(
        tmp_path,
        """
        - name: a
          display_name: A
          description: 指向 b
          expression: "{b} + 1"
          base_table: orders
          time_column: order_id

        - name: b
          display_name: B
          description: 指回 a
          expression: "{a} + 1"
          base_table: orders
          time_column: order_id
        """,
    )
    with pytest.raises(SemanticLayerError, match="循环依赖"):
        load_semantic_layer(d)


def test_relation_to_unknown_table_is_rejected(tmp_path):
    tables = """
    - name: orders
      description: 订单
      grain: 一行一单
      relations:
        - to: nowhere
          "on": orders.order_id = nowhere.id
      columns:
        - name: order_id
          type: VARCHAR
          description: id
    """
    d = _write_layer(
        tmp_path,
        """
        - name: cnt
          display_name: 计数
          description: 计数
          definition: COUNT(*)
          base_table: orders
          time_column: order_id
        """,
        tables=tables,
    )
    with pytest.raises(SemanticLayerError, match="不存在的表"):
        load_semantic_layer(d)
