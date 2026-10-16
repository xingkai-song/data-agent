"""多路径对算的测试：用编译路径复核手写 SQL 的口径。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.semantic import load_semantic_layer
from data_agent.semantic.multipath import cross_check


@pytest.fixture(scope="module")
def layer():
    return load_semantic_layer()


@pytest.fixture
def execute():
    from data_agent.infra.db import Warehouse

    with Warehouse() as wh:
        yield lambda s: wh.query(s)


def _run(layer, execute, sql, columns=None):
    from data_agent.infra.db import Warehouse

    with Warehouse() as wh:
        cols = columns or wh.columns_of(sql)
        rows = [list(r) for r in wh.query(sql)]
    return cross_check(layer, sql, list(cols), rows, execute=execute)


WINDOW = "o.pay_time >= '2026-09-01' AND o.pay_time < '2026-10-01'"
FILTER = "o.status IN ('paid','refunded')"
NET = (
    "COALESCE((SELECT SUM(r.refund_amount) FROM refunds r "
    "JOIN orders x ON r.order_id = x.order_id "
    "WHERE x.pay_time >= '2026-09-01' AND x.pay_time < '2026-10-01'), 0)"
)


def test_correct_scalar_sql_passes(layer, execute):
    sql = f"SELECT SUM(o.amount) AS gmv FROM orders o WHERE {FILTER} AND {WINDOW}"
    assert _run(layer, execute, sql) == []


def test_substituted_caliber_is_flagged(layer, execute):
    """列名写 gmv，算的却是"扣掉退款后的金额"——这正是静默错误。"""
    sql = f"SELECT SUM(o.amount) - {NET} AS gmv FROM orders o WHERE {FILTER} AND {WINDOW}"
    warnings = _run(layer, execute, sql)
    assert warnings and "对不上" in warnings[0]


def test_ratio_with_wrong_numerator_is_flagged(layer, execute):
    sql = (
        f"SELECT (SUM(o.amount) - {NET}) / COUNT(DISTINCT o.order_id) AS aov "
        f"FROM orders o WHERE {FILTER} AND {WINDOW}"
    )
    warnings = _run(layer, execute, sql)
    assert warnings and "aov" in warnings[0]


def test_correct_ratio_passes(layer, execute):
    sql = (
        "SELECT SUM(o.amount) / COUNT(DISTINCT o.order_id) AS aov "
        f"FROM orders o WHERE {FILTER} AND {WINDOW}"
    )
    assert _run(layer, execute, sql) == []


def test_extra_filter_is_skipped(layer, execute):
    """只算 VIP 的客单价本来就是另一个口径，对照没有意义，不许误报。"""
    sql = (
        "SELECT SUM(o.amount) / COUNT(DISTINCT o.order_id) AS aov "
        "FROM orders o JOIN users u ON u.user_id = o.user_id "
        f"WHERE u.is_vip AND {FILTER} AND {WINDOW}"
    )
    assert _run(layer, execute, sql) == []


def test_grouped_result_is_skipped(layer, execute):
    """编译对照只做标量，分组结果不判定。"""
    sql = (
        "SELECT u.city, SUM(o.amount) AS gmv FROM orders o "
        "JOIN users u ON u.user_id = o.user_id "
        f"WHERE {FILTER} AND {WINDOW} GROUP BY 1"
    )
    assert _run(layer, execute, sql) == []


def test_unknown_column_name_is_ignored(layer, execute):
    sql = f"SELECT SUM(o.amount) AS 随便一个名字 FROM orders o WHERE {FILTER} AND {WINDOW}"
    assert _run(layer, execute, sql) == []
